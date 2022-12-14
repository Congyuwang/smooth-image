use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::gpu_metal::GpuLib;
use crate::inpaint_worker::CsrMatrixF16;
use half::f16;
use metal::{Buffer, ComputePipelineDescriptor, ComputePipelineState, MTLResourceOptions};

/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked<CB: FnMut(i32, f32, f32)>(
    b_mat: &CsrMatrixF16,
    c: Buffer,
    layer: Buffer,
    x: Buffer,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
    gpu: &GpuLib,
) -> (Vec<u8>, i32) {
    let size = b_mat.nrows();
    let (arg, data) = gpu.init_cg_argument(b_mat, c, layer, x);
    let mut iter_round = 0;
    let queue = gpu.new_queue();
    let mut command = gpu.iter_cg(&arg, &data, &queue, size as u64);
    command.commit();
    loop {
        // running GPU and CPU in parallel
        let new_command = gpu.iter_cg(&arg, &data, &queue, size as u64);
        command.wait_until_completed();

        let r_new_norm = unsafe { *(data[0].contents() as *mut f32) }.sqrt();
        let diff_squared = unsafe { *(data[3].contents() as *mut f32) };

        // check return condition
        if r_new_norm <= tol {
            let queue = gpu.new_queue();
            let cmd = queue.new_command_buffer();
            let x_data = data[6];
            let new_x_buffer = gpu
                .device()
                .new_buffer(x_data.length(), MTLResourceOptions::StorageModeShared);
            gpu.copy_from_buffer(x_data, &new_x_buffer, cmd);
            cmd.commit();
            cmd.wait_until_completed();
            let result = slice_from_raw_parts(x_data.contents() as *const f16, size).iter().map(|f| {
                (f.to_f32() * 256.0f32) as u8
            }).collect::<Vec<_>>();
            return (result, iter_round);
        }

        new_command.commit();
        command = new_command;

        // metric callback and return condition
        if metric_step > 0 && iter_round % metric_step == 0 {
            metric_cb(iter_round, diff_squared, r_new_norm);
        }

        // inc iter_round
        iter_round += 1;
    }
}

pub fn cg_method<CB: FnMut(i32, f32, f32)>(
    b_mat: &CsrMatrixF16,
    c: Buffer,
    layer: Buffer,
    x: Buffer,
    tol: f32,
    metric_step: i32,
    metric_cb: CB,
) -> Result<(Vec<u8>, i32)> {
    if tol <= 0.0 {
        return Err(ErrorMessage(format!("tol must be positive (tol={tol})")));
    }
    if b_mat.ncols() != b_mat.nrows() {
        return Err(ErrorMessage(format!(
            "B should be square. #B.rows: {} != #B.cols: {}",
            x.len(),
            b_mat.ncols()
        )));
    }
    if x.len() != b_mat.ncols() {
        return Err(ErrorMessage(format!(
            "#x.rows={} should equal to #B.cols={}",
            x.len(),
            b_mat.ncols()
        )));
    }
    if c.len() != b_mat.nrows() {
        return Err(ErrorMessage(format!(
            "#c.rows={} should equal to #B.rows={}",
            c.len(),
            b_mat.nrows()
        )));
    }
    Ok(cg_method_unchecked(
        b_mat,
        c,
        layer,
        x,
        tol,
        metric_step,
        metric_cb,
        gpu,
    ))
}
