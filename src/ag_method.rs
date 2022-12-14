use std::ptr::slice_from_raw_parts;
use half::f16;
use metal::{Buffer, MTLResourceOptions};
use crate::error::{Error::ErrorMessage, Result};
use crate::gpu_metal::GpuLib;
use crate::inpaint_worker::CsrMatrixF16;

/// f(x) = ||a * x - b ||^2 / 2 + mu / 2 * ||D * x||^2
/// Df(x) = (A^T * A + mu * D^T * D) * x - A^T * b
///
/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn ag_method_unchecked<CB: FnMut(i32, f32, f32)>(
    b_mat: &CsrMatrixF16,
    c: Buffer,
    mu: f32,
    layer: Buffer,
    x: Buffer,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
    gpu: &GpuLib,
) -> (Vec<u8>, i32) {
    let size = b_mat.nrows();
    let (arg, data) = gpu.init_ag_argument(b_mat, c, layer, mu, x);
    let mut iter_round = 0;
    let queue = gpu.new_queue();
    let mut command = gpu.iter_ag(&arg, &data, &queue, size as u64);
    command.commit();
    loop {
        let new_command = gpu.iter_ag(&arg, &data, &queue, size as u64);
        command.wait_until_completed();

        let grad_norm = unsafe { *(data[4].contents() as *mut f32) }.sqrt();
        let diff_squared = unsafe { *(data[6].contents() as *mut f32) };

        // check return condition
        if grad_norm <= tol {
            let queue = gpu.new_queue();
            let cmd = queue.new_command_buffer();
            let y_data = data[3];
            let new_y_buffer = gpu
                .device()
                .new_buffer(y_data.length(), MTLResourceOptions::StorageModeShared);
            gpu.copy_from_buffer(y_data, &new_y_buffer, cmd);
            cmd.commit();
            cmd.wait_until_completed();
            let result = slice_from_raw_parts(y_data.contents() as *const f16, size).iter().map(|f| {
                (f.to_f32() * 256.0f32) as u8
            }).collect::<Vec<_>>();
            return (result, iter_round);
        }

        new_command.commit();
        command = new_command;

        // metric callback and return condition
        if metric_step > 0 && iter_round % metric_step == 0 {
            metric_cb(iter_round, diff_squared, grad_norm);
        }

        // inc iter_round
        iter_round += 1;
    }
}

pub fn ag_method<CB: FnMut(i32, f32, f32)>(
    b_mat: &CsrMatrixF16,
    c: Buffer,
    mu: f32,
    layer: Buffer,
    x: Buffer,
    tol: f32,
    metric_step: i32,
    metric_cb: CB,
    gpu: &GpuLib,
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
    Ok(ag_method_unchecked(
        b_mat,
        c,
        mu,
        layer,
        x,
        tol,
        metric_step,
        metric_cb,
        gpu,
    ))
}
