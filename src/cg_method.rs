use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::gpu_metal::GpuLib;
use crate::inpaint_worker::CsrMatrixF16;
use half::f16;
use metal::{Buffer, MTLResourceOptions};
use std::mem::size_of;
use std::slice;
use std::sync::{Arc, Mutex};

/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked<CB: FnMut(i32, f32, f32)>(
    b_mat: &Arc<Mutex<CsrMatrixF16>>,
    (c, layer, x): (Buffer, Buffer, Buffer),
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
    gpu: &Arc<Mutex<GpuLib>>,
) -> (Vec<u8>, i32) {
    let size = b_mat.lock().unwrap().nrows();
    let (arg, data) = gpu.lock().unwrap().init_cg_argument(b_mat, c, layer, x);
    let mut iter_round = 0;
    let queue = gpu.lock().unwrap().new_queue();
    let mut command = gpu
        .lock()
        .unwrap()
        .iter_cg(&arg, &data, &queue, size as u64)
        .to_owned();
    command.commit();
    loop {
        // running GPU and CPU in parallel
        let new_command = gpu
            .lock()
            .unwrap()
            .iter_cg(&arg, &data, &queue, size as u64)
            .to_owned();
        command.wait_until_completed();

        let r_new_norm = unsafe { *(data[0].contents() as *mut f32) }.sqrt();
        let diff_squared = unsafe { *(data[3].contents() as *mut f32) };

        // check return condition
        if r_new_norm <= tol {
            let queue = gpu.lock().unwrap().new_queue();
            let cmd = queue.new_command_buffer();
            let x_data = &data[6];
            let new_x_buffer = gpu
                .lock()
                .unwrap()
                .device()
                .new_buffer(x_data.length(), MTLResourceOptions::StorageModeShared);
            gpu.lock()
                .unwrap()
                .copy_from_buffer(x_data, &new_x_buffer, cmd);
            cmd.commit();
            cmd.wait_until_completed();
            let result = unsafe { slice::from_raw_parts(x_data.contents() as *const f16, size) }
                .iter()
                .map(|f| (f.to_f32() * 256.0f32) as u8)
                .collect::<Vec<_>>();
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
    b_mat: &Arc<Mutex<CsrMatrixF16>>,
    (c, layer, x): (Buffer, Buffer, Buffer),
    tol: f32,
    metric_step: i32,
    metric_cb: CB,
    gpu: &Arc<Mutex<GpuLib>>,
) -> Result<(Vec<u8>, i32)> {
    if tol <= 0.0 {
        return Err(ErrorMessage(format!("tol must be positive (tol={tol})")));
    }
    {
        let b_mat = b_mat.lock().unwrap();
        if b_mat.ncols() != b_mat.nrows() {
            return Err(ErrorMessage(format!(
                "B should be square. #B.rows: {} != #B.cols: {}",
                b_mat.nrows(),
                b_mat.ncols()
            )));
        }
        if x.length() as usize / size_of::<f16>() != b_mat.ncols() {
            return Err(ErrorMessage(format!(
                "#x.rows={} should equal to #B.cols={}",
                x.length() as usize / size_of::<f16>(),
                b_mat.ncols()
            )));
        }
        if c.length() as usize / size_of::<f16>() != b_mat.nrows() {
            return Err(ErrorMessage(format!(
                "#c.rows={} should equal to #B.rows={}",
                c.length() as usize / size_of::<f16>(),
                b_mat.nrows()
            )));
        }
    }
    Ok(cg_method_unchecked(
        b_mat,
        (c, layer, x),
        tol,
        metric_step,
        metric_cb,
        gpu,
    ))
}
