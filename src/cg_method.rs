use crate::accelerate::{saxpby, saxpy, sdot, spmv_csc_dense, sset, CscMatrixF32};
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use nalgebra::DVector;
use nalgebra_sparse::ops::Op::NoOp;

/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked<CB: FnMut(i32, &DVector<f32>, f32)>(
    b_mat: &CscMatrixF32,
    mut c: DVector<f32>,
    mut x: DVector<f32>,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
) -> Result<(DVector<f32>, i32)> {
    // r = c
    let r = &mut c;
    // r = -r
    r.neg_mut();
    // r += b * x
    spmv_csc_dense(r, 1.0, NoOp(b_mat), &x)?;
    // p = -r
    let mut p = -r.clone();
    let mut iter_round = 0;
    let mut bp = DVector::<f32>::zeros(b_mat.nrows());
    loop {
        // ||r||2
        let r_norm_squared = sdot(r, r);
        // compute b * p
        // bp = 0; bp = b * p
        sset(0.0, &mut bp);
        spmv_csc_dense(&mut bp, 1.0, NoOp(b_mat), &p)?;
        // alpha = ||r||2 / (p' * b * p)
        let alpha = r_norm_squared / sdot(&p, &bp);
        // x = x + alpha * p
        saxpy(alpha, &p, &mut x);
        // r = r + alpha * b * p
        saxpy(alpha, &bp, r);
        let r_new_norm_squared = sdot(r, r);
        let r_new_norm = r_new_norm_squared.sqrt();
        // metric callback
        if metric_step > 0 && iter_round % metric_step == 0 {
            metric_cb(iter_round, &x, r_new_norm);
        }
        // return condition
        if r_new_norm <= tol {
            return Ok((x, iter_round));
        }
        let beta = r_new_norm_squared / r_norm_squared;
        // p = beta * p - r
        saxpby(-1.0, r, beta, &mut p);

        // inc iter_round
        iter_round += 1;
    }
}

pub fn cg_method<CB: FnMut(i32, &DVector<f32>, f32)>(
    b_mat: &CscMatrixF32,
    c: DVector<f32>,
    x: DVector<f32>,
    tol: f32,
    metric_step: i32,
    metric_cb: CB,
) -> Result<(DVector<f32>, i32)> {
    if tol <= 0.0 {
        return Err(ErrorMessage(format!("tol must be positive (tol={})", tol)));
    }
    if b_mat.ncols() != b_mat.nrows() {
        return Err(ErrorMessage(format!(
            "B should be square. #B.rows: {} != #B.cols: {}",
            x.nrows(),
            b_mat.ncols()
        )));
    }
    if x.nrows() != b_mat.ncols() {
        return Err(ErrorMessage(format!(
            "#x.rows={} should equal to #B.cols={}",
            x.nrows(),
            b_mat.ncols()
        )));
    }
    if c.nrows() != b_mat.nrows() {
        return Err(ErrorMessage(format!(
            "#c.rows={} should equal to #B.rows={}",
            c.nrows(),
            b_mat.nrows()
        )));
    }
    let output = cg_method_unchecked(b_mat, c, x, tol, metric_step, metric_cb)?;
    Ok(output)
}
