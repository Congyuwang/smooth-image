use crate::error::Error::ErrorMessage;
use crate::error::Result;
use nalgebra::DVector;
use nalgebra_sparse::ops::{serial::spmm_csc_dense, Op::NoOp};
use nalgebra_sparse::CscMatrix;

/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked<CB: FnMut(i32, &DVector<f32>, f32)>(
    b_mat: &CscMatrix<f32>,
    mut c: DVector<f32>,
    mut x: DVector<f32>,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
) -> (DVector<f32>, i32) {
    // r = c
    let r = &mut c;
    // r = -r + b * x
    spmm_csc_dense(-1.0, &mut *r, 1.0, NoOp(b_mat), NoOp(&x));
    // p = -r
    let mut p = -r.clone();
    let mut iter_round = 0;
    let mut bp = DVector::<f32>::zeros(b_mat.nrows());
    loop {
        // ||r||2
        let r_norm_squared = r.norm_squared();
        // b * p (the only allocation in the loop)
        spmm_csc_dense(0.0, &mut bp, 1.0, NoOp(b_mat), NoOp(&p));
        // alpha = ||r||2 / (p' * b * p)
        let alpha = r_norm_squared / p.dot(&bp);
        // x = x + alpha * p
        x.axpy(alpha, &p, 1.0);
        // r = r + alpha * b * p
        r.axpy(alpha, &bp, 1.0);
        let r_new_norm_squared = r.norm_squared();
        let r_new_norm = r_new_norm_squared.sqrt();
        // metric callback
        if metric_step > 0 && iter_round % metric_step == 0 {
            metric_cb(iter_round, &x, r_new_norm);
        }
        // return condition
        if r_new_norm <= tol {
            return (x, iter_round);
        }
        let beta = r_new_norm_squared / r_norm_squared;
        // p = beta * p - r
        p.axpy(-1.0, r, beta);

        // inc iter_round
        iter_round += 1;
    }
}

pub fn cg_method<CB: FnMut(i32, &DVector<f32>, f32)>(
    b_mat: &CscMatrix<f32>,
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
    Ok(cg_method_unchecked(b_mat, c, x, tol, metric_step, metric_cb))
}
