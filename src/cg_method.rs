use crate::error::Error::ErrorMessage;
use crate::error::Result;
use nalgebra::DVector;
use nalgebra_sparse::ops::{serial::spmm_csr_dense, Op::NoOp};
use nalgebra_sparse::CsrMatrix;

/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked(
    b_mat: &CsrMatrix<f32>,
    mut c: DVector<f32>,
    mut x: DVector<f32>,
    tol: f32,
) -> (DVector<f32>, i32) {
    // r = c
    let r = &mut c;
    // r = -r + b * x
    spmm_csr_dense(-1.0, &mut *r, 1.0, NoOp(b_mat), NoOp(&x));
    // p = -r
    let mut p = -r.clone();
    let mut iter_count = 0;
    loop {
        iter_count += 1;
        // ||r||2
        let r_norm_squared = r.norm_squared();
        // b * p (the only allocation in the loop)
        let bp = b_mat * &p;
        // alpha = ||r||2 / (p' * b * p)
        let alpha = r_norm_squared / p.dot(&bp);
        // x = x + alpha * p
        x.axpy(alpha, &p, 1.0);
        // r = r + alpha * b * p
        r.axpy(alpha, &bp, 1.0);
        let r_new_norm_squared = r.norm_squared();
        // return condition
        if r_new_norm_squared <= tol {
            return (x, iter_count);
        }
        let beta = r_new_norm_squared / r_norm_squared;
        // p = beta * p - r
        p.axpy(-1.0, r, beta);
    }
}

pub fn cg_method(
    b_mat: &CsrMatrix<f32>,
    c: DVector<f32>,
    x: DVector<f32>,
    tol: f32,
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
    Ok(cg_method_unchecked(b_mat, c, x, tol))
}
