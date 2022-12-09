use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::simd::{
    axpby, axpy, dot, matrix_vector_prod, neg, norm_squared, spbmv_cs_dense, CsrMatrixF32,
};




/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked<CB: FnMut(i32, &[f32], f32)>(
    b_mat: &CsrMatrixF32,
    mut c: Vec<f32>,
    mut x: Vec<f32>,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
) -> (Vec<f32>, i32) {
    // r = c
    let r = c.as_mut_slice();
    // r = -r + b * x
    spbmv_cs_dense(-1.0, r, 1.0, b_mat, &x);
    // p = -r
    let mut p = Vec::from(&*r);
    neg(&mut p);
    let mut iter_round = 0;
    let mut bp = vec![0.0f32; b_mat.nrows()];
    loop {
        // ||r||2
        let r_norm_squared = norm_squared(r);
        // b * p (the only allocation in the loop)
        matrix_vector_prod(&mut bp, b_mat, &p);
        // alpha = ||r||2 / (p' * b * p)
        let alpha = r_norm_squared / dot(&p, &bp);
        // x = x + alpha * p
        axpy(alpha, &p, &mut x);
        // r = r + alpha * b * p
        axpy(alpha, &bp, r);
        let r_new_norm_squared = norm_squared(r);
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
        axpby(-1.0, r, beta, &mut p);

        // inc iter_round
        iter_round += 1;
    }
}

pub fn cg_method<CB: FnMut(i32, &[f32], f32)>(
    b_mat: &CsrMatrixF32,
    c: Vec<f32>,
    x: Vec<f32>,
    tol: f32,
    metric_step: i32,
    metric_cb: CB,
) -> Result<(Vec<f32>, i32)> {
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
        x,
        tol,
        metric_step,
        metric_cb,
    ))
}
