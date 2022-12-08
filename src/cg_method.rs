use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::simd_utils::{
    axpby, axpy, dot, norm_squared, spbmv_cs_dense, spmv_cs_dense, ONE_F32X4, ZERO_F32X4,
};
use nalgebra_sparse::CsrMatrix;
use std::ops::Neg;
use std::simd::{f32x4, SimdFloat, StdFloat};

/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn cg_method_unchecked<CB: FnMut(i32, &[f32x4], f32)>(
    b_mat: &CsrMatrix<f32>,
    mut c: Vec<f32x4>,
    mut x: Vec<f32x4>,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
) -> (Vec<f32x4>, i32) {
    // r = c
    let r = c.as_mut_slice();
    // r = -r + b * x
    spbmv_cs_dense(ONE_F32X4.neg(), r, ONE_F32X4, b_mat, &x);
    // p = -r
    let mut p = r.iter().map(|r| -*r).collect::<Vec<_>>();
    let mut iter_round = 0;
    let mut bp = vec![ZERO_F32X4; b_mat.nrows()];
    loop {
        // ||r||2
        let r_norm_squared = norm_squared(r);
        // b * p (the only allocation in the loop)
        spmv_cs_dense(&mut bp, ONE_F32X4, b_mat, &p);
        // alpha = ||r||2 / (p' * b * p)
        let alpha = r_norm_squared / dot(&p, &bp);
        // x = x + alpha * p
        axpy(alpha, &p, &mut x);
        // r = r + alpha * b * p
        axpy(alpha, &bp, r);
        let r_new_norm_squared = norm_squared(r);
        let r_new_norm = r_new_norm_squared.sqrt().reduce_max();
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
        axpby(ONE_F32X4.neg(), r, beta, &mut p);

        // inc iter_round
        iter_round += 1;
    }
}

pub fn cg_method<CB: FnMut(i32, &[f32x4], f32)>(
    b_mat: &CsrMatrix<f32>,
    c: Vec<f32x4>,
    x: Vec<f32x4>,
    tol: f32,
    metric_step: i32,
    metric_cb: CB,
) -> Result<(Vec<f32x4>, i32)> {
    if tol <= 0.0 {
        return Err(ErrorMessage(format!("tol must be positive (tol={})", tol)));
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
