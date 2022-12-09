use crate::error::{Error::ErrorMessage, Result};
use crate::simd::{axpby, copy_simd, norm_squared, spbmv_cs_dense, subtract_from, CsrMatrixF32};

/// f(x) = ||a * x - b ||^2 / 2 + mu / 2 * ||D * x||^2
/// Df(x) = (A^T * A + mu * D^T * D) * x - A^T * b
///
/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn ag_method_unchecked<CB: FnMut(i32, &[f32], f32)>(
    b_mat: &CsrMatrixF32,
    c: Vec<f32>,
    mu: f32,
    mut x: Vec<f32>,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
) -> (Vec<f32>, i32) {
    // constants
    let l = 1.0 + 8.0 * mu;
    let alpha = 1.0 / l;

    // init
    let mut t = 1.0f32;
    let mut beta = 0.0f32;
    let mut y = vec![0.0f32; x.len()];
    let mut x_tmp = vec![0.0f32; x.len()];
    let mut x_old = x.clone();
    let mut iter_round = 0;
    loop {
        // execute the following x = (1 + beta) * z * x - beta * z * x_old + alpha * c;

        // 1. x_tmp for memorizing x
        copy_simd(&mut x_tmp, &x);
        // 2. x is now y^k+1
        axpby(-beta, &x_old, 1.0 + beta, &mut x);
        copy_simd(&mut y, &x);
        // 3. x is now Df(y^k+1)
        spbmv_cs_dense(0.0, &mut x, 1.0, b_mat, &y);
        subtract_from(&mut x, &c);
        let grad_norm = norm_squared(&x).sqrt();
        // metric callback
        if metric_step > 0 && iter_round % metric_step == 0 {
            metric_cb(iter_round, &y, grad_norm);
        }
        if grad_norm <= tol {
            return (y, iter_round);
        }
        // 4. x in now x^k+1
        axpby(1.0, &y, -alpha, &mut x);
        // 5. put x_tmp back
        copy_simd(&mut x_old, &x_tmp);

        // update beta
        let t_new = 0.5 + 0.5 * (1.0 + 4.0 * t * t).sqrt();
        beta = (t - 1.0) / t_new;
        t = t_new;

        // inc iter_round
        iter_round += 1;
    }
}

pub fn ag_method<CB: FnMut(i32, &[f32], f32)>(
    b_mat: &CsrMatrixF32,
    c: Vec<f32>,
    mu: f32,
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
    Ok(ag_method_unchecked(
        b_mat,
        c,
        mu,
        x,
        tol,
        metric_step,
        metric_cb,
    ))
}
