use crate::accelerate::copy_f32;
use crate::error::{Error::ErrorMessage, Result};
use nalgebra::DVector;
use nalgebra_sparse::ops::serial::spmm_csc_dense;
use nalgebra_sparse::ops::Op::NoOp;
use nalgebra_sparse::CscMatrix;

/// f(x) = ||a * x - b ||^2 / 2 + mu / 2 * ||D * x||^2
/// Df(x) = (A^T * A + mu * D^T * D) * x - A^T * b
///
/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
#[inline(always)]
fn ag_method_unchecked<CB: FnMut(i32, &DVector<f32>, f32)>(
    b_mat: &CscMatrix<f32>,
    c: DVector<f32>,
    mu: f32,
    mut x: DVector<f32>,
    tol: f32,
    metric_step: i32,
    mut metric_cb: CB,
) -> (DVector<f32>, i32) {
    // constants
    let l = 1.0 + 8.0 * mu;
    let alpha = 1.0 / l;

    // init
    let mut t = 1.0f32;
    let mut beta = 0.0f32;
    let mut y = DVector::zeros(x.nrows());
    let mut x_tmp = DVector::zeros(x.nrows());
    let mut x_old = x.clone();
    let mut iter_round = 0;
    loop {
        // execute the following x = (1 + beta) * z * x - beta * z * x_old + alpha * c;

        // 1. x_tmp for memorizing x
        copy_f32(&x, &mut x_tmp);
        // 2. x is now y^k+1
        x.axpy(-beta, &x_old, 1.0 + beta);
        copy_f32(&x, &mut y);
        // 3. x is now Df(y^k+1)
        spmm_csc_dense(0.0, &mut x, 1.0, NoOp(b_mat), NoOp(&y));
        x.axpy(-1.0, &c, 1.0);
        let grad_norm = x.norm();
        // metric callback
        if metric_step > 0 && iter_round % metric_step == 0 {
            metric_cb(iter_round, &y, grad_norm);
        }
        if grad_norm <= tol {
            return (y, iter_round);
        }
        // 4. x in now x^k+1
        x.axpy(1.0, &y, -alpha);
        // 5. put x_tmp back
        copy_f32(&x_old, &mut x_tmp);

        // update beta
        let t_new = 0.5 + 0.5 * (1.0 + 4.0 * t * t).sqrt();
        beta = (t - 1.0) / t_new;
        t = t_new;

        // inc iter_round
        iter_round += 1;
    }
}

pub fn ag_method<CB: FnMut(i32, &DVector<f32>, f32)>(
    b_mat: &CscMatrix<f32>,
    c: DVector<f32>,
    mu: f32,
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
