use crate::error::{Error::ErrorMessage, Result};
use nalgebra::DVector;
use nalgebra_sparse::ops::serial::spmm_csr_dense;
use nalgebra_sparse::ops::Op::NoOp;
use nalgebra_sparse::CsrMatrix;

/// f(x) = ||a * x - b ||^2 / 2 + mu / 2 * ||D * x||^2
/// Df(x) = (A^T * A + mu * D^T * D) * x - A^T * b
///
/// B_mat = A^T * A + mu * D^T * D
/// c = A^T * b
///
/// Z = I - alpha * B_mat
///
/// x = (1 + beta) * Z * x -  beta * Z * x_old + alpha * c
#[inline(always)]
fn ag_method_unchecked(
    b_mat: &CsrMatrix<f32>,
    c: DVector<f32>,
    mu: f32,
    mut x: DVector<f32>,
    tol: f32,
) -> (DVector<f32>, i32) {
    // constants
    let l = 1.0 + 8.0 * mu;
    let alpha = 1.0 / l;
    let z = CsrMatrix::identity(b_mat.nrows()) - alpha * b_mat;

    // init
    let mut t = 1.0f32;
    let mut beta = 0.0f32;
    let mut x_tmp = x.clone();
    let mut x_old = x.clone();
    let mut iter_count = 0;
    loop {
        iter_count += 1;
        // execute the following x = (1 + beta) * z * x - beta * z * x_old + alpha * c;

        // 1. backup x value
        x_tmp.copy_from(&x);
        // 2. x = (1 + beta) * z * x
        spmm_csr_dense(0.0, &mut x, 1.0 + beta, NoOp(&z), NoOp(&x_tmp));
        // 3. ||Df(x)|| compare with tol
        let mut df = x.clone();
        df.axpy(1.0 / alpha, &x_tmp, -1.0 / alpha / (1.0 + beta));
        if df.metric_distance(&c) <= tol {
            return (x, iter_count);
        }
        // 4. x -= beta * z * x_old
        spmm_csr_dense(1.0, &mut x, -beta, NoOp(&z), NoOp(&x_old));
        // 5. x += alpha * c
        x.axpy(alpha, &c, 1.0);
        // 6. update x_old
        x_old.copy_from(&x_tmp);

        // update beta
        let t_new = 0.5 + 0.5 * (1.0 + 4.0 * t * t).sqrt();
        beta = (t - 1.0) / t_new;
        t = t_new;
    }
}

pub fn ag_method(
    b_mat: &CsrMatrix<f32>,
    c: DVector<f32>,
    mu: f32,
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
    Ok(ag_method_unchecked(b_mat, c, mu, x, tol))
}
