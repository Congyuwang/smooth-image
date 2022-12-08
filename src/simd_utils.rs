use nalgebra_sparse::CsrMatrix;
use std::ops::AddAssign;
use std::simd::{f32x4, StdFloat};

pub const ZERO_F32X4: f32x4 = f32x4::from_array([0.0; 4]);
pub const ONE_F32X4: f32x4 = f32x4::from_array([1.0; 4]);

/// c = beta * c + alpha * Op(a) * b
pub fn spbmv_cs_dense(beta: f32x4, c: &mut [f32x4], alpha: f32x4, a: &CsrMatrix<f32>, b: &[f32x4]) {
    c.iter_mut()
        .zip(a.row_iter())
        .for_each(|(c_i, a_row_i)| {
            let dot_ij = a_row_i
                .col_indices()
                .iter()
                .zip(a_row_i.values().iter())
                .map(|(j, v)| unsafe { b.get_unchecked(*j) } * f32x4::splat(*v))
                .sum::<f32x4>();
            *c_i = c_i.mul_add(beta, dot_ij * alpha);
        })
}

/// c = alpha * Op(a) * b
pub fn spmv_cs_dense(c: &mut [f32x4], alpha: f32x4, a: &CsrMatrix<f32>, b: &[f32x4]) {
    c.iter_mut()
        .zip(a.row_iter())
        .for_each(|(c_i, a_row_i)| {
            *c_i = a_row_i
                .col_indices()
                .iter()
                .zip(a_row_i.values().iter())
                .map(|(j, v)| alpha * unsafe { b.get_unchecked(*j) } * f32x4::splat(*v))
                .sum::<f32x4>();
        })
}

// y <- a * x + y
pub fn axpy(a: f32x4, x: &[f32x4], y: &mut [f32x4]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(y, x)| y.add_assign(a * x))
}

// y <- a * x + b * y
pub fn axpby(a: f32x4, x: &[f32x4], b: f32x4, y: &mut [f32x4]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(y, x)| *y = y.mul_add(b, a * x))
}

#[inline(always)]
pub fn dot(a: &[f32x4], b: &[f32x4]) -> f32x4 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

#[inline(always)]
pub fn norm_squared(v: &[f32x4]) -> f32x4 {
    v.iter().map(|v| v * v).sum()
}
