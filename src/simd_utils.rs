use nalgebra_sparse::CsrMatrix;
use rayon::join;
use std::ops::{AddAssign, SubAssign};
use std::simd::{f32x4, StdFloat};

pub const ZERO_F32X4: f32x4 = f32x4::from_array([0.0; 4]);
pub const ONE_F32X4: f32x4 = f32x4::from_array([1.0; 4]);

/// c = beta * c + alpha * Op(a) * b
#[inline(always)]
pub fn spbmv_cs_dense(beta: f32x4, c: &mut [f32x4], alpha: f32x4, a: &CsrMatrix<f32>, b: &[f32x4]) {
    let (mid, mid_lo, mid_hi, p0, p1, p2, p3) = split_in_four_mut(c);
    join(
        || {
            join(
                || spbmv_cs_dense_part(beta, p0, alpha, a, b, 0),
                || spbmv_cs_dense_part(beta, p1, alpha, a, b, mid_lo),
            )
        },
        || {
            join(
                || spbmv_cs_dense_part(beta, p2, alpha, a, b, mid),
                || spbmv_cs_dense_part(beta, p3, alpha, a, b, mid_hi),
            )
        },
    );
}

#[inline]
pub fn spbmv_cs_dense_part(
    beta: f32x4,
    c: &mut [f32x4],
    alpha: f32x4,
    a: &CsrMatrix<f32>,
    b: &[f32x4],
    start_row: usize,
) {
    c.iter_mut()
        .enumerate()
        .map(|(i, c_i)| (c_i, a.get_row(i + start_row).unwrap()))
        .for_each(|(c_i, a_row_i)| {
            let dot_ij = a_row_i
                .col_indices()
                .iter()
                .zip(a_row_i.values().iter())
                .map(|(j, v)| unsafe { b.get_unchecked(*j) } * f32x4::splat(*v))
                .sum::<f32x4>();
            *c_i = c_i.mul_add(beta, dot_ij * alpha);
        });
}

/// c = alpha * Op(a) * b
#[inline(always)]
pub fn spmv_cs_dense(c: &mut [f32x4], alpha: f32x4, a: &CsrMatrix<f32>, b: &[f32x4]) {
    let (mid, mid_lo, mid_hi, p0, p1, p2, p3) = split_in_four_mut(c);
    join(
        || {
            join(
                || spmv_cs_dense_part(p0, alpha, a, b, 0),
                || spmv_cs_dense_part(p1, alpha, a, b, mid_lo),
            )
        },
        || {
            join(
                || spmv_cs_dense_part(p2, alpha, a, b, mid),
                || spmv_cs_dense_part(p3, alpha, a, b, mid_hi),
            )
        },
    );
}

#[inline(always)]
pub fn spmv_cs_dense_part(
    c: &mut [f32x4],
    alpha: f32x4,
    a: &CsrMatrix<f32>,
    b: &[f32x4],
    start_row: usize,
) {
    c.iter_mut()
        .enumerate()
        .map(|(i, c_i)| (c_i, a.get_row(i + start_row).unwrap()))
        .for_each(|(c_i, a_row_i)| {
            *c_i = a_row_i
                .col_indices()
                .iter()
                .zip(a_row_i.values().iter())
                .map(|(j, v)| alpha * unsafe { b.get_unchecked(*j) } * f32x4::splat(*v))
                .sum::<f32x4>();
        })
}

/// y <- a * x + y
#[inline(always)]
pub fn axpy(a: f32x4, x: &[f32x4], y: &mut [f32x4]) {
    let (_, _, _, x0, x1, x2, x3) = split_in_four(x);
    let (_, _, _, y0, y1, y2, y3) = split_in_four_mut(y);
    join(
        || {
            join(
                || axpy_part(a, x0, y0),
                || axpy_part(a, x1, y1),
            )
        },
        || {
            join(
                || axpy_part(a, x2, y2),
                || axpy_part(a, x3, y3),
            )
        },
    );
}

#[inline(always)]
pub fn axpy_part(a: f32x4, x: &[f32x4], y: &mut [f32x4]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(y, x)| y.add_assign(a * x))
}

/// y <- a * x + b * y
#[inline(always)]
pub fn axpby(a: f32x4, x: &[f32x4], b: f32x4, y: &mut [f32x4]) {
    let (_, _, _, x0, x1, x2, x3) = split_in_four(x);
    let (_, _, _, y0, y1, y2, y3) = split_in_four_mut(y);
    join(
        || {
            join(
                || axpby_part(a, x0, b, y0),
                || axpby_part(a, x1, b, y1),
            )
        },
        || {
            join(
                || axpby_part(a, x2, b, y2),
                || axpby_part(a, x3, b, y3),
            )
        },
    );
}

#[inline(always)]
pub fn axpby_part(a: f32x4, x: &[f32x4], b: f32x4, y: &mut [f32x4]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(y, x)| *y = y.mul_add(b, a * x))
}

#[inline(always)]
pub fn dot(a: &[f32x4], b: &[f32x4]) -> f32x4 {
    let (_, _, _, x0, x1, x2, x3) = split_in_four(a);
    let (_, _, _, y0, y1, y2, y3) = split_in_four(b);
    let (a, b) = join(
        || {
            let (a, b) = join(
                || dot_part(x0, y0),
                || dot_part(x1, y1),
            );
            a + b
        },
        || {
            let (a, b) = join(
                || dot_part(x2, y2),
                || dot_part(x3, y3),
            );
            a + b
        },
    );
    a + b
}

#[inline(always)]
pub fn dot_part(a: &[f32x4], b: &[f32x4]) -> f32x4 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

#[inline(always)]
pub fn norm_squared(v: &[f32x4]) -> f32x4 {
    let (_, _, _, y0, y1, y2, y3) = split_in_four(v);
    let (a, b) = join(
        || {
            let (a, b) = join(
                || norm_squared_part(y0),
                || norm_squared_part(y1),
            );
            a + b
        },
        || {
            let (a, b) = join(
                || norm_squared_part(y2),
                || norm_squared_part(y3),
            );
            a + b
        },
    );
    a + b
}

#[inline(always)]
pub fn norm_squared_part(v: &[f32x4]) -> f32x4 {
    v.iter().map(|v| v * v).sum()
}

#[inline(always)]
pub fn copy(to: &mut[f32x4], from: &[f32x4]) {
    let (_, _, _, x0, x1, x2, x3) = split_in_four_mut(to);
    let (_, _, _, y0, y1, y2, y3) = split_in_four(from);
    join(
        || {
            join(
                || x0.copy_from_slice(y0),
                || x1.copy_from_slice(y1),
            );
        },
        || {
            join(
                || x2.copy_from_slice(y2),
                || x3.copy_from_slice(y3),
            );
        },
    );
}

#[inline(always)]
pub fn subtract_from(x: &mut[f32x4], c: &[f32x4]) {
    let (_, _, _, x0, x1, x2, x3) = split_in_four_mut(x);
    let (_, _, _, y0, y1, y2, y3) = split_in_four(c);
    join(
        || {
            join(
                || subtract_from_part(x0, y0),
                || subtract_from_part(x1, y1),
            );
        },
        || {
            join(
                || subtract_from_part(x2, y2),
                || subtract_from_part(x3, y3),
            );
        },
    );
}

#[inline(always)]
pub fn subtract_from_part(x: &mut[f32x4], c: &[f32x4]) {
    x.iter_mut()
        .zip(c.iter())
        .for_each(|(x, c)| x.sub_assign(c));
}

#[inline(always)]
fn split_in_four(c: &[f32x4]) -> (usize, usize, usize, &[f32x4], &[f32x4], &[f32x4], &[f32x4]) {
    let mid = c.len() / 2;
    let mid_lo = mid / 2;
    let mid_hi = (c.len() + mid) / 2;
    let (re, p3) = c.split_at(mid_hi);
    let (re, p2) = re.split_at(mid);
    let (p0, p1) = re.split_at(mid_lo);
    (mid, mid_lo, mid_hi, p0, p1, p2, p3)
}

#[inline(always)]
fn split_in_four_mut(
    c: &mut [f32x4],
) -> (
    usize,
    usize,
    usize,
    &mut [f32x4],
    &mut [f32x4],
    &mut [f32x4],
    &mut [f32x4],
) {
    let mid = c.len() / 2;
    let mid_lo = mid / 2;
    let mid_hi = (c.len() + mid) / 2;
    let (re, p3) = c.split_at_mut(mid_hi);
    let (re, p2) = re.split_at_mut(mid);
    let (p0, p1) = re.split_at_mut(mid_lo);
    (mid, mid_lo, mid_hi, p0, p1, p2, p3)
}
