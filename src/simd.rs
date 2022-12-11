use nalgebra_sparse::CsrMatrix;
use std::simd::{LaneCount, Mask, Simd, SimdFloat, StdFloat, SupportedLaneCount};

pub const LANE: usize = 16;

pub struct CsrMatrixF32 {
    nrows: usize,
    ncols: usize,
    // length = n_row + 1
    row_offsets: Vec<usize>,
    // length = nnz
    col_indices: Vec<usize>,
    // length = nnz
    values: Vec<f32>,
}

/// Computation of sparse matrix with dense matrix
///
/// c = beta * c + alpha * A * b
#[inline]
pub fn spmmv_dense(beta: f32, c: &mut [f32], alpha: f32, a: &CsrMatrixF32, b: &[f32]) {
    let row_offsets = a.row_offsets.as_slice();
    let col_indices = a.col_indices.as_slice();
    let values = a.values.as_slice();

    if beta == 0.0 && alpha == 1.0 {
        // c = A * b
        sparse_matrix_vector_prod(c, row_offsets, col_indices, values, b)
    } else {
        mpmmv_full(beta, c, alpha, row_offsets, col_indices, values, b)
    }
}

/// y <- a * x + y
#[inline(always)]
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    let (y0, y_sim, y1) = y.as_simd_mut::<LANE>();
    let len0 = y0.len();
    let len1 = len0 + y_sim.len() * LANE;
    let (x0, x_sim, x1) = (
        &x[..len0],
        x[len0..len1]
            .array_chunks()
            .map(|&b| Simd::<f32, LANE>::from_array(b)),
        &x[len1..],
    );
    let a_sim = Simd::<f32, LANE>::splat(a);
    y_sim
        .iter_mut()
        .zip(x_sim)
        .for_each(|(y, x)| *y += a_sim * x);
    y0.iter_mut().zip(x0).for_each(|(y, x)| *y += a * x);
    y1.iter_mut().zip(x1).for_each(|(y, x)| *y += a * x);
}

/// y <- a * x + b * y
#[inline(always)]
pub fn axpby(a: f32, x: &[f32], b: f32, y: &mut [f32]) {
    let (y0, y_sim, y1) = y.as_simd_mut::<LANE>();
    let len0 = y0.len();
    let len1 = len0 + y_sim.len() * LANE;
    let (x0, x_sim, x1) = (
        &x[..len0],
        x[len0..len1]
            .array_chunks()
            .map(|&b| Simd::<f32, LANE>::from_array(b)),
        &x[len1..],
    );
    let a_sim = Simd::<f32, LANE>::splat(a);
    let b_sim = Simd::<f32, LANE>::splat(b);
    y_sim
        .iter_mut()
        .zip(x_sim)
        .for_each(|(y, x)| *y = y.mul_add(b_sim, a_sim * x));
    y0.iter_mut()
        .zip(x0)
        .for_each(|(y, x)| *y = y.mul_add(b, a * x));
    y1.iter_mut()
        .zip(x1)
        .for_each(|(y, x)| *y = y.mul_add(b, a * x));
}

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let (a0, a_sim, a1) = a.as_simd::<LANE>();
    let p0 = a0.len();
    let p1 = p0 + a_sim.len() * LANE;
    let (b0, b_sim, b1) = (
        &b[..p0],
        b[p0..p1]
            .array_chunks()
            .copied()
            .map(Simd::<f32, LANE>::from_array),
        &b[p1..],
    );
    dot += a_sim
        .iter()
        .zip(b_sim)
        .fold(Simd::<f32, LANE>::splat(0.0f32), |acc, (a, b)| {
            a.mul_add(b, acc)
        })
        .reduce_sum();
    dot += dot_slow(a0, b0);
    dot += dot_slow(a1, b1);
    dot
}

#[inline(always)]
fn dot_slow(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter().copied())
        .fold(0.0, |acc, (a0, b0)| a0.mul_add(b0, acc))
}

#[inline(always)]
pub fn metric_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let (a0, a_sim, a1) = a.as_simd::<LANE>();
    let p0 = a0.len();
    let p1 = p0 + a_sim.len() * LANE;
    let (b0, b_sim, b1) = (
        &b[..p0],
        b[p0..p1]
            .array_chunks()
            .copied()
            .map(Simd::<f32, LANE>::from_array),
        &b[p1..],
    );
    dot += a_sim
        .iter()
        .zip(b_sim)
        .fold(Simd::<f32, LANE>::splat(0.0f32), |acc, (a, b)| {
            let d = a - b;
            d.mul_add(d, acc)
        })
        .reduce_sum();
    dot += metric_distance_squared_slow(a0, b0);
    dot += metric_distance_squared_slow(a1, b1);
    dot
}

#[inline(always)]
fn metric_distance_squared_slow(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).fold(0.0, |acc, (a, b)| {
        let d = a - b;
        d.mul_add(d, acc)
    })
}

#[inline(always)]
pub fn norm_squared(v: &[f32]) -> f32 {
    let mut norm = 0.0f32;
    let mut norm_sim = Simd::<f32, LANE>::splat(0.0f32);
    let (v0, v_sim, v1) = v.as_simd::<LANE>();
    for i in v0 {
        norm = i.mul_add(*i, norm);
    }
    for i in v1 {
        norm = i.mul_add(*i, norm);
    }
    for i_sim in v_sim {
        norm_sim = i_sim.mul_add(*i_sim, norm_sim);
    }
    norm += norm_sim.reduce_sum();
    norm
}

#[inline(always)]
pub fn subtract_from(x: &mut [f32], c: &[f32]) {
    x.iter_mut().zip(c).for_each(|(x, c)| *x -= c);
}

#[inline(always)]
pub fn neg(x: &mut [f32]) {
    x.iter_mut().for_each(|x| *x = -*x);
}

#[inline(always)]
fn mpmmv_full(
    beta: f32,
    c: &mut [f32],
    alpha: f32,
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f32],
    b: &[f32],
) {
    let beta_sim = Simd::<f32, LANE>::splat(beta);
    let alpha_sim = Simd::<f32, LANE>::splat(alpha);
    let (c0, c_sim, c1) = c.as_simd_mut::<LANE>();
    let mut ind = 0usize;
    unsafe {
        for i in c0 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            let dot = dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            *i = i.mul_add(beta, dot * alpha);
            ind += 1;
        }
        for i_sim in c_sim {
            let dot_sim = dot_i_lane(ind, row_offsets, col_indices, values, b);
            *i_sim = i_sim.mul_add(beta_sim, dot_sim * alpha_sim);
            ind += LANE;
        }
        for i in c1 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            let dot = dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            *i = i.mul_add(beta, dot * alpha);
            ind += 1;
        }
    }
}

/// c = A * b
#[inline(always)]
fn sparse_matrix_vector_prod(
    c: &mut [f32],
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f32],
    b: &[f32],
) {
    let (c0, c_sim, c1) = c.as_simd_mut::<LANE>();
    let mut ind = 0usize;
    unsafe {
        for i in c0 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            *i = dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            ind += 1;
        }
        for i_sim in c_sim {
            *i_sim = dot_i_lane(ind, row_offsets, col_indices, values, b);
            ind += LANE;
        }
        for i in c1 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            *i = dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            ind += 1;
        }
    }
}

#[inline(always)]
unsafe fn dot_i_lane(
    i: usize,
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f32],
    b: &[f32],
) -> Simd<f32, LANE> {
    let p0 = *row_offsets.get_unchecked(i);
    let p1 = *row_offsets.get_unchecked(i + 1);
    let p2 = *row_offsets.get_unchecked(i + 2);
    let p3 = *row_offsets.get_unchecked(i + 3);
    let p4 = *row_offsets.get_unchecked(i + 4);
    let p5 = *row_offsets.get_unchecked(i + 5);
    let p6 = *row_offsets.get_unchecked(i + 6);
    let p7 = *row_offsets.get_unchecked(i + 7);
    let p8 = *row_offsets.get_unchecked(i + 8);
    let p9 = *row_offsets.get_unchecked(i + 9);
    let p10 = *row_offsets.get_unchecked(i + 10);
    let p11 = *row_offsets.get_unchecked(i + 11);
    let p12 = *row_offsets.get_unchecked(i + 12);
    let p13 = *row_offsets.get_unchecked(i + 13);
    let p14 = *row_offsets.get_unchecked(i + 14);
    let p15 = *row_offsets.get_unchecked(i + 15);
    let p16 = *row_offsets.get_unchecked(i + 16);
    Simd::<f32, LANE>::from_array([
        dot_i(
            col_indices.get_unchecked(p0..p1),
            values.get_unchecked(p0..p1),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p1..p2),
            values.get_unchecked(p1..p2),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p2..p3),
            values.get_unchecked(p2..p3),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p3..p4),
            values.get_unchecked(p3..p4),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p4..p5),
            values.get_unchecked(p4..p5),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p5..p6),
            values.get_unchecked(p5..p6),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p6..p7),
            values.get_unchecked(p6..p7),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p7..p8),
            values.get_unchecked(p7..p8),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p8..p9),
            values.get_unchecked(p8..p9),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p9..p10),
            values.get_unchecked(p9..p10),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p10..p11),
            values.get_unchecked(p10..p11),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p11..p12),
            values.get_unchecked(p11..p12),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p12..p13),
            values.get_unchecked(p12..p13),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p13..p14),
            values.get_unchecked(p13..p14),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p14..p15),
            values.get_unchecked(p14..p15),
            b,
        ),
        dot_i(
            col_indices.get_unchecked(p15..p16),
            values.get_unchecked(p15..p16),
            b,
        ),
    ])
}

#[inline(always)]
unsafe fn dot_i(col_indices: &[usize], values: &[f32], b: &[f32]) -> f32 {
    match col_indices.len() {
        0 => 0.0,
        1 => values.get_unchecked(0) * b.get_unchecked(*col_indices.get_unchecked(0)),
        2 => {
            values.get_unchecked(0) * b.get_unchecked(*col_indices.get_unchecked(0))
                + values.get_unchecked(1) * b.get_unchecked(*col_indices.get_unchecked(1))
        }
        3 => dot_i_loop::<3>(col_indices, values, b),
        4 => dot_i_loop::<4>(col_indices, values, b),
        5 => dot_i_loop::<5>(col_indices, values, b),
        6 => dot_i_loop::<6>(col_indices, values, b),
        7 => dot_i_loop::<7>(col_indices, values, b),
        8..=15 => dot_i_sim::<4>(col_indices, values, b),
        16..=31 => dot_i_sim::<8>(col_indices, values, b),
        _ => dot_i_sim::<16>(col_indices, values, b),
    }
}

#[inline(always)]
unsafe fn dot_i_sim<const LANE: usize>(col_indices: &[usize], values: &[f32], b: &[f32]) -> f32
where
    LaneCount<LANE>: SupportedLaneCount,
{
    let mut dot = 0.0f32;
    let (v0, v_sim, v1) = values.as_simd::<LANE>();
    let p0 = v0.len();
    let p1 = p0 + v_sim.len() * LANE;
    let (c0, c_sim, c1) = (
        &col_indices[..p0],
        col_indices[p0..p1]
            .array_chunks::<LANE>()
            .copied()
            .map(Simd::<usize, LANE>::from_array),
        &col_indices[p1..],
    );
    dot += v_sim
        .iter()
        .zip(c_sim)
        .fold(Simd::<f32, LANE>::splat(0.0f32), |acc, (v, c)| {
            v.mul_add(gather_select::<LANE>(b, c), acc)
        })
        .reduce_sum();
    dot += v0
        .iter()
        .zip(c0)
        .fold(0.0f32, |acc, (v, c)| v.mul_add(*b.get_unchecked(*c), acc));
    dot += v1
        .iter()
        .zip(c1)
        .fold(0.0f32, |acc, (v, c)| v.mul_add(*b.get_unchecked(*c), acc));
    dot
}

#[inline(always)]
unsafe fn dot_i_loop<const N: usize>(col_indices: &[usize], values: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..N {
        let col_idx = *col_indices.get_unchecked(j);
        let b_j = b.get_unchecked(col_idx);
        let a_ij = values.get_unchecked(j);
        sum = b_j.mul_add(*a_ij, sum);
    }
    sum
}

#[inline(always)]
unsafe fn gather_select<const LANE: usize>(
    full_slice: &[f32],
    index: Simd<usize, LANE>,
) -> Simd<f32, LANE>
where
    LaneCount<LANE>: SupportedLaneCount,
{
    let enable = Mask::<isize, LANE>::splat(true);
    let or = Simd::<f32, LANE>::splat(0.0f32);
    unsafe { Simd::<f32, LANE>::gather_select_unchecked(full_slice, enable, index, or) }
}

impl CsrMatrixF32 {
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

impl From<CsrMatrix<f32>> for CsrMatrixF32 {
    fn from(value: CsrMatrix<f32>) -> Self {
        let nrows = value.nrows();
        let ncols = value.ncols();
        let (row_offsets, col_indices, values) = value.disassemble();
        Self {
            nrows,
            ncols,
            row_offsets,
            col_indices,
            values,
        }
    }
}

#[cfg(test)]
mod simd_test {
    use super::*;

    #[test]
    fn test_norm() {
        let a = vec![1., 2., 3., 4., 5.];
        assert_eq!(norm_squared(&a), a.iter().map(|a| a * a).sum());
    }

    #[test]
    fn test_metric() {
        let a = vec![1., 2., 3., 4., 5.];
        let b = vec![5., 4., 3., 2., 1.];
        assert_eq!(
            metric_distance_squared(&a, &b),
            metric_distance_squared_slow(&a, &b)
        );
    }

    #[test]
    fn test_axbpy() {
        let x = vec![1., 2., 3., 4., 5.];
        let mut y = vec![5., 4., 3., 2., 1.];
        axpby(2.0, &x, -1.0, &mut y);
        assert_eq!(y, vec![-3., 0., 3., 6., 9.])
    }

    #[test]
    fn test_axpy() {
        let x = vec![1., 2., 3., 4., 5.];
        let mut y = vec![5., 4., 3., 2., 1.];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![7., 8., 9., 10., 11.])
    }

    #[test]
    fn test_dot() {
        let a = vec![1., 2., 3., 4., 5.];
        let b = vec![5., 4., 3., 2., 1.];
        assert_eq!(dot(&a, &b), dot_slow(&a, &b));
    }

    #[test]
    fn test_neg() {
        let mut a = vec![1., 2., 3., 4., 5.];
        neg(&mut a);
        assert_eq!(a, vec![-1., -2., -3., -4., -5.])
    }

    #[test]
    fn test_sub() {
        let x = vec![1., 2., 3., 4., 5.];
        let mut y = vec![5., 4., 3., 2., 1.];
        subtract_from(&mut y, &x);
        assert_eq!(y, vec![4., 2., 0., -2., -4.])
    }
}
