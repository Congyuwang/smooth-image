use nalgebra_sparse::CsrMatrix;

use std::simd::{f32x16, f32x4, Simd, SimdFloat, StdFloat};

pub const LANE: usize = 16;
pub const LANE_S: usize = 4;

/// if the inner loop length for sparse is greater than
/// this value, use SIMD; otherwise, use simple loop
pub const SPARSE_INNER_SIMD_THRESH_HOLD: usize = 8;

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

/// c = beta * c + alpha * Op(a) * b
#[inline]
pub fn spbmv_cs_dense(beta: f32, c: &mut [f32], alpha: f32, a: &CsrMatrixF32, b: &[f32]) {
    let (c0, c_sim, c1) = c.as_simd_mut::<LANE>();
    let mut ind = 0usize;
    let beta_sim = Simd::<f32, LANE>::splat(beta);
    let alpha_sim = Simd::<f32, LANE>::splat(alpha);
    let row_offsets = a.row_offsets.as_slice();
    let col_indices = a.col_indices.as_slice();
    let values = a.values.as_slice();
    unsafe {
        for i in c0 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            let dot = dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            *i = i.mul_add(beta, dot * alpha);
            ind += 1;
        }
        for i_sim in c_sim {
            // note that 16 is hard coded here
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

/// c = c + Op(a) * b
pub fn matrix_vector_prod(c: &mut [f32], a: &CsrMatrixF32, b: &[f32]) {
    let (c0, c_sim, c1) = c.as_simd_mut::<LANE>();
    let mut ind = 0usize;
    let row_offsets = a.row_offsets.as_slice();
    let col_indices = a.col_indices.as_slice();
    let values = a.values.as_slice();
    unsafe {
        for i in c0 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            *i += dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            ind += 1;
        }
        for i_sim in c_sim {
            // note that 16 is hard coded here
            *i_sim = dot_i_lane(ind, row_offsets, col_indices, values, b);
            ind += LANE;
        }
        for i in c1 {
            let p0 = *row_offsets.get_unchecked(ind);
            let p1 = *row_offsets.get_unchecked(ind + 1);
            *i += dot_i(&col_indices[p0..p1], &values[p0..p1], b);
            ind += 1;
        }
    }
}

/// y <- a * x + y
#[inline(always)]
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    let mut ind = 0usize;
    let (y0, y_sim, y1) = y.as_simd_mut::<LANE>();
    let a_sim = Simd::<f32, LANE>::splat(a);
    unsafe {
        for i in y0 {
            *i += a * x.get_unchecked(ind);
            ind += 1;
        }
        for i_sim in y_sim {
            // note that 16 is hard coded here
            *i_sim += a_sim * sim_from_slice_16(&x[ind..]);
            ind += LANE;
        }
        for i in y1 {
            *i += a * x.get_unchecked(ind);
            ind += 1;
        }
    }
}

/// y <- a * x + b * y
#[inline(always)]
pub fn axpby(a: f32, x: &[f32], b: f32, y: &mut [f32]) {
    let mut ind = 0usize;
    let (y0, y_sim, y1) = y.as_simd_mut::<LANE>();
    let a_sim = Simd::<f32, LANE>::splat(a);
    let b_sim = Simd::<f32, LANE>::splat(b);
    unsafe {
        for i in y0 {
            *i *= b;
            *i += a * x.get_unchecked(ind);
            ind += 1;
        }
        for i_sim in y_sim {
            // note that 16 is hard coded here
            *i_sim = i_sim.mul_add(b_sim, a_sim * sim_from_slice_16(&x[ind..]));
            ind += LANE;
        }
        for i in y1 {
            *i *= b;
            *i += a * x.get_unchecked(ind);
            ind += 1;
        }
    }
}

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut dot_sim = Simd::<f32, LANE>::splat(0.0f32);
    let mut ind = 0usize;
    let (a0, a_sim, a1) = a.as_simd::<LANE>();
    unsafe {
        for i in a0 {
            dot += i * b.get_unchecked(ind);
            ind += 1;
        }
        for i_sim in a_sim {
            // note that 16 is hard coded here
            dot_sim += i_sim * sim_from_slice_16(&b[ind..]);
            ind += LANE;
        }
        for i in a1 {
            dot += i * b.get_unchecked(ind);
            ind += 1;
        }
    }
    dot += dot_sim.reduce_sum();
    dot
}

#[inline(always)]
pub fn metric_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sq_sum = 0.0f32;
    let mut sq_sum_sim = Simd::<f32, LANE>::splat(0.0f32);
    let mut ind = 0usize;
    let (a0, a_sim, a1) = a.as_simd::<LANE>();
    unsafe {
        for i in a0 {
            let diff = i - b.get_unchecked(ind);
            sq_sum += diff * diff;
            ind += 1;
        }
        for i_sim in a_sim {
            // note that 16 is hard coded here
            let diff_sim = i_sim - sim_from_slice_16(&b[ind..]);
            sq_sum_sim += diff_sim * diff_sim;
            ind += LANE;
        }
        for i in a1 {
            let diff = i - b.get_unchecked(ind);
            sq_sum += diff * diff;
            ind += 1;
        }
    }
    sq_sum += sq_sum_sim.reduce_sum();
    sq_sum
}

#[inline(always)]
pub fn norm_squared(v: &[f32]) -> f32 {
    let mut norm = 0.0f32;
    let mut norm_sim = Simd::<f32, LANE>::splat(0.0f32);
    let (v0, v_sim, v1) = v.as_simd::<LANE>();
    for i in v0 {
        norm += i * i;
    }
    for i in v1 {
        norm += i * i;
    }
    for i_sim in v_sim {
        norm_sim += i_sim * i_sim;
    }
    norm += norm_sim.reduce_sum();
    norm
}

#[inline(always)]
pub fn subtract_from(x: &mut [f32], c: &[f32]) {
    let mut ind = 0usize;
    let (x0, x_sim, x1) = x.as_simd_mut::<LANE>();
    unsafe {
        for i in x0 {
            *i -= c.get_unchecked(ind);
            ind += 1;
        }
        for i_sim in x_sim {
            // note that 16 is hard coded here
            *i_sim -= sim_from_slice_16(&c[ind..]);
            ind += LANE;
        }
        for i in x1 {
            *i += c.get_unchecked(ind);
            ind += 1;
        }
    }
}

pub fn neg(x: &mut [f32]) {
    let (x0, x_sim, x1) = x.as_simd_mut::<LANE>();
    for i in x0 {
        *i = -*i;
    }
    for i_sim in x_sim {
        // note that 16 is hard coded here
        *i_sim = -*i_sim;
    }
    for i in x1 {
        *i = -*i;
    }
}

pub fn clone(x: &mut [f32], c: &[f32]) {
    let mut ind = 0usize;
    let (x0, x_sim, x1) = x.as_simd_mut::<LANE>();
    unsafe {
        for i in x0 {
            *i = *c.get_unchecked(ind);
            ind += 1;
        }
        for i_sim in x_sim {
            // note that 16 is hard coded here
            *i_sim = sim_from_slice_16(&c[ind..]);
            ind += LANE;
        }
        for i in x1 {
            *i = *c.get_unchecked(ind);
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
        dot_i(&col_indices[p0..p1], &values[p0..p1], b),
        dot_i(&col_indices[p1..p2], &values[p1..p2], b),
        dot_i(&col_indices[p2..p3], &values[p2..p3], b),
        dot_i(&col_indices[p3..p4], &values[p3..p4], b),
        dot_i(&col_indices[p4..p5], &values[p4..p5], b),
        dot_i(&col_indices[p5..p6], &values[p5..p6], b),
        dot_i(&col_indices[p6..p7], &values[p6..p7], b),
        dot_i(&col_indices[p7..p8], &values[p7..p8], b),
        dot_i(&col_indices[p8..p9], &values[p8..p9], b),
        dot_i(&col_indices[p9..p10], &values[p9..p10], b),
        dot_i(&col_indices[p10..p11], &values[p10..p11], b),
        dot_i(&col_indices[p11..p12], &values[p11..p12], b),
        dot_i(&col_indices[p12..p13], &values[p12..p13], b),
        dot_i(&col_indices[p13..p14], &values[p13..p14], b),
        dot_i(&col_indices[p14..p15], &values[p14..p15], b),
        dot_i(&col_indices[p15..p16], &values[p15..p16], b),
    ])
}

#[inline(always)]
unsafe fn dot_i(col_indices: &[usize], values: &[f32], b: &[f32]) -> f32 {
    if col_indices.len() > SPARSE_INNER_SIMD_THRESH_HOLD {
        dot_i_sim(col_indices, values, b)
    } else {
        dot_i_simple(col_indices, values, b)
    }
}

#[inline(always)]
unsafe fn dot_i_sim(col_indices: &[usize], values: &[f32], b: &[f32]) -> f32 {
    let (col0, col_sim, col1) = values.as_simd::<LANE_S>();
    let mut dot = 0.0f32;
    let mut dot_sim = Simd::<f32, LANE_S>::splat(0.0f32);
    let mut ind = 0usize;
    unsafe {
        for i in col0 {
            dot += i * b.get_unchecked(*col_indices.get_unchecked(ind));
            ind += 1;
        }
        for i_sim in col_sim {
            // note that 16 is hard coded here
            dot_sim += i_sim * sim_from_slice_index_f(b, &col_indices[ind..]);
            ind += LANE_S;
        }
        for i in col1 {
            dot += i * b.get_unchecked(*col_indices.get_unchecked(ind));
            ind += 1;
        }
    }
    dot += dot_sim.reduce_sum();
    dot
}

#[inline(always)]
unsafe fn dot_i_simple(col_indices: &[usize], col_values: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for (i, v) in col_indices.iter().zip(col_values.iter()) {
        dot += v * b.get_unchecked(*i);
    }
    dot
}

#[inline(always)]
unsafe fn sim_from_slice_16(slice: &[f32]) -> f32x16 {
    f32x16::from_array([
        *slice.get_unchecked(0),
        *slice.get_unchecked(1),
        *slice.get_unchecked(2),
        *slice.get_unchecked(3),
        *slice.get_unchecked(4),
        *slice.get_unchecked(5),
        *slice.get_unchecked(6),
        *slice.get_unchecked(7),
        *slice.get_unchecked(8),
        *slice.get_unchecked(9),
        *slice.get_unchecked(10),
        *slice.get_unchecked(11),
        *slice.get_unchecked(12),
        *slice.get_unchecked(13),
        *slice.get_unchecked(14),
        *slice.get_unchecked(15),
    ])
}

#[inline(always)]
unsafe fn sim_from_slice_index_f(full_slice: &[f32], index: &[usize]) -> f32x4 {
    f32x4::from_array([
        *full_slice.get_unchecked(*index.get_unchecked(0)),
        *full_slice.get_unchecked(*index.get_unchecked(1)),
        *full_slice.get_unchecked(*index.get_unchecked(2)),
        *full_slice.get_unchecked(*index.get_unchecked(3)),
    ])
}
