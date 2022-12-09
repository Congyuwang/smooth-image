use nalgebra_sparse::CsrMatrix;
use std::ops::{AddAssign, Neg, SubAssign};
use std::simd::{f32x16, Simd, SimdFloat, StdFloat};

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
    unsafe {
        for i in c0 {
            *i = i.mul_add(beta, dot_i(ind, a, b) * alpha);
            ind += 1;
        }
        for i_sim in c_sim {
            // note that 16 is hard coded here
            *i_sim = i_sim.mul_add(beta_sim, dot_i_sim(ind, a, b) * alpha_sim);
            ind += LANE;
        }
        for i in c1 {
            *i = i.mul_add(beta, dot_i(ind, a, b) * alpha);
            ind += 1;
        }
    }
}

/// c = c + Op(a) * b
pub fn matrix_vector_prod(c: &mut [f32], a: &CsrMatrixF32, b: &[f32]) {
    let (c0, c_sim, c1) = c.as_simd_mut::<LANE>();
    let mut ind = 0usize;
    unsafe {
        for i in c0 {
            *i += dot_i(ind, a, b);
            ind += 1;
        }
        for i_sim in c_sim {
            // note that 16 is hard coded here
            *i_sim = dot_i_sim(ind, a, b);
            ind += LANE;
        }
        for i in c1 {
            *i = dot_i(ind, a, b);
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
unsafe fn dot_i_sim(i: usize, a: &CsrMatrixF32, b: &[f32]) -> Simd<f32, LANE> {
    Simd::<f32, LANE>::from_array([
        dot_i(i, a, b),
        dot_i(i + 1, a, b),
        dot_i(i + 2, a, b),
        dot_i(i + 3, a, b),
        dot_i(i + 4, a, b),
        dot_i(i + 5, a, b),
        dot_i(i + 6, a, b),
        dot_i(i + 7, a, b),
        dot_i(i + 8, a, b),
        dot_i(i + 9, a, b),
        dot_i(i + 10, a, b),
        dot_i(i + 11, a, b),
        dot_i(i + 12, a, b),
        dot_i(i + 13, a, b),
        dot_i(i + 14, a, b),
        dot_i(i + 15, a, b),
    ])
}

#[inline(always)]
unsafe fn dot_i(i: usize, a: &CsrMatrixF32, b: &[f32]) -> f32 {
    let row_start = *a.row_offsets.get_unchecked(i);
    let row_end = *a.row_offsets.get_unchecked(i + 1);
    let col_indices = &a.col_indices[row_start..row_end];
    let col_values = &a.values[row_start..row_end];

    let (col0, col_sim, col1) = col_values.as_simd::<LANE>();

    let mut dot = 0.0f32;
    let mut dot_sim = Simd::<f32, LANE>::splat(0.0f32);
    let mut ind = 0usize;
    unsafe {
        for i in col0 {
            dot += i * b.get_unchecked(*col_indices.get_unchecked(ind));
            ind += 1;
        }
        for i_sim in col_sim {
            // note that 16 is hard coded here
            dot_sim += i_sim * sim_from_slice_index_16(b, &col_indices[ind..]);
            ind += LANE;
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

/// Warning: note the difference with `sim_from_slice_16`.
/// The first arg full_slice here has no offset.
///
/// index must have a length longer than 16
#[inline(always)]
unsafe fn sim_from_slice_index_16(full_slice: &[f32], index: &[usize]) -> f32x16 {
    f32x16::from_array([
        *full_slice.get_unchecked(*index.get_unchecked(0)),
        *full_slice.get_unchecked(*index.get_unchecked(1)),
        *full_slice.get_unchecked(*index.get_unchecked(2)),
        *full_slice.get_unchecked(*index.get_unchecked(3)),
        *full_slice.get_unchecked(*index.get_unchecked(4)),
        *full_slice.get_unchecked(*index.get_unchecked(5)),
        *full_slice.get_unchecked(*index.get_unchecked(6)),
        *full_slice.get_unchecked(*index.get_unchecked(7)),
        *full_slice.get_unchecked(*index.get_unchecked(8)),
        *full_slice.get_unchecked(*index.get_unchecked(9)),
        *full_slice.get_unchecked(*index.get_unchecked(10)),
        *full_slice.get_unchecked(*index.get_unchecked(11)),
        *full_slice.get_unchecked(*index.get_unchecked(12)),
        *full_slice.get_unchecked(*index.get_unchecked(13)),
        *full_slice.get_unchecked(*index.get_unchecked(14)),
        *full_slice.get_unchecked(*index.get_unchecked(15)),
    ])
}
