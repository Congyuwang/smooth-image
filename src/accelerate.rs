use crate::error::Error::ErrorMessage;
use crate::error::Result;
use accelerate_blas_sys as ffi;
use nalgebra::DVector;
use nalgebra_sparse::ops::Op;
use std::ffi::{c_int, c_void};

/// A f32 Csc Matrix
pub struct CscMatrixF32(ffi::sparse_matrix_float);

pub enum Property {
    _UpperTriangular,
    _LowerTriangular,
    _UpperSymmetric,
    LowerSymmetric,
}

impl CscMatrixF32 {
    pub fn new(rows: usize, cols: usize) -> Self {
        unsafe { CscMatrixF32(ffi::sparse_matrix_create_float(rows as u64, cols as u64)) }
    }

    pub fn set_property(&mut self, name: Property) -> Result<()> {
        unsafe {
            try_sparse(ffi::sparse_set_matrix_property(
                self.0 as *mut c_void,
                match name {
                    Property::_UpperTriangular => {
                        ffi::sparse_matrix_property_SPARSE_UPPER_TRIANGULAR
                    }
                    Property::_LowerTriangular => {
                        ffi::sparse_matrix_property_SPARSE_LOWER_TRIANGULAR
                    }
                    Property::_UpperSymmetric => ffi::sparse_matrix_property_SPARSE_UPPER_SYMMETRIC,
                    Property::LowerSymmetric => ffi::sparse_matrix_property_SPARSE_LOWER_SYMMETRIC,
                },
            ))
        }
    }

    pub fn insert_col(&mut self, j: usize, nz: usize, val: &[f32], jndx: &[usize]) -> Result<()> {
        unsafe {
            try_sparse(ffi::sparse_insert_col_float(
                self.0,
                j as i64,
                nz as u64,
                val.as_ptr(),
                jndx.as_ptr() as *const i64,
            ))
        }
    }

    pub fn nrows(&self) -> usize {
        unsafe { ffi::sparse_get_matrix_number_of_rows(self.0 as *mut c_void) as usize }
    }

    pub fn ncols(&self) -> usize {
        unsafe { ffi::sparse_get_matrix_number_of_columns(self.0 as *mut c_void) as usize }
    }

    /// commit after writing to it
    pub fn commit(&mut self) -> Result<()> {
        unsafe { try_sparse(ffi::sparse_commit(self.0 as *mut c_void)) }
    }
}

impl Drop for CscMatrixF32 {
    fn drop(&mut self) {
        unsafe {
            ffi::sparse_matrix_destroy(self.0 as *mut c_void);
        }
    }
}

///  c <- c + alpha * op(A) * b.
pub fn spmv_csc_dense(
    c: &mut DVector<f32>,
    alpha: f32,
    a: Op<&CscMatrixF32>,
    b: &DVector<f32>,
) -> Result<()> where
{
    let (trans_a, a) = match a {
        Op::NoOp(a) => (ffi::CBLAS_TRANSPOSE_CblasNoTrans, a),
        Op::Transpose(a) => (ffi::CBLAS_TRANSPOSE_CblasTrans, a),
    };
    unsafe {
        try_sparse(ffi::sparse_matrix_vector_product_dense_float(
            trans_a,
            alpha,
            a.0,
            b.as_ptr(),
            1,
            c.as_mut_ptr(),
            1,
        ))
    }
}

/// X -> Y
#[inline(always)]
pub fn scopy(src: &DVector<f32>, dest: &mut DVector<f32>) {
    let n = src.nrows().min(dest.nrows()) as c_int;
    unsafe {
        ffi::cblas_scopy(n, src.as_ptr(), 1, dest.as_mut_ptr(), 1);
    }
}

/// y <- alpha
#[inline(always)]
pub fn sset(alpha: f32, y: &mut DVector<f32>) {
    unsafe { ffi::catlas_sset(y.nrows() as i32, alpha, y.as_mut_ptr(), 1) }
}

#[inline(always)]
pub fn snorm(x: &DVector<f32>) -> f32 {
    unsafe { ffi::cblas_snrm2(x.nrows() as i32, x.as_ptr(), 1) }
}

#[inline(always)]
pub fn sdot(x: &DVector<f32>, y: &DVector<f32>) -> f32 {
    let n = x.nrows().min(y.nrows()) as c_int;
    unsafe { ffi::cblas_sdot(n, x.as_ptr(), 1, y.as_ptr(), 1) }
}

/// Y = (alpha * X) + Y
#[inline(always)]
pub fn saxpy(alpha: f32, x: &DVector<f32>, y: &mut DVector<f32>) {
    unsafe { ffi::cblas_saxpy(x.nrows() as c_int, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
}

/// Y = (alpha * X) + (beta * Y)
#[inline(always)]
pub fn saxpby(alpha: f32, x: &DVector<f32>, beta: f32, y: &mut DVector<f32>) {
    unsafe {
        ffi::catlas_saxpby(
            x.nrows() as c_int,
            alpha,
            x.as_ptr(),
            1,
            beta,
            y.as_mut_ptr(),
            1,
        )
    }
}

#[inline(always)]
fn try_sparse(status: ffi::sparse_status) -> Result<()> {
    match status {
        ffi::sparse_status_SPARSE_SUCCESS => Ok(()),
        ffi::sparse_status_SPARSE_ILLEGAL_PARAMETER => {
            Err(ErrorMessage("sparse illegal parameter".to_string()))
        }
        ffi::sparse_status_SPARSE_CANNOT_SET_PROPERTY => {
            Err(ErrorMessage("sparse cannot set property".to_string()))
        }
        ffi::sparse_status_SPARSE_SYSTEM_ERROR => {
            Err(ErrorMessage("sparse system error".to_string()))
        }
        _ => Err(ErrorMessage("sparse unknown error".to_string())),
    }
}
