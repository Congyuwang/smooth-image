use crate::error::Error::ErrorMessage;
use crate::error::Result;
use accelerate_blas_sys as ffi;
use nalgebra::{DVector, DVectorSlice, DVectorSliceMut};
use nalgebra_sparse::ops::Op;
use std::ffi::{c_int, c_void};

/// A f32 Csc Matrix
pub struct CscMatrixF32(ffi::sparse_matrix_float);

pub enum CscMatrixProperty {
    UpperTriangular,
    LowerTriangular,
    UpperSymmetric,
    LowerSymmetric,
}

impl CscMatrixF32 {
    pub fn new(rows: u64, cols: u64) -> Self {
        unsafe { CscMatrixF32(ffi::sparse_matrix_create_float(rows, cols)) }
    }

    pub fn insert(&mut self, val: f32, i: i64, j: i64) -> Result<()> {
        unsafe { try_sparse(ffi::sparse_insert_entry_float(self.0, val, i, j)) }
    }

    pub fn insert_entries(&mut self, val: &[f32], indx: &[i64], jndx: &[i64]) -> Result<()> {
        let n = val.len();
        if indx.len() != n || jndx.len() != n {
            return Err(ErrorMessage("entries/index length unequal".to_string()));
        }
        unsafe {
            try_sparse(ffi::sparse_insert_entries_float(
                self.0,
                n as u64,
                val.as_ptr(),
                indx.as_ptr(),
                jndx.as_ptr(),
            ))
        }
    }

    pub fn set_property(&mut self, name: CscMatrixProperty) -> Result<()> {
        unsafe {
            try_sparse(ffi::sparse_set_matrix_property(
                self.0 as *mut c_void,
                match name {
                    CscMatrixProperty::UpperTriangular => {
                        ffi::sparse_matrix_property_SPARSE_UPPER_TRIANGULAR
                    }
                    CscMatrixProperty::LowerTriangular => {
                        ffi::sparse_matrix_property_SPARSE_LOWER_TRIANGULAR
                    }
                    CscMatrixProperty::UpperSymmetric => {
                        ffi::sparse_matrix_property_SPARSE_UPPER_SYMMETRIC
                    }
                    CscMatrixProperty::LowerSymmetric => {
                        ffi::sparse_matrix_property_SPARSE_LOWER_SYMMETRIC
                    }
                },
            ))
        }
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
pub fn fast_spmv_csc_dense<'a>(
    c: impl Into<DVectorSliceMut<'a, f32>>,
    alpha: f32,
    a: Op<&CscMatrixF32>,
    b: impl Into<DVectorSlice<'a, f32>>,
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
            b.into().as_ptr(),
            1,
            c.into().as_mut_ptr(),
            1,
        ))
    }
}

/// X -> Y
pub fn copy_f32(src: &DVector<f32>, dest: &mut DVector<f32>) {
    let n = src.nrows().min(dest.nrows()) as c_int;
    unsafe {
        ffi::cblas_scopy(n, src.as_ptr(), 1, dest.as_mut_ptr(), 1);
    }
}

/// Y = (alpha * X) + Y
pub fn axpy_f32(alpha: f32, x: &DVector<f32>, y: &mut DVector<f32>) -> Result<()> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(ErrorMessage("X, Y length unequal".to_string()));
    }
    unsafe { ffi::cblas_saxpy(n as c_int, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
    Ok(())
}

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
