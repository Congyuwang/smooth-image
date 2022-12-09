use crate::ag_method::ag_method;
use crate::cg_method::cg_method;
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::io::{read_img, resize_img_mask_to_luma, write_png};
use crate::opt_utils::{matrix_a, matrix_d, psnr};
use crate::simd::CsrMatrixF32;
use image::{DynamicImage, GrayImage};
use nalgebra::DVector;
use nalgebra_sparse::ops::serial::{
    spadd_pattern, spmm_csr_dense, spmm_csr_pattern, spmm_csr_prealloc,
};
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::CsrMatrix;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt::Debug;
use std::path::Path;
use std::time::Instant;

pub enum OptAlgo {
    Cg,
    Ag,
}

pub enum InitType {
    Rand,
    Zero,
}

#[derive(Debug)]
pub struct RuntimeStats {
    pub start_time: Instant,
    pub image_read_time: Instant,
    pub matrix_generation_time: Instant,
    pub optimization_time: Instant,
    pub image_write_time: Instant,
    pub total_iteration: i32,
    pub psnr_history: Vec<(i32, f32)>,
}

/// return the generated GrayImage, and the number of iteration
pub fn run_inpaint<I, M, O>(
    (image, mask, out): (I, M, O),
    algo: OptAlgo,
    mu: f32,
    tol: f32,
    init_type: InitType,
    metric_step: i32,
) -> Result<RuntimeStats>
where
    I: AsRef<Path> + Debug,
    M: AsRef<Path> + Debug,
    O: AsRef<Path> + Debug,
{
    let start_time = Instant::now();
    let image = read_img(image)?;
    let mask = read_img(mask)?;
    let (img, mask) = resize_img_mask_to_luma(image, mask)?;
    let width = img.width() as usize;
    let height = img.height() as usize;
    let img = img.as_raw();
    let mask = mask.as_raw();
    let image_read_time = Instant::now();
    let orig = DVector::<f32>::from_vec(img.iter().map(|p| *p as f32 / 256.0).collect::<Vec<_>>());

    let x = gen_random_x(width, height, init_type);
    let (b_mat, c) = prepare_matrix(width, height, orig.as_slice(), mask, mu)?;
    let matrix_generation_time = Instant::now();
    let mut metrics = Vec::<(i32, f32)>::new();
    let metric_cb = |iter_round, inferred: &[f32], tol_var: f32| {
        let psnr = psnr(inferred, orig.as_slice());
        println!("iter={iter_round}, psnr={psnr}, tol_var={tol_var}");
        metrics.push((iter_round, psnr));
    };
    let b_mat = CsrMatrixF32::from(b_mat);

    let (output, iter_round) = match algo {
        OptAlgo::Cg => cg_method(&b_mat, c, x, tol, metric_step, metric_cb)?,
        OptAlgo::Ag => ag_method(&b_mat, c, mu, x, tol, metric_step, metric_cb)?,
    };
    let optimization_time = Instant::now();

    let pixels = output
        .iter()
        .map(|p| (*p * 256.0) as u8)
        .collect::<Vec<_>>();
    let img = GrayImage::from_vec(width as u32, height as u32, pixels).unwrap();
    write_png(out, &DynamicImage::ImageLuma8(img))?;
    let image_write_time = Instant::now();
    let runtime_stats = RuntimeStats {
        start_time,
        image_read_time,
        matrix_generation_time,
        optimization_time,
        image_write_time,
        total_iteration: iter_round,
        psnr_history: metrics,
    };

    Ok(runtime_stats)
}

pub fn gen_random_x(width: usize, height: usize, init_type: InitType) -> Vec<f32> {
    let size = width * height;

    match init_type {
        InitType::Rand => {
            let small_rng = SmallRng::from_entropy();
            let uniform = Uniform::<f32>::new(0.0, 1.0);
            uniform
                .sample_iter(small_rng)
                .take(size)
                .collect::<Vec<_>>()
        }
        InitType::Zero => {
            vec![0.0f32; size]
        }
    }
}

/// compute matrix B and vector c, init y
pub fn prepare_matrix(
    width: usize,
    height: usize,
    orig: &[f32],
    mask: &[u8],
    mu: f32,
) -> Result<(CsrMatrix<f32>, Vec<f32>)> {
    let size = width * height;
    if mask.len() != size || orig.len() != size {
        return Err(ErrorMessage(
            "unmatched lengths detected when executing algo".to_string(),
        ));
    }
    let (matrix_a, vector_b) = matrix_a(mask, orig);
    let matrix_d = matrix_d(width, height);

    // compute B with preallocate
    let pattern_ata = spmm_csr_pattern(&matrix_a.pattern().transpose(), matrix_a.pattern());
    let pattern_dtd = spmm_csr_pattern(&matrix_d.pattern().transpose(), matrix_d.pattern());
    let pattern = spadd_pattern(&pattern_ata, &pattern_dtd);
    let nnz = pattern.nnz();
    let mut b_mat = CsrMatrix::try_from_pattern_and_values(pattern, vec![0.0f32; nnz]).unwrap();
    // B += A^t * A
    if let Err(e) = spmm_csr_prealloc(
        0.,
        &mut b_mat,
        1.,
        Op::Transpose(&matrix_a),
        Op::NoOp(&matrix_a),
    ) {
        return Err(ErrorMessage(format!(
            "error computing matrix B step 1: {e:?}"
        )));
    }
    // B += mu * D^T * D
    if let Err(e) = spmm_csr_prealloc(
        1.,
        &mut b_mat,
        mu,
        Op::Transpose(&matrix_d),
        Op::NoOp(&matrix_d),
    ) {
        return Err(ErrorMessage(format!(
            "error computing matrix B step 2: {e:?}"
        )));
    }
    let mut c = DVector::zeros(size);
    spmm_csr_dense(
        0.0,
        &mut c,
        1.0,
        Op::Transpose(&matrix_a),
        Op::NoOp(&vector_b),
    );
    Ok((b_mat, Vec::from(c.as_slice())))
}
