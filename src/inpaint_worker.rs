use crate::accelerate::{CscMatrixF32, Property};
use crate::ag_method::ag_method;
use crate::cg_method::cg_method;
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::io::{read_img, resize_img_mask_to_luma, write_png};
use crate::opt_utils::{matrix_a, matrix_d, psnr};
use image::{DynamicImage, GrayImage};
use nalgebra::DVector;
use nalgebra_sparse::ops::serial::{
    spadd_pattern, spmm_csc_dense, spmm_csc_pattern, spmm_csc_prealloc,
};
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::CscMatrix;
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
    let (b_mat, c) = prepare_matrix(width, height, img, mask, mu)?;
    let matrix_generation_time = Instant::now();
    let mut metrics = Vec::<(i32, f32)>::new();
    let metric_cb = |iter_round, inferred: &DVector<f32>, tol_var: f32| {
        let psnr = psnr(inferred, &orig);
        println!("iter={iter_round}, psnr={psnr}, tol_var={tol_var}");
        metrics.push((iter_round, psnr));
    };

    let (output, iter_round) = match algo {
        OptAlgo::Cg => cg_method(&b_mat, c, x, tol, metric_step, metric_cb)?,
        OptAlgo::Ag => ag_method(&b_mat, c, mu, x, tol, metric_step, metric_cb)?,
    };
    let optimization_time = Instant::now();

    let pixels = output
        .data
        .as_slice()
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

pub fn gen_random_x(width: usize, height: usize, init_type: InitType) -> DVector<f32> {
    let size = width * height;
    let vec = match init_type {
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
    };
    DVector::from_vec(vec)
}

/// compute matrix B and vector c, init y
pub fn prepare_matrix(
    width: usize,
    height: usize,
    img: &[u8],
    mask: &[u8],
    mu: f32,
) -> Result<(CscMatrixF32, DVector<f32>)> {
    let size = width * height;
    if mask.len() != size || img.len() != size {
        return Err(ErrorMessage(
            "unmatched lengths detected when executing algo".to_string(),
        ));
    }
    let (matrix_a, vector_b) = matrix_a(mask, img);
    let matrix_d = matrix_d(width, height);

    // compute B with preallocate
    let pattern_ata = spmm_csc_pattern(&matrix_a.pattern().transpose(), matrix_a.pattern());
    let pattern_dtd = spmm_csc_pattern(&matrix_d.pattern().transpose(), matrix_d.pattern());
    let pattern = spadd_pattern(&pattern_ata, &pattern_dtd);
    let nnz = pattern.nnz();
    let mut b_mat = CscMatrix::try_from_pattern_and_values(pattern, vec![0.0f32; nnz]).unwrap();
    // B += A^t * A
    if let Err(e) = spmm_csc_prealloc(
        0.,
        &mut b_mat,
        1.,
        Op::Transpose(&matrix_a),
        Op::NoOp(&matrix_a),
    ) {
        return Err(ErrorMessage(format!(
            "error computing matrix B step 1: {:?}",
            e
        )));
    }
    // B += mu * D^T * D
    if let Err(e) = spmm_csc_prealloc(
        1.,
        &mut b_mat,
        mu,
        Op::Transpose(&matrix_d),
        Op::NoOp(&matrix_d),
    ) {
        return Err(ErrorMessage(format!(
            "error computing matrix B step 2: {:?}",
            e
        )));
    }
    let mut c = DVector::zeros(size);
    spmm_csc_dense(
        0.0,
        &mut c,
        1.0,
        Op::Transpose(&matrix_a),
        Op::NoOp(&vector_b),
    );
    let (val, idx, jdx) = {
        let mut vals = Vec::new();
        let mut idx = Vec::new();
        let mut jdx = Vec::new();
        let (pat, v) = b_mat.into_pattern_and_values();
        for (v, (i, j)) in v
            .into_iter()
            .zip(pat.entries())
            .filter(|(_, (i, j))| i >= j)
        {
            vals.push(v);
            idx.push(i as i64);
            jdx.push(j as i64);
        }
        (vals, idx, jdx)
    };
    let mut b_mat_accelerate = CscMatrixF32::new(size as u64, size as u64);
    b_mat_accelerate.set_property(Property::LowerSymmetric)?;
    b_mat_accelerate.insert_entries(&val, &idx, &jdx)?;
    b_mat_accelerate.commit()?;
    Ok((b_mat_accelerate, c))
}
