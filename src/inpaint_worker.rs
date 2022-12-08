use crate::accelerate::{CsrMatrixF32, Property};
use crate::ag_method::ag_method;
use crate::cg_method::cg_method;
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::io::{read_img, resize_img_to_luma_layer, write_png};
use crate::opt_utils::{matrix_a, matrix_d, psnr};
use image::{DynamicImage, GrayImage, RgbaImage};
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
use std::sync::{Arc, RwLock};
use std::thread::spawn;
use std::time::Instant;

#[derive(Copy, Clone)]
pub enum OptAlgo {
    Cg,
    Ag,
}

#[derive(Copy, Clone)]
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
    let (img, mask) = resize_img_to_luma_layer(image, mask)?;
    let width = mask.width() as usize;
    let height = mask.height() as usize;
    let image_read_time = Instant::now();

    let (b_mat, rgba) = prepare_matrix(width, height, &img, mask.as_raw(), mu)?;
    let matrix_generation_time = Instant::now();
    let b_mat = Arc::new(RwLock::new(b_mat));

    let handles = rgba
        .into_iter()
        .zip(img.into_iter())
        .map(|(layer, orig)| {
            let b_mat = b_mat.clone();
            spawn(move || {
                let orig = DVector::from_vec(
                    orig.into_iter()
                        .map(|p| *p as f32 / 256.0)
                        .collect::<Vec<_>>(),
                );
                let mut metrics = Vec::<(i32, f32)>::new();
                let metric_cb = |iter_round, inferred: &DVector<f32>, tol_var: f32| {
                    let psnr = psnr(inferred, &orig);
                    println!("iter={iter_round}, psnr={psnr}, tol_var={tol_var}");
                    metrics.push((iter_round, psnr));
                };
                let x = gen_random_x(width, height, &init_type);
                let b_mat_lock = b_mat.read().unwrap();
                let result = match algo {
                    OptAlgo::Cg => cg_method(&b_mat_lock, layer, x, tol, metric_step, metric_cb),
                    OptAlgo::Ag => ag_method(&b_mat_lock, layer, mu, x, tol, metric_step, metric_cb),
                };
                (result, metrics)
            })
        })
        .collect::<Vec<_>>();
    let mut iter_counts = Vec::with_capacity(4);
    let mut layers = Vec::with_capacity(4);
    let mut metrics_list = Vec::with_capacity(4);
    for handle in handles {
        let (result, metrics) = handle
            .join()
            .map_err(|_| ErrorMessage("exection failure".to_string()))?;
        let (layer, iter) = result?;
        iter_counts.push(iter);
        layers.push(layer);
        metrics_list.push(metrics);
    }
    let optimization_time = Instant::now();

    let r = layers[0].as_slice();
    let g = layers[1].as_slice();
    let b = layers[2].as_slice();
    let a = layers[3].as_slice();

    let raw_image = r
        .iter()
        .zip(g.iter().zip(b.iter().zip(a.iter())))
        .flat_map(|(r, (g, (b, a)))| {
            [
                (r * 256.0) as u8,
                (g * 256.0) as u8,
                (b * 256.0) as u8,
                (a * 256.0) as u8,
            ]
        })
        .collect::<Vec<_>>();

    let img = RgbaImage::from_raw(width as u32, height as u32, raw_image).unwrap();
    write_png(out, &DynamicImage::ImageRgba8(img))?;
    let image_write_time = Instant::now();

    let longest_round = iter_counts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(index, _)| index)
        .unwrap();

    let runtime_stats = RuntimeStats {
        start_time,
        image_read_time,
        matrix_generation_time,
        optimization_time,
        image_write_time,
        total_iteration: iter_counts[longest_round],
        psnr_history: metrics_list[longest_round].clone(),
    };

    Ok(runtime_stats)
}

pub fn gen_random_x(width: usize, height: usize, init_type: &InitType) -> DVector<f32> {
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
    img: &[GrayImage; 4],
    mask: &[u8],
    mu: f32,
) -> Result<(CsrMatrixF32, [DVector<f32>; 4])> {
    let size = width * height;
    if mask.len() != size {
        return Err(ErrorMessage(
            "unmatched lengths detected when executing algo".to_string(),
        ));
    }
    let (matrix_a, vec_b) = matrix_a(mask, &img);
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
            "error computing matrix B step 1: {:?}",
            e
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
            "error computing matrix B step 2: {:?}",
            e
        )));
    }
    let [vr, vg, vb, va] = vec_b;
    let mut r = DVector::zeros(size);
    let mut g = DVector::zeros(size);
    let mut b = DVector::zeros(size);
    let mut a = DVector::zeros(size);
    spmm_csr_dense(
        0.0,
        &mut r,
        1.0,
        Op::Transpose(&matrix_a),
        Op::NoOp(&DVector::from_vec(vr)),
    );

    spmm_csr_dense(
        0.0,
        &mut g,
        1.0,
        Op::Transpose(&matrix_a),
        Op::NoOp(&DVector::from_vec(vg)),
    );

    spmm_csr_dense(
        0.0,
        &mut b,
        1.0,
        Op::Transpose(&matrix_a),
        Op::NoOp(&DVector::from_vec(vb)),
    );

    spmm_csr_dense(
        0.0,
        &mut a,
        1.0,
        Op::Transpose(&matrix_a),
        Op::NoOp(&DVector::from_vec(va)),
    );

    let b_mat = {
        let mut b_mat_apple = CsrMatrixF32::new(size, size);
        b_mat_apple.set_property(Property::LowerSymmetric)?;
        for (i, row) in b_mat.row_iter().enumerate() {
            // symmetric, doesn't matter
            b_mat_apple.insert_row(i, row.nnz(), row.values(), row.col_indices())?;
        }
        b_mat_apple.commit()?;
        b_mat_apple
    };

    Ok((b_mat, [r, g, b, a]))
}
