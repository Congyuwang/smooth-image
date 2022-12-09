use crate::ag_method::ag_method;
use crate::cg_method::cg_method;
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::image_format::{make_mask_image, ImageFormat, Mask};
use crate::io::{read_img, write_png};
use crate::opt_utils::{matrix_a, matrix_d, psnr};
use crate::simd::CsrMatrixF32;
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
use rayon::scope;
use std::fmt::Debug;
use std::path::Path;
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
    let (mask, image) = make_mask_image(mask, image)?;

    let image_read_time = Instant::now();

    let (b_mat, matrix_a) = prepare_matrix(&mask, mu)?;
    let b_mat = CsrMatrixF32::from(b_mat);

    let matrix_generation_time = Instant::now();

    struct ProcessOutput {
        output: ImageFormat,
        rounds: i32,
        metrics: Vec<(i32, f32)>,
    }

    let layer_worker = |thread_id: i32, layer: &[f32]| {
        let x = init_x(mask.width as usize, mask.height as usize, init_type);
        let mut metrics = Vec::<(i32, f32)>::new();
        let metric_cb = |iter_round, inferred: &[f32], tol_var: f32| {
            let psnr = psnr(inferred, layer);
            println!("thread={thread_id}, iter={iter_round}, psnr={psnr}, tol_var={tol_var}");
            metrics.push((iter_round, psnr));
        };
        let c = compute_c(&matrix_a, &mask, layer);
        let run_result = match algo {
            OptAlgo::Cg => cg_method(&b_mat, c, x, tol, metric_step, metric_cb),
            OptAlgo::Ag => ag_method(&b_mat, c, mu, x, tol, metric_step, metric_cb),
        };
        match run_result {
            Ok((output, iter_round)) => {
                Ok((output, iter_round, metrics))
            },
            Err(e) => Err(e),
        }
    };

    let output = match &image {
        ImageFormat::Luma { width, height, l } => {
            let (output, rounds, metrics) = layer_worker(0 , l.as_slice())?;
            ProcessOutput {
                output: ImageFormat::Luma {
                    width: *width,
                    height: *height,
                    l: output,
                },
                rounds,
                metrics,
            }
        }
        ImageFormat::LumaA {
            width,
            height,
            l,
            a,
        } => {
            let mut l_output = None;
            let mut a_output = None;
            scope(|s| {
                s.spawn(|_| {
                    l_output = Some(layer_worker(0, l.as_slice()));
                });
                s.spawn(|_| {
                    a_output = Some(layer_worker(1, a.as_slice()));
                })
            });
            let (l_output, l_rounds, l_metrics) = l_output.unwrap()?;
            let (a_output, a_rounds, a_metrics) = a_output.unwrap()?;
            ProcessOutput {
                output: ImageFormat::LumaA {
                    width: *width,
                    height: *height,
                    l: l_output,
                    a: a_output,
                },
                rounds: l_rounds.max(a_rounds),
                metrics: if l_rounds > a_rounds {
                    l_metrics
                } else {
                    a_metrics
                },
            }
        }
        ImageFormat::Rgb {
            width,
            height,
            r,
            g,
            b,
        } => {
            let mut r_output = None;
            let mut g_output = None;
            let mut b_output = None;
            scope(|s| {
                s.spawn(|_| {
                    r_output = Some(layer_worker(0, r.as_slice()));
                });
                s.spawn(|_| {
                    g_output = Some(layer_worker(1, g.as_slice()));
                });
                s.spawn(|_| {
                    b_output = Some(layer_worker(2, b.as_slice()));
                })
            });
            let (r_output, r_rounds, r_metrics) = r_output.unwrap()?;
            let (g_output, g_rounds, g_metrics) = g_output.unwrap()?;
            let (b_output, b_rounds, b_metrics) = b_output.unwrap()?;
            let max_rounds = r_rounds.max(g_rounds).max(b_rounds);
            ProcessOutput {
                output: ImageFormat::Rgb {
                    width: *width,
                    height: *height,
                    r: r_output,
                    g: g_output,
                    b: b_output,
                },
                rounds: max_rounds,
                metrics: if r_rounds == max_rounds {
                    r_metrics
                } else if g_rounds == max_rounds {
                    g_metrics
                } else {
                    b_metrics
                },
            }
        }
        ImageFormat::Rgba {
            width,
            height,
            r,
            g,
            b,
            a,
        } => {
            let mut r_output = None;
            let mut g_output = None;
            let mut b_output = None;
            let mut a_output = None;
            scope(|s| {
                s.spawn(|_| {
                    r_output = Some(layer_worker(0, r.as_slice()));
                });
                s.spawn(|_| {
                    g_output = Some(layer_worker(1, g.as_slice()));
                });
                s.spawn(|_| {
                    b_output = Some(layer_worker(2, b.as_slice()));
                });
                s.spawn(|_| {
                    a_output = Some(layer_worker(3, a.as_slice()));
                })
            });
            let (r_output, r_rounds, r_metrics) = r_output.unwrap()?;
            let (g_output, g_rounds, g_metrics) = g_output.unwrap()?;
            let (b_output, b_rounds, b_metrics) = b_output.unwrap()?;
            let (a_output, a_rounds, a_metrics) = a_output.unwrap()?;
            let max_rounds = r_rounds.max(g_rounds).max(b_rounds).max(a_rounds);
            ProcessOutput {
                output: ImageFormat::Rgba {
                    width: *width,
                    height: *height,
                    r: r_output,
                    g: g_output,
                    b: b_output,
                    a: a_output,
                },
                rounds: max_rounds,
                metrics: if r_rounds == max_rounds {
                    r_metrics
                } else if g_rounds == max_rounds {
                    g_metrics
                } else if b_rounds == max_rounds {
                    b_metrics
                } else {
                    a_metrics
                },
            }
        }
    };

    let optimization_time = Instant::now();

    write_png(out, &output.output.to_img())?;

    let image_write_time = Instant::now();

    let runtime_stats = RuntimeStats {
        start_time,
        image_read_time,
        matrix_generation_time,
        optimization_time,
        image_write_time,
        total_iteration: output.rounds,
        psnr_history: output.metrics,
    };

    Ok(runtime_stats)
}

pub fn init_x(width: usize, height: usize, init_type: InitType) -> Vec<f32> {
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

/// compute matrix B and matrix A
pub fn prepare_matrix(mask: &Mask, mu: f32) -> Result<(CsrMatrix<f32>, CsrMatrix<f32>)> {
    let matrix_a = matrix_a(mask);
    let matrix_d = matrix_d(mask.width as usize, mask.height as usize);

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
    Ok((b_mat, matrix_a))
}

fn compute_c(matrix_a: &CsrMatrix<f32>, mask: &Mask, layer: &[f32]) -> Vec<f32> {
    let vector_b = DVector::<f32>::from_vec(
        layer
            .iter()
            .zip(mask.mask.iter())
            .filter_map(|(p, m)| if *m { Some(*p) } else { None })
            .collect(),
    );
    // c = A^T * b
    let mut c = DVector::zeros((mask.width * mask.height) as usize);
    spmm_csr_dense(
        0.0,
        &mut c,
        1.0,
        Op::Transpose(matrix_a),
        Op::NoOp(&vector_b),
    );
    Vec::from(c.as_slice())
}
