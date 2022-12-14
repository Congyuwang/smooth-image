use crate::ag_method::ag_method;
use crate::cg_method::cg_method;
use crate::error::Error::ErrorMessage;
use crate::error::Result;
use crate::gpu_metal::GpuLib;
use crate::image_format::{make_mask_image, ImageFormat, Mask, PX_MAX};
use crate::io::{read_img, write_png};
use crate::opt_utils::{matrix_a, matrix_d, psnr};
use metal::Buffer;
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
use std::sync::{Arc, Mutex};
use std::thread::scope;
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

enum ImageBuffer {
    Luma {
        width: u32,
        height: u32,
        l_original: Buffer,
        l_c: Buffer,
    },
    LumaA {
        width: u32,
        height: u32,
        l_original: Buffer,
        l_c: Buffer,
        a_original: Buffer,
        a_c: Buffer,
    },
    Rgb {
        width: u32,
        height: u32,
        r_original: Buffer,
        r_c: Buffer,
        g_original: Buffer,
        g_c: Buffer,
        b_original: Buffer,
        b_c: Buffer,
    },
    Rgba {
        width: u32,
        height: u32,
        r_original: Buffer,
        r_c: Buffer,
        g_original: Buffer,
        g_c: Buffer,
        b_original: Buffer,
        b_c: Buffer,
        a_original: Buffer,
        a_c: Buffer,
    },
}

pub struct CsrMatrixF16 {
    nrows: usize,
    ncols: usize,
    // length = n_row + 1
    pub row_offsets: Buffer,
    // length = nnz
    pub col_indices: Buffer,
    // length = nnz
    pub values: Buffer,
}

unsafe impl Send for CsrMatrixF16 {}

impl CsrMatrixF16 {
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

impl CsrMatrixF16 {
    pub fn from_mat(value: CsrMatrix<f32>, gpu: &GpuLib) -> Self {
        let nrows = value.nrows();
        let ncols = value.ncols();
        let (row_offsets, col_indices, values) = value.disassemble();
        let (row_offsets, cmd_r) = gpu.private_buffer_u32(&row_offsets);
        let (col_indices, cmd_c) = gpu.private_buffer_u32(&col_indices);
        let (values, cmd_v) = gpu.private_buffer_f16(&values);
        cmd_r.wait_until_completed();
        cmd_c.wait_until_completed();
        cmd_v.wait_until_completed();
        Self {
            nrows,
            ncols,
            row_offsets,
            col_indices,
            values,
        }
    }
}

/// return the generated GrayImage, and the number of iteration
pub fn run_inpaint<I, M, O>(
    (image, mask, out): (I, M, O),
    algo: OptAlgo,
    (mu, tol): (f32, f32),
    init_type: InitType,
    metric_step: i32,
    color: bool,
    gpu: GpuLib,
) -> Result<RuntimeStats>
where
    I: AsRef<Path> + Debug,
    M: AsRef<Path> + Debug,
    O: AsRef<Path> + Debug,
{
    let start_time = Instant::now();
    let mut image = read_img(image)?;
    if !color {
        image = image.grayscale();
    }
    let mask = read_img(mask)?;
    let (mask, image) = make_mask_image(mask, image)?;
    let size = (mask.width * mask.height) as usize;
    let image_read_time = Instant::now();

    let (b_mat, matrix_a) = prepare_matrix(&mask, mu)?;
    let b_mat = Arc::new(Mutex::new(CsrMatrixF16::from_mat(b_mat, &gpu)));
    let img_buffers = make_image_buffer_buffer(image, mask, matrix_a, &gpu);

    let matrix_generation_time = Instant::now();

    struct ProcessOutput {
        output: ImageFormat,
        rounds: i32,
        metrics: Vec<(i32, f32)>,
    }

    struct SendBuffer(Buffer);

    unsafe impl Send for SendBuffer {}

    let gpu = Arc::new(Mutex::new(gpu));

    let layer_worker = |thread_id: i32, c: SendBuffer, layer: SendBuffer| {
        let mut metrics = Vec::<(i32, f32)>::new();
        let x = init_x(size, init_type, &gpu);
        let metric_cb = |iter_round, diff_squared: f32, tol_var: f32| {
            let psnr = psnr(size, diff_squared);
            println!("thread={thread_id}, iter={iter_round}, psnr={psnr}, tol_var={tol_var}");
            metrics.push((iter_round, psnr));
        };
        let run_result = match algo {
            OptAlgo::Cg => cg_method(&b_mat, (c.0, layer.0, x), tol, metric_step, metric_cb, &gpu),
            OptAlgo::Ag => ag_method(&b_mat, (c.0, layer.0, x), mu, tol, metric_step, metric_cb, &gpu),
        };
        match run_result {
            Ok((output, iter_round)) => Ok((output, iter_round, metrics)),
            Err(e) => Err(e),
        }
    };

    let output = match img_buffers {
        ImageBuffer::Luma {
            width,
            height,
            l_c,
            l_original,
        } => {
            let l_c = SendBuffer(l_c);
            let l_original = SendBuffer(l_original);
            let (output, rounds, metrics) = layer_worker(0, l_c, l_original)?;
            ProcessOutput {
                output: ImageFormat::Luma {
                    width,
                    height,
                    l: output,
                },
                rounds,
                metrics,
            }
        }
        ImageBuffer::LumaA {
            width,
            height,
            l_c,
            l_original,
            a_c,
            a_original,
        } => {
            let l_c = SendBuffer(l_c);
            let l_original = SendBuffer(l_original);
            let a_c = SendBuffer(a_c);
            let a_original = SendBuffer(a_original);
            let (l_output, a_output) = scope(|s| {
                let l_output = s.spawn(|| layer_worker(0, l_c, l_original));
                let a_output = s.spawn(|| layer_worker(1, a_c, a_original));
                (l_output.join(), a_output.join())
            });
            let (l_output, l_rounds, l_metrics) = l_output.unwrap()?;
            let (a_output, a_rounds, a_metrics) = a_output.unwrap()?;
            ProcessOutput {
                output: ImageFormat::LumaA {
                    width,
                    height,
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
        ImageBuffer::Rgb {
            width,
            height,
            r_c,
            r_original,
            g_c,
            g_original,
            b_c,
            b_original,
        } => {
            let r_c = SendBuffer(r_c);
            let r_original = SendBuffer(r_original);
            let g_c = SendBuffer(g_c);
            let g_original = SendBuffer(g_original);
            let b_c = SendBuffer(b_c);
            let b_original = SendBuffer(b_original);
            let (r_output, g_output, b_output) = scope(|s| {
                let r_output = s.spawn(|| layer_worker(0, r_c, r_original));
                let g_output = s.spawn(|| layer_worker(1, g_c, g_original));
                let b_output = s.spawn(|| layer_worker(2, b_c, b_original));
                (r_output.join(), g_output.join(), b_output.join())
            });
            let (r_output, r_rounds, r_metrics) = r_output.unwrap()?;
            let (g_output, g_rounds, g_metrics) = g_output.unwrap()?;
            let (b_output, b_rounds, b_metrics) = b_output.unwrap()?;
            let max_rounds = r_rounds.max(g_rounds).max(b_rounds);
            ProcessOutput {
                output: ImageFormat::Rgb {
                    width,
                    height,
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
        ImageBuffer::Rgba {
            width,
            height,
            r_c,
            r_original,
            g_c,
            g_original,
            b_c,
            b_original,
            a_c,
            a_original,
        } => {
            let r_c = SendBuffer(r_c);
            let r_original = SendBuffer(r_original);
            let g_c = SendBuffer(g_c);
            let g_original = SendBuffer(g_original);
            let b_c = SendBuffer(b_c);
            let b_original = SendBuffer(b_original);
            let a_c = SendBuffer(a_c);
            let a_original = SendBuffer(a_original);
            let (r_output, g_output, b_output, a_output) = scope(|s| {
                let r_output = s.spawn(|| layer_worker(0, r_c, r_original));
                let g_output = s.spawn(|| layer_worker(1, g_c, g_original));
                let b_output = s.spawn(|| layer_worker(2, b_c, b_original));
                let a_output = s.spawn(|| layer_worker(3, a_c, a_original));
                (
                    r_output.join(),
                    g_output.join(),
                    b_output.join(),
                    a_output.join(),
                )
            });
            let (r_output, r_rounds, r_metrics) = r_output.unwrap()?;
            let (g_output, g_rounds, g_metrics) = g_output.unwrap()?;
            let (b_output, b_rounds, b_metrics) = b_output.unwrap()?;
            let (a_output, a_rounds, a_metrics) = a_output.unwrap()?;
            let max_rounds = r_rounds.max(g_rounds).max(b_rounds).max(a_rounds);
            ProcessOutput {
                output: ImageFormat::Rgba {
                    width,
                    height,
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

    write_png(out, &output.output.into_img())?;

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

pub fn init_x(size: usize, init_type: InitType, gpu: &Arc<Mutex<GpuLib>>) -> Buffer {
    match init_type {
        InitType::Rand => {
            let small_rng = SmallRng::from_entropy();
            let uniform = Uniform::<f32>::new(0.0, 1.0);
            let v = uniform
                .sample_iter(small_rng)
                .take(size)
                .collect::<Vec<_>>();
            let lock = gpu.lock().unwrap();
            let (buf, cmd) = lock.private_buffer_f16(&v);
            cmd.wait_until_completed();
            buf
        }
        InitType::Zero => {
            let lock = gpu.lock().unwrap();
            let (buf, cmd) = lock.private_buffer_f16(&vec![0.0f32; size]);
            cmd.wait_until_completed();
            buf
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

fn compute_c(matrix_a: &CsrMatrix<f32>, mask: &Mask, layer: &[u8], gpu: &GpuLib) -> Buffer {
    let vector_b = DVector::<f32>::from_vec(
        layer
            .iter()
            .zip(mask.mask.iter())
            .filter_map(|(p, m)| if *m { Some(*p as f32 / PX_MAX) } else { None })
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
    let (buffer, cmd) = gpu.private_buffer_f16(c.as_slice());
    cmd.wait_until_completed();
    buffer
}

fn make_image_buffer_buffer(
    image: ImageFormat,
    mask: Mask,
    matrix_a: CsrMatrix<f32>,
    gpu: &GpuLib,
) -> ImageBuffer {
    match image {
        ImageFormat::Luma { l, width, height } => {
            let (l_original, l_cmd) = gpu.private_buffer_f16_from_u8(&l);
            l_cmd.wait_until_completed();
            ImageBuffer::Luma {
                width,
                height,
                l_original,
                l_c: compute_c(&matrix_a, &mask, &l, gpu),
            }
        }
        ImageFormat::LumaA { l, a, width, height } => {
            let (l_original, l_cmd) = gpu.private_buffer_f16_from_u8(&l);
            let (a_original, a_cmd) = gpu.private_buffer_f16_from_u8(&a);
            l_cmd.wait_until_completed();
            a_cmd.wait_until_completed();
            ImageBuffer::LumaA {
                width,
                height,
                l_c: compute_c(&matrix_a, &mask, &l, gpu),
                l_original,
                a_c: compute_c(&matrix_a, &mask, &a, gpu),
                a_original,
            }
        }
        ImageFormat::Rgb { r, g, b, width, height } => {
            let (r_original, r_cmd) = gpu.private_buffer_f16_from_u8(&r);
            let (g_original, g_cmd) = gpu.private_buffer_f16_from_u8(&g);
            let (b_original, b_cmd) = gpu.private_buffer_f16_from_u8(&b);
            r_cmd.wait_until_completed();
            g_cmd.wait_until_completed();
            b_cmd.wait_until_completed();
            ImageBuffer::Rgb {
                width,
                height,
                r_c: compute_c(&matrix_a, &mask, &r, gpu),
                r_original,
                g_c: compute_c(&matrix_a, &mask, &g, gpu),
                g_original,
                b_c: compute_c(&matrix_a, &mask, &b, gpu),
                b_original,
            }
        }
        ImageFormat::Rgba { r, g, b, a, width, height } => {
            let (r_original, r_cmd) = gpu.private_buffer_f16_from_u8(&r);
            let (g_original, g_cmd) = gpu.private_buffer_f16_from_u8(&g);
            let (b_original, b_cmd) = gpu.private_buffer_f16_from_u8(&b);
            let (a_original, a_cmd) = gpu.private_buffer_f16_from_u8(&a);
            r_cmd.wait_until_completed();
            g_cmd.wait_until_completed();
            b_cmd.wait_until_completed();
            a_cmd.wait_until_completed();
            ImageBuffer::Rgba {
                width,
                height,
                r_c: compute_c(&matrix_a, &mask, &r, gpu),
                r_original,
                g_c: compute_c(&matrix_a, &mask, &g, gpu),
                g_original,
                b_c: compute_c(&matrix_a, &mask, &b, gpu),
                b_original,
                a_c: compute_c(&matrix_a, &mask, &a, gpu),
                a_original,
            }
        }
    }
}
