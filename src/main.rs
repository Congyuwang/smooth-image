#![feature(portable_simd)]
#![feature(get_many_mut)]

use crate::inpaint_worker::{run_inpaint, InitType, OptAlgo, RuntimeStats};
use crate::mask_painter::produce_gray_mask_image;
use clap::{Args, Parser, Subcommand};
use humantime::FormattedDuration;
use tabled::{Table, Tabled};

mod ag_method;
mod cg_method;
mod error;
mod image_format;
mod inpaint_worker;
mod io;
mod mask_painter;
mod opt_utils;
mod simd;

#[derive(Parser)]
#[command(name = "Assignment4")]
#[command(author = "Congyu Wang. <116020237@link.cuhk.edu.cn>")]
#[command(version = "1.0")]
#[command(about = "Solve Assignment 4 and process images with Rust", long_about = None)]
struct Tasks {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Problem a, mask image
    MaskImage(MaskImage),
    /// Problem e, inpaint algorithm
    InPaint(InPaint),
}

#[derive(Args)]
struct MaskImage {
    #[arg(short, long)]
    image: String,
    #[arg(short, long)]
    mask: String,
    #[arg(short, long)]
    output: String,
}

#[derive(Args)]
struct InPaint {
    #[arg(short, long)]
    image: String,
    #[arg(short, long)]
    mask: String,
    #[arg(short, long)]
    output: String,
    #[arg(long)]
    algo: String,
    #[arg(long)]
    init: String,
    #[arg(long)]
    tol: f32,
    #[arg(long)]
    mu: f32,
    /// negative step means no output
    #[arg(long, default_value_t = 10)]
    metric_step: i32,
}

#[derive(Tabled)]
pub struct IterStat {
    iter_round: i32,
    psnr: f32,
}

#[derive(Tabled)]
pub struct StatsDisplay {
    total_iteration: i32,
    image_read_time: FormattedDuration,
    matrix_generation_time: FormattedDuration,
    optimization_time: FormattedDuration,
    image_write_time: FormattedDuration,
    average_iteration_time: FormattedDuration,
    total_time: FormattedDuration,
}

impl From<RuntimeStats> for StatsDisplay {
    fn from(s: RuntimeStats) -> Self {
        let image_read_time = s.image_read_time - s.start_time;
        let matrix_generation_time = s.matrix_generation_time - s.image_read_time;
        let optimization_time = s.optimization_time - s.matrix_generation_time;
        let image_write_time = s.image_write_time - s.optimization_time;
        let total_time = s.image_write_time - s.start_time;
        let average_iteration_time = optimization_time / (s.total_iteration as u32);
        StatsDisplay {
            total_iteration: s.total_iteration,
            image_read_time: humantime::format_duration(image_read_time),
            matrix_generation_time: humantime::format_duration(matrix_generation_time),
            optimization_time: humantime::format_duration(optimization_time),
            image_write_time: humantime::format_duration(image_write_time),
            average_iteration_time: humantime::format_duration(average_iteration_time),
            total_time: humantime::format_duration(total_time),
        }
    }
}

fn main() {
    let task: Tasks = Tasks::parse();

    match &task.task {
        Task::MaskImage(mask_img) => {
            if let Err(e) =
                produce_gray_mask_image(&mask_img.image, &mask_img.mask, &mask_img.output)
            {
                println!("Error producing image: {e:?}");
            }
        }
        Task::InPaint(inpaint) => {
            let init = match inpaint.init.as_str() {
                "zero" => InitType::Zero,
                "random" => InitType::Rand,
                _ => {
                    println!("unknown init type, choose `zero` or `random`");
                    return;
                }
            };
            let algo = match inpaint.algo.as_str() {
                "ag" => OptAlgo::Ag,
                "cg" => OptAlgo::Cg,
                _ => {
                    println!("unknown algo, choose `ag` or `cg`.");
                    return;
                }
            };
            match run_inpaint(
                (&inpaint.image, &inpaint.mask, &inpaint.output),
                algo,
                inpaint.mu,
                inpaint.tol,
                init,
                inpaint.metric_step,
            ) {
                Err(e) => {
                    println!("Error executing inpaint: {e:?}");
                }
                Ok(stats) => {
                    let metric_table =
                        Table::new(stats.psnr_history.iter().map(|(i, m)| IterStat {
                            iter_round: *i,
                            psnr: *m,
                        }));
                    let stats = StatsDisplay::from(stats);
                    let stats_table = Table::new(vec![stats]);
                    println!("++ Run Stats ++");
                    println!("{stats_table}");
                    if inpaint.metric_step > 0 {
                        println!("++ Metric History ++");
                        println!("{metric_table}");
                    }
                }
            }
        }
    }
}
