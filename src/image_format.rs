use crate::error::Error::ErrorMessage;
use crate::error::Result;
use image::imageops::FilterType;
use image::{DynamicImage as Im, DynamicImage};
use image::{GrayAlphaImage, GrayImage, RgbImage, RgbaImage};

const PX_MAX: f32 = 256.0f32;

pub fn make_mask_image(mask: Im, image: Im) -> Result<(Mask, ImageFormat)> {
    let width = image.width();
    let height = image.height();
    let size = (width * height) as usize;
    let mask = make_mask(mask, width, height)?;
    let img = match image {
        Im::ImageLuma8(i) => ImageFormat::Luma {
            height,
            width,
            l: i.as_flat_samples()
                .samples
                .iter()
                .map(|i| *i as f32 / PX_MAX)
                .collect(),
        },
        Im::ImageLumaA8(i) => {
            let mut l = Vec::with_capacity(size);
            let mut a = Vec::with_capacity(size);
            i.pixels().for_each(|p| {
                let [pl, pa] = p.0;
                l.push(pl as f32 / PX_MAX);
                a.push(pa as f32 / PX_MAX);
            });
            ImageFormat::LumaA {
                height,
                width,
                l,
                a,
            }
        }
        Im::ImageRgb8(i) => {
            let mut r = Vec::with_capacity(size);
            let mut g = Vec::with_capacity(size);
            let mut b = Vec::with_capacity(size);
            i.pixels().for_each(|p| {
                let [pr, pg, pb] = p.0;
                r.push(pr as f32 / PX_MAX);
                g.push(pg as f32 / PX_MAX);
                b.push(pb as f32 / PX_MAX);
            });
            ImageFormat::Rgb {
                height,
                width,
                r,
                g,
                b,
            }
        }
        Im::ImageRgba8(i) => {
            let mut r = Vec::with_capacity(size);
            let mut g = Vec::with_capacity(size);
            let mut b = Vec::with_capacity(size);
            let mut a = Vec::with_capacity(size);
            i.pixels().for_each(|p| {
                let [pr, pg, pb, pa] = p.0;
                r.push(pr as f32 / PX_MAX);
                g.push(pg as f32 / PX_MAX);
                b.push(pb as f32 / PX_MAX);
                a.push(pa as f32 / PX_MAX)
            });
            ImageFormat::Rgba {
                height,
                width,
                r,
                g,
                b,
                a,
            }
        }
        _ => return Err(ErrorMessage("unsupported image format".to_string())),
    };
    Ok((mask, img))
}

pub fn make_mask(mask: Im, width: u32, height: u32) -> Result<Mask> {
    let mask = if mask.width() != width || mask.height() != height {
        mask.resize_exact(width, height, FilterType::Nearest)
    } else {
        mask
    };
    let mut nnz = 0;
    let mask = match mask {
        DynamicImage::ImageLuma8(i) => i
            .pixels()
            .map(|px| {
                let pass_through = px.0 != [0; 1];
                if pass_through {
                    nnz += 1;
                };
                pass_through
            })
            .collect(),
        _ => return Err(ErrorMessage("unsupported mask format, only allow luma8".to_string())),
    };
    Ok(Mask {
        height,
        width,
        nnz,
        mask,
    })
}

/// `true` to reserve
/// `false` to block
pub struct Mask {
    pub height: u32,
    pub width: u32,
    pub nnz: usize,
    pub mask: Vec<bool>,
}

pub enum ImageFormat {
    Luma {
        height: u32,
        width: u32,
        l: Vec<f32>,
    },
    LumaA {
        height: u32,
        width: u32,
        l: Vec<f32>,
        a: Vec<f32>,
    },
    Rgb {
        height: u32,
        width: u32,
        r: Vec<f32>,
        g: Vec<f32>,
        b: Vec<f32>,
    },
    Rgba {
        height: u32,
        width: u32,
        r: Vec<f32>,
        g: Vec<f32>,
        b: Vec<f32>,
        a: Vec<f32>,
    },
}

impl ImageFormat {
    pub fn to_img(&self) -> Im {
        match self {
            ImageFormat::Luma { height, width, l } => Im::ImageLuma8(
                GrayImage::from_vec(
                    *width,
                    *height,
                    l.iter().map(|p| (p * PX_MAX) as u8).collect(),
                )
                .unwrap(),
            ),
            ImageFormat::LumaA {
                height,
                width,
                l,
                a,
            } => Im::ImageLumaA8(
                GrayAlphaImage::from_vec(
                    *width,
                    *height,
                    l.iter()
                        .zip(a.iter())
                        .flat_map(|(l, a)| [(l * PX_MAX) as u8, (a * PX_MAX) as u8])
                        .collect(),
                )
                .unwrap(),
            ),
            ImageFormat::Rgb {
                height,
                width,
                r,
                g,
                b,
            } => Im::ImageRgb8(
                RgbImage::from_vec(
                    *width,
                    *height,
                    r.iter()
                        .zip(g.iter())
                        .zip(b.iter())
                        .flat_map(|((r, g), b)| {
                            [(r * PX_MAX) as u8, (g * PX_MAX) as u8, (b * PX_MAX) as u8]
                        })
                        .collect(),
                )
                .unwrap(),
            ),
            ImageFormat::Rgba {
                height,
                width,
                r,
                g,
                b,
                a,
            } => Im::ImageRgba8(
                RgbaImage::from_vec(
                    *width,
                    *height,
                    r.iter()
                        .zip(g.iter())
                        .zip(b.iter())
                        .zip(a.iter())
                        .flat_map(|(((r, g), b), a)| {
                            [
                                (r * PX_MAX) as u8,
                                (g * PX_MAX) as u8,
                                (b * PX_MAX) as u8,
                                (a * PX_MAX) as u8,
                            ]
                        })
                        .collect(),
                )
                .unwrap(),
            ),
        }
    }
}
