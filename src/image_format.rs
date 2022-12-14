use crate::error::Error::ErrorMessage;
use crate::error::Result;
use image::imageops::FilterType;
use image::{DynamicImage as Im, DynamicImage};
use image::{GrayAlphaImage, GrayImage, RgbImage, RgbaImage};

pub const PX_MAX: f32 = 256.0f32;

pub fn make_mask_image(mask: Im, image: Im) -> Result<(Mask, ImageFormat)> {
    let width = image.width();
    let height = image.height();
    let size = (width * height) as usize;
    let mask = make_mask(mask, width, height)?;
    let img = match image {
        Im::ImageLuma8(i) => ImageFormat::Luma {
            height,
            width,
            l: i.as_flat_samples().samples.iter().copied().collect(),
        },
        Im::ImageLumaA8(i) => {
            let mut l = Vec::with_capacity(size);
            let mut a = Vec::with_capacity(size);
            i.pixels().for_each(|p| {
                let [pl, pa] = p.0;
                l.push(pl);
                a.push(pa);
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
                r.push(pr);
                g.push(pg);
                b.push(pb);
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
                r.push(pr);
                g.push(pg);
                b.push(pb);
                a.push(pa)
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
        _ => {
            return Err(ErrorMessage(
                "unsupported mask format, only allow luma8".to_string(),
            ))
        }
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
        l: Vec<u8>,
    },
    LumaA {
        height: u32,
        width: u32,
        l: Vec<u8>,
        a: Vec<u8>,
    },
    Rgb {
        height: u32,
        width: u32,
        r: Vec<u8>,
        g: Vec<u8>,
        b: Vec<u8>,
    },
    Rgba {
        height: u32,
        width: u32,
        r: Vec<u8>,
        g: Vec<u8>,
        b: Vec<u8>,
        a: Vec<u8>,
    },
}

impl ImageFormat {
    pub fn to_img(self) -> Im {
        match self {
            ImageFormat::Luma { height, width, l } => {
                Im::ImageLuma8(GrayImage::from_vec(*width, *height, l).unwrap())
            }
            ImageFormat::LumaA {
                height,
                width,
                l,
                a,
            } => Im::ImageLumaA8(
                GrayAlphaImage::from_vec(
                    *width,
                    *height,
                    l.into_iter().zip(a.into_iter()).flatten().collect(),
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
                    r.into_iter()
                        .zip(g.into_iter())
                        .zip(b.into_iter())
                        .flatten()
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
                    r.into_iter()
                        .zip(g.into_iter())
                        .zip(b.into_iter())
                        .zip(a.into_iter())
                        .flatten()
                        .collect(),
                )
                .unwrap(),
            ),
        }
    }
}
