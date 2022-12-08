use crate::error::{Error, Result};
use image::imageops::{grayscale, FilterType};
use image::io::Reader;
use image::{DynamicImage, GenericImageView, GrayImage, ImageFormat, Rgba};
use std::fmt::Debug;
use std::path::Path;

pub fn resize_img_to_luma_layer(
    image: DynamicImage,
    mask: DynamicImage,
) -> Result<([GrayImage; 4], GrayImage)> {
    // compute output dimensions
    let image_dims = image.dimensions();
    let mask_dims = mask.dimensions();

    // resize mask to image size if needed
    let mask = if image_dims != mask_dims {
        mask.resize_exact(image.width(), image.height(), FilterType::Nearest)
    } else {
        mask
    };

    // get R, G, B, A
    let w = image.width();
    let h = image.height();
    let n = (w * h) as usize;
    let image = image.to_rgba8();
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut a = Vec::with_capacity(n);
    image.pixels().for_each(|Rgba(rgba)| {
        r.push(rgba[0]);
        g.push(rgba[1]);
        b.push(rgba[2]);
        a.push(rgba[3]);
    });
    let r = GrayImage::from_raw(w, h, r).unwrap();
    let g = GrayImage::from_raw(w, h, g).unwrap();
    let b = GrayImage::from_raw(w, h, b).unwrap();
    let a = GrayImage::from_raw(w, h, a).unwrap();
    let mask = grayscale(&mask);

    Ok(([r, g, b, a], mask))
}

pub fn resize_img_mask_to_luma(
    image: DynamicImage,
    mask: DynamicImage,
) -> Result<(GrayImage, GrayImage)> {
    // compute output dimensions
    let image_dims = image.dimensions();
    let mask_dims = mask.dimensions();

    // resize mask to image size if needed
    let mask = if image_dims != mask_dims {
        mask.resize_exact(image.width(), image.height(), FilterType::Nearest)
    } else {
        mask
    };

    // grayscale background
    let image = grayscale(&image);
    let mask = grayscale(&mask);

    Ok((image, mask))
}

/// support png and jpeg
pub fn read_img<P: AsRef<Path> + Debug>(image: P) -> Result<DynamicImage> {
    let image = Reader::open(&image)
        .map_err(|e| Error::ErrorMessage(format!("Fail to open {:?} ({:?})", image, e)))?
        .decode()
        .map_err(|e| Error::ErrorMessage(format!("Fail to decode {:?} ({:?})", image, e)))?;
    Ok(image)
}

pub fn write_png<P: AsRef<Path> + Debug>(out: P, img: &DynamicImage) -> Result<()> {
    img.save_with_format(&out, ImageFormat::Png).map_err(|e| {
        Error::ErrorMessage(format!("Failed to save image to out {:?} ({:?})", out, e))
    })?;
    Ok(())
}
