use crate::error::{Error, Result};
use image::imageops::{grayscale, FilterType};
use image::io::Reader;
use image::{DynamicImage, GenericImageView, GrayImage, ImageFormat};
use std::fmt::Debug;
use std::path::Path;

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
