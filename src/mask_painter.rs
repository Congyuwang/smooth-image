use crate::error::Result;
use crate::io::{read_img, resize_img_mask_to_luma, write_png};
use image::{DynamicImage, RgbImage};
use std::fmt::Debug;
use std::path::Path;

const RED: [u8; 3] = [255, 0, 0];

pub fn produce_gray_mask_image<I, M, O>(image: I, mask: M, output: O) -> Result<()>
where
    I: AsRef<Path> + Debug,
    M: AsRef<Path> + Debug,
    O: AsRef<Path> + Debug,
{
    let image = read_img(image)?;
    let mask = read_img(mask)?;
    let masked = gray_mask_image(image, mask)?;
    write_png(output, &DynamicImage::ImageRgb8(masked))?;
    Ok(())
}

fn gray_mask_image(image: DynamicImage, mask: DynamicImage) -> Result<RgbImage> {
    let (img, mask) = resize_img_mask_to_luma(image, mask)?;

    // compute output image
    let mask_raw = mask.as_raw();
    let img_raw = img.as_raw();
    let mut output = RgbImage::new(img.width(), img.height());
    mask_raw
        .iter()
        .zip(img_raw)
        .zip(output.chunks_exact_mut(3))
        .for_each(|((m, i), o)| o.copy_from_slice(&(if m == &0 { RED } else { [*i, *i, *i] })));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::mask_painter::gray_mask_image;
    use image::io::Reader;

    #[test]
    fn test_gray_image() {
        let image = Reader::open("./test/test_images/512_512_buildings.png")
            .unwrap()
            .decode()
            .unwrap();
        let mask = Reader::open("./test/test_masks/640_640_handwriting.png")
            .unwrap()
            .decode()
            .unwrap();
        gray_mask_image(image, mask).unwrap();
    }
}
