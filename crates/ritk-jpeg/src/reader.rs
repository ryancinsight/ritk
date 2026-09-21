use anyhow::{Context, Result};
use coeus_core::ComputeBackend;
use consus_raster::PixelFormat;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

use crate::decode::decode_file;

const SRGB_LUMA: [u32; 3] = [2126, 7152, 722];
const SRGB_LUMA_DIVISOR: u32 = 10_000;

/// Read a JPEG file into a native 3-D grayscale image with shape `[1, height, width]`.
///
/// Grayscale samples are stored as `f32` values without normalization. RGB
/// input is converted to 8-bit sRGB luminance using the CIE Y coefficients
/// `[0.2126, 0.7152, 0.0722]`. Wide grayscale samples are scaled to Luma8 with
/// nearest-integer rounding. Encoded raster orientation is preserved.
///
/// # Errors
///
/// Returns an error if the file cannot be read or contains malformed,
/// unsupported, truncated, or over-limit JPEG data.
pub fn read_jpeg<B, P>(path: P, backend: &B) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let DecodedJpeg { data, dims } = decode_jpeg(path)?;
    Image::from_flat_on(
        data,
        dims,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        backend,
    )
}

struct DecodedJpeg {
    data: Vec<f32>,
    dims: [usize; 3],
}

fn decode_jpeg<P: AsRef<Path>>(path: P) -> Result<DecodedJpeg> {
    let path = path.as_ref();
    let image = decode_file(path)?;
    let width = usize::try_from(image.width()).context("JPEG width exceeds usize")?;
    let height = usize::try_from(image.height()).context("JPEG height exceeds usize")?;
    let format = image.format();
    let data = match format {
        PixelFormat::Gray => image.pixels().iter().copied().map(f32::from).collect(),
        PixelFormat::GrayWide => image
            .pixels()
            .chunks_exact(2)
            .map(|sample| {
                let wide = u16::from_ne_bytes([sample[0], sample[1]]);
                let luma = u8::try_from((u32::from(wide) + 128) / 257)
                    .expect("invariant: scaled u16 luminance fits in u8");
                f32::from(luma)
            })
            .collect(),
        PixelFormat::Rgb => image
            .pixels()
            .chunks_exact(3)
            .map(|rgb| {
                let luminance = SRGB_LUMA[0] * u32::from(rgb[0])
                    + SRGB_LUMA[1] * u32::from(rgb[1])
                    + SRGB_LUMA[2] * u32::from(rgb[2]);
                let sample = u8::try_from(luminance / SRGB_LUMA_DIVISOR)
                    .expect("invariant: weighted average of u8 channels fits in u8");
                f32::from(sample)
            })
            .collect(),
        _ => anyhow::bail!("JPEG decoder returned an unsupported pixel format"),
    };
    tracing::debug!(path = %path.display(), width, height, ?format, "read JPEG image");
    Ok(DecodedJpeg {
        data,
        dims: [1, height, width],
    })
}

/// Backend-bound native JPEG reader.
pub struct JpegReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> JpegReader<B> {
    /// Creates a reader that constructs images on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads a grayscale JPEG on the configured backend.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`read_jpeg`].
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<f32, B, 3>> {
        read_jpeg(path, &self.backend)
    }
}
