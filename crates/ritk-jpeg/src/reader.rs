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
/// Samples are mapped from their encoded precision to the full 8-bit display
/// range with nearest-integer rounding. RGB input is then converted to sRGB
/// luminance using the CIE Y coefficients `[0.2126, 0.7152, 0.0722]`. Encoded
/// raster orientation is preserved.
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
    let pixel_count = width
        .checked_mul(height)
        .context("JPEG pixel count overflow")?;
    let format = image.format();
    let data = match format {
        PixelFormat::Gray | PixelFormat::GrayWide => {
            let samples = image.display_samples();
            validate_sample_count(samples.len(), pixel_count, format)?;
            samples.map(f32::from).collect()
        }
        PixelFormat::Rgb | PixelFormat::RgbWide => {
            rgb_luminance_samples(image.display_samples(), pixel_count, format)?
        }
        _ => anyhow::bail!("JPEG decoder returned an unsupported pixel format"),
    };
    tracing::debug!(path = %path.display(), width, height, ?format, "read JPEG image");
    Ok(DecodedJpeg {
        data,
        dims: [1, height, width],
    })
}

fn validate_sample_count(actual: usize, expected: usize, format: PixelFormat) -> Result<()> {
    if actual != expected {
        anyhow::bail!(
            "JPEG decoder returned {} display samples; expected {} for {:?}",
            actual,
            expected,
            format
        );
    }
    Ok(())
}

fn rgb_luminance_samples(
    mut samples: impl ExactSizeIterator<Item = u8>,
    pixel_count: usize,
    format: PixelFormat,
) -> Result<Vec<f32>> {
    let expected = pixel_count
        .checked_mul(3)
        .context("JPEG RGB sample count overflow")?;
    validate_sample_count(samples.len(), expected, format)?;

    let mut output = Vec::with_capacity(pixel_count);
    for _ in 0..pixel_count {
        let rgb = [
            samples
                .next()
                .expect("invariant: validated RGB iterator contains a red sample"),
            samples
                .next()
                .expect("invariant: validated RGB iterator contains a green sample"),
            samples
                .next()
                .expect("invariant: validated RGB iterator contains a blue sample"),
        ];
        output.push(f32::from(luminance(rgb)));
    }
    Ok(output)
}

fn luminance(rgb: [u8; 3]) -> u8 {
    let luminance = SRGB_LUMA[0] * u32::from(rgb[0])
        + SRGB_LUMA[1] * u32::from(rgb[1])
        + SRGB_LUMA[2] * u32::from(rgb[2]);
    u8::try_from(luminance / SRGB_LUMA_DIVISOR)
        .expect("invariant: weighted average of u8 channels fits in u8")
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
