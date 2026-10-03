use std::collections::TryReserveError;
use std::error::Error as StdError;
use std::path::PathBuf;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use image::{ImageBuffer, ImageFormat, Luma};
use ritk_image::Image;
use std::path::Path;

/// A failure while writing a grayscale PNG image.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PngWriteError {
    /// The image has more than one slice.
    #[error("PNG only supports one-slice images; got depth {depth}")]
    UnsupportedDepth {
        /// Number of slices in the image.
        depth: usize,
    },
    /// A row or column dimension is zero.
    #[error("PNG dimensions must be nonzero; got {height}x{width}")]
    EmptyDimensions {
        /// Number of rows in the image.
        height: usize,
        /// Number of columns in the image.
        width: usize,
    },
    /// The dimensions cannot be multiplied on this platform.
    #[error("PNG dimensions {height}x{width} overflow the host address space")]
    DimensionOverflow {
        /// Number of rows in the image.
        height: usize,
        /// Number of columns in the image.
        width: usize,
    },
    /// The image width cannot be represented in a PNG header.
    #[error("PNG width {width} exceeds u32")]
    WidthTooLarge {
        /// Number of columns in the image.
        width: usize,
    },
    /// The image height cannot be represented in a PNG header.
    #[error("PNG height {height} exceeds u32")]
    HeightTooLarge {
        /// Number of rows in the image.
        height: usize,
    },
    /// The voxel count differs from the dimensions.
    #[error("PNG voxel count {actual} does not match the required count {expected}")]
    VoxelCountMismatch {
        /// Number of values supplied by the image.
        actual: usize,
        /// Number of values required by its dimensions.
        expected: usize,
    },
    /// A voxel cannot be represented as an unsigned 16-bit PNG sample.
    #[error("PNG sample at flat index {index} is not a nonnegative integer through 65535: {value}")]
    InvalidSample {
        /// Zero-based index of the first unsupported value.
        index: usize,
        /// The unsupported value.
        value: f32,
    },
    /// Allocating the encoded sample buffer failed.
    #[error("PNG sample buffer allocation failed")]
    Allocation(#[source] TryReserveError),
    /// The encoder rejected a buffer whose checked dimensions and length differ.
    #[error("PNG pixel buffer length {actual} does not match the required count {expected}")]
    PixelBufferMismatch {
        /// Number of samples in the buffer.
        actual: usize,
        /// Number of samples required by the image dimensions.
        expected: usize,
    },
    /// Encoding failed. The source is erased so the image crate is not part of this API.
    #[error("PNG encoding failed for {path}")]
    Encoding {
        /// Destination path.
        path: PathBuf,
        /// Underlying encoding failure.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}
/// Writes one grayscale image as an 8-bit or 16-bit PNG.
///
/// PNG stores integer samples and no physical-space metadata. This writer
/// accepts only finite, nonnegative, integral samples through `u16::MAX`; it
/// selects 8-bit encoding when every value fits and 16-bit encoding otherwise.
/// The image must have depth one. It does not rescale or clamp voxel values.
///
/// # Errors
///
/// Returns an error when the image is not one slice, dimensions do not fit the
/// PNG fields, a sample is not exactly representable as an unsigned PNG value,
/// allocation fails, or the encoder cannot write the path.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use ritk_image::Image;
/// use ritk_spatial::{Direction, Point, Spacing};
///
/// # fn main() -> anyhow::Result<()> {
/// let backend = SequentialBackend;
/// let image = Image::from_flat_on(
///     vec![0.0_f32, 255.0],
///     [1, 1, 2],
///     Point::new([0.0; 3]),
///     Spacing::new([1.0; 3]),
///     Direction::identity(),
///     &backend,
/// )?;
/// let directory = tempfile::tempdir()?;
/// ritk_png::write_png(directory.path().join("slice.png"), &image, &backend)?;
/// # Ok(())
/// # }
/// ```
pub fn write_png<B, P>(
    path: P,
    image: &Image<f32, B, 3>,
    backend: &B,
) -> std::result::Result<(), PngWriteError>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let [depth, height, width] = image.shape();
    if depth != 1 {
        return Err(PngWriteError::UnsupportedDepth { depth });
    }
    if height == 0 || width == 0 {
        return Err(PngWriteError::EmptyDimensions { height, width });
    }

    let expected = height
        .checked_mul(width)
        .ok_or(PngWriteError::DimensionOverflow { height, width })?;
    let values = image.data_cow_on(backend);
    if values.len() != expected {
        return Err(PngWriteError::VoxelCountMismatch {
            actual: values.len(),
            expected,
        });
    }

    let depth = sample_depth(&values)?;
    let width =
        u32::try_from(width).map_err(|_| PngWriteError::WidthTooLarge { width })?;
    let height =
        u32::try_from(height).map_err(|_| PngWriteError::HeightTooLarge { height })?;
    let path = path.as_ref();

    match depth {
        SampleDepth::Byte => {
            let mut pixels = Vec::new();
            pixels
                .try_reserve_exact(values.len())
                .map_err(PngWriteError::Allocation)?;
            pixels.extend(values.iter().copied().map(byte_sample));
            let actual = pixels.len();
            let image = ImageBuffer::<Luma<u8>, _>::from_vec(width, height, pixels)
                .ok_or(PngWriteError::PixelBufferMismatch { actual, expected })?;
            image
                .save_with_format(path, ImageFormat::Png)
                .map_err(|source| PngWriteError::Encoding {
                    path: path.to_path_buf(),
                    source: Box::new(source),
                })
        }
        SampleDepth::Word => {
            let mut pixels = Vec::new();
            pixels
                .try_reserve_exact(values.len())
                .map_err(PngWriteError::Allocation)?;
            pixels.extend(values.iter().copied().map(word_sample));
            let actual = pixels.len();
            let image = ImageBuffer::<Luma<u16>, _>::from_vec(width, height, pixels)
                .ok_or(PngWriteError::PixelBufferMismatch { actual, expected })?;
            image
                .save_with_format(path, ImageFormat::Png)
                .map_err(|source| PngWriteError::Encoding {
                    path: path.to_path_buf(),
                    source: Box::new(source),
                })
        }
    }
}
#[derive(Clone, Copy)]
enum SampleDepth {
    Byte,
    Word,
}

fn sample_depth(values: &[f32]) -> std::result::Result<SampleDepth, PngWriteError> {
    let mut maximum = 0.0_f32;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite()
            || value.is_sign_negative()
            || value.fract() != 0.0
            || value > f32::from(u16::MAX)
        {
            return Err(PngWriteError::InvalidSample { index, value });
        }
        maximum = maximum.max(value);
    }

    Ok(if maximum <= f32::from(u8::MAX) {
        SampleDepth::Byte
    } else {
        SampleDepth::Word
    })
}
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "sample_depth proves this sample is an integral value in u8 range"
)]
fn byte_sample(value: f32) -> u8 {
    value as u8
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "sample_depth proves this sample is an integral value in u16 range"
)]
fn word_sample(value: f32) -> u16 {
    value as u16
}

#[cfg(test)]
mod tests {
    use super::{write_png, PngWriteError};
    use coeus_core::SequentialBackend;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, SequentialBackend, 3> {
        Image::from_flat_on(
            values,
            shape,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
            &SequentialBackend,
        )
        .expect("invariant: the test samples match their image shape")
    }

    #[test]
    fn round_trips_eight_and_sixteen_bit_grayscale() {
        let directory = tempdir().expect("temporary directory");
        for (name, samples) in [
            ("byte", vec![0.0, 9.0, 128.0, 255.0]),
            ("word", vec![0.0, 255.0, 256.0, 65_535.0]),
        ] {
            let source = image(samples.clone(), [1, 2, 2]);
            let path = directory.path().join(format!("{name}.png"));
            write_png(&path, &source, &SequentialBackend).expect("write PNG");
            let decoded =
                crate::read_png_to_image(&path, &SequentialBackend).expect("read written PNG");

            assert_eq!(decoded.shape(), source.shape());
            assert_eq!(decoded.data_slice().expect("contiguous"), samples);
            assert_eq!(decoded.origin().to_array(), [0.0; 3]);
            assert_eq!(decoded.spacing().to_array(), [1.0; 3]);
        }
    }

    #[test]
    fn rejects_unrepresentable_samples_and_volumes_without_output() {
        let directory = tempdir().expect("temporary directory");
        let cases = [
            ("negative", vec![-1.0], [1, 1, 1]),
            ("fractional", vec![1.5], [1, 1, 1]),
            ("overflow", vec![65_536.0], [1, 1, 1]),
            ("infinite", vec![f32::INFINITY], [1, 1, 1]),
            ("volume", vec![1.0, 2.0], [2, 1, 1]),
        ];

        for (name, values, shape) in cases {
            let path = directory.path().join(format!("{name}.png"));
            let input = image(values, shape);
            let error = write_png(&path, &input, &SequentialBackend)
                .expect_err("unsupported PNG input must fail");
            match name {
                "volume" => assert!(matches!(
                    error,
                    PngWriteError::UnsupportedDepth { depth: 2 }
                )),
                _ => assert!(matches!(
                    error,
                    PngWriteError::InvalidSample { index: 0, .. }
                )),
            }
            assert!(!path.exists(), "rejected PNG input must leave no file");
        }
    }
}
