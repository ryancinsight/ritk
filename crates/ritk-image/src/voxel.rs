//! Runtime-selected scalar types for format-independent voxel images.

use coeus_core::ComputeBackend;

use crate::Image;

/// An image whose stored scalar type is selected at runtime.
///
/// Each variant owns a complete [`Image`], keeping its voxel representation and
/// physical-coordinate metadata together. Wrapping an image does not alter its
/// values, geometry, or [`crate::CoordinateMap`]. Format decoding and numeric
/// conversion policies belong to the I/O layer; this type preserves the
/// representation selected by the format reader.
///
/// # Examples
///
/// ```
/// use ritk_image::tensor::SequentialBackend;
/// use ritk_image::{Image, VoxelImage};
/// use ritk_spatial::{Direction, Point, Spacing};
///
/// let backend = SequentialBackend;
/// let image = Image::from_flat_on(
///     vec![12_u16, 900],
///     [2],
///     Point::new([0.0]),
///     Spacing::new([1.0]),
///     Direction::identity(),
///     &backend,
/// )
/// .expect("valid image dimensions");
/// let image = VoxelImage::Unsigned16(image);
///
/// assert!(matches!(image, VoxelImage::Unsigned16(_)));
/// assert_eq!(image.shape(), [2]);
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum VoxelImage<B, const D: usize>
where
    B: ComputeBackend,
{
    /// An image with unsigned 8-bit integer samples.
    Unsigned8(Image<u8, B, D>),
    /// An image with signed 8-bit integer samples.
    Signed8(Image<i8, B, D>),
    /// An image with unsigned 16-bit integer samples.
    Unsigned16(Image<u16, B, D>),
    /// An image with signed 16-bit integer samples.
    Signed16(Image<i16, B, D>),
    /// An image with unsigned 32-bit integer samples.
    Unsigned32(Image<u32, B, D>),
    /// An image with signed 32-bit integer samples.
    Signed32(Image<i32, B, D>),
    /// An image with unsigned 64-bit integer samples.
    Unsigned64(Image<u64, B, D>),
    /// An image with signed 64-bit integer samples.
    Signed64(Image<i64, B, D>),
    /// An image with IEEE 754 binary32 samples.
    Float32(Image<f32, B, D>),
    /// An image with IEEE 754 binary64 samples.
    Float64(Image<f64, B, D>),
}

impl<B, const D: usize> VoxelImage<B, D>
where
    B: ComputeBackend,
{
    /// Return the image extent in each spatial dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_image::tensor::SequentialBackend;
    /// use ritk_image::{Image, VoxelImage};
    /// use ritk_spatial::{Direction, Point, Spacing};
    ///
    /// let backend = SequentialBackend;
    /// let image = Image::from_flat_on(
    ///     vec![12_u16, 900],
    ///     [2],
    ///     Point::new([0.0]),
    ///     Spacing::new([1.0]),
    ///     Direction::identity(),
    ///     &backend,
    /// )
    /// .expect("valid image dimensions");
    /// let image = VoxelImage::Unsigned16(image);
    ///
    /// assert_eq!(image.shape(), [2]);
    /// ```
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        match self {
            Self::Unsigned8(image) => image.shape(),
            Self::Signed8(image) => image.shape(),
            Self::Unsigned16(image) => image.shape(),
            Self::Signed16(image) => image.shape(),
            Self::Unsigned32(image) => image.shape(),
            Self::Signed32(image) => image.shape(),
            Self::Unsigned64(image) => image.shape(),
            Self::Signed64(image) => image.shape(),
            Self::Float32(image) => image.shape(),
            Self::Float64(image) => image.shape(),
        }
    }
}

#[cfg(test)]
#[path = "tests_voxel.rs"]
mod tests;
