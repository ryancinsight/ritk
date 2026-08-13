//! Canonical Coeus-backed image contract.

use std::fmt;

use anyhow::{anyhow, bail};
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};

/// Medical image backed by a Coeus tensor.
///
/// The `D` const generic is the image dimensionality. Construction validates
/// that the tensor rank matches `D`, so index-space metadata cannot be paired
/// with a tensor of a different rank.
#[derive(Clone)]
pub struct Image<T, B, const D: usize>
where
    T: Scalar,
    B: ComputeBackend,
{
    // Crate-visible so the operation families in `crate::access` and
    // `crate::transform` can read them. Still crate-private: every external
    // path goes through the accessors, so the validating constructors stay
    // the only way to build an `Image`.
    pub(crate) data: Tensor<T, B>,
    pub(crate) origin: Point<D>,
    pub(crate) spacing: Spacing<D>,
    pub(crate) direction: Direction<D>,
    /// How index space maps into physical space. `Cartesian` for an ordinary
    /// raster; a non-Cartesian variant for beam-space acquisitions.
    pub(crate) map: CoordinateMap,
}

impl<T, B, const D: usize> fmt::Debug for Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("shape", &self.data.shape())
            .field("origin", &self.origin)
            .field("spacing", &self.spacing)
            .field("direction", &self.direction)
            .field("map", &self.map)
            .finish()
    }
}
impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Create an image from flat voxel data, shape, and physical metadata.
    ///
    /// This constructor validates the shape product before constructing the
    /// tensor, so malformed external buffers fail at the image boundary instead
    /// of relying on a downstream tensor panic.
    ///
    /// # Errors
    ///
    /// Returns an error when the checked product of `dims` overflows, or when
    /// `data.len()` does not equal that product.
    pub fn from_flat_on(
        data: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
        backend: &B,
    ) -> anyhow::Result<Self> {
        let expected = checked_numel(&dims)?;
        if data.len() != expected {
            bail!(
                "image flat data length {} does not match shape {:?} product {}",
                data.len(),
                dims,
                expected
            );
        }

        Self::new(
            Tensor::from_slice_on(dims, &data, backend),
            origin,
            spacing,
            direction,
        )
    }

    /// Create an image from Coeus tensor data and physical metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.ndim() != D`.
    pub fn new(
        data: Tensor<T, B>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> anyhow::Result<Self> {
        let rank = data.ndim();
        if rank != D {
            bail!("image tensor rank mismatch: expected {D}, got {rank}");
        }

        Ok(Self {
            data,
            origin,
            spacing,
            direction,
            map: CoordinateMap::Cartesian,
        })
    }

    /// Get the image data tensor.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &Tensor<T, B> {
        &self.data
    }

    /// Get the physical coordinate of the first pixel.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Get the physical distance between neighboring pixels.
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Get the direction cosine matrix for the image axes.
    #[inline]
    #[must_use]
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    /// Get the acquisition coordinate map.
    ///
    /// [`CoordinateMap::Cartesian`] for an ordinary raster, which is what every
    /// constructor produces; a beam-space variant only when set explicitly by
    /// [`Self::with_coordinate_map`].
    #[inline]
    #[must_use]
    pub fn coordinate_map(&self) -> &CoordinateMap {
        &self.map
    }

    /// Attach an acquisition coordinate map.
    ///
    /// # Errors
    ///
    /// Returns an error when the map is not meaningful at this image's
    /// dimensionality — see [`CoordinateMap::validate_dimensionality`].
    pub fn with_coordinate_map(mut self, map: CoordinateMap) -> anyhow::Result<Self> {
        map.validate_dimensionality(D)?;
        self.map = map;
        Ok(self)
    }

    /// Get the image shape as a fixed-rank array.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.data
            .shape()
            .try_into()
            .expect("invariant: Image::new validates tensor rank equals D")
    }

    /// Consume the image and return the underlying Coeus tensor.
    #[inline]
    #[must_use]
    pub fn into_tensor(self) -> Tensor<T, B> {
        self.data
    }

    /// Consume the image and return all components.
    ///
    /// Returns `(tensor, origin, spacing, direction, coordinate_map)`. The map
    /// is part of the image's identity: a caller that rebuilds an `Image` from
    /// these parts and drops it would silently reinterpret beam-space data as a
    /// Cartesian raster, so it is returned rather than left implicit.
    #[inline]
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Tensor<T, B>,
        Point<D>,
        Spacing<D>,
        Direction<D>,
        CoordinateMap,
    ) {
        (
            self.data,
            self.origin,
            self.spacing,
            self.direction,
            self.map,
        )
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    /// Create an image from flat voxel data on `B::default()`.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Image::from_flat_on`].
    #[inline]
    pub fn from_flat(
        data: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> anyhow::Result<Self> {
        Self::from_flat_on(data, dims, origin, spacing, direction, &B::default())
    }
}

fn checked_numel(dims: &[usize]) -> anyhow::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("image shape {:?} product overflows usize", dims))
    })
}

#[cfg(test)]
#[path = "tests_image_types.rs"]
mod tests;
