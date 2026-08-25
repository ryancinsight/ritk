use anyhow::{bail, Result};
use ritk_spatial::{Direction, Point, Spacing};

use super::iter::Tiles;
use super::iter::VoxelIter;
use super::rows::RegionRows;

/// A borrowed, possibly strided, rectangular sub-region of an image's voxels.
///
/// Holds the region's logical `shape`, the `strides` that walk it inside the
/// parent's backing buffer, the physical `offset` of its first voxel, and the
/// physical metadata *of the region itself* — the origin is already shifted to
/// the region's first voxel, so a region is a self-describing image view rather
/// than a bare array window.
///
/// Every field is a fixed-size array, so narrowing a region allocates nothing.
///
/// # Axis order
///
/// `shape` and `strides` are row-major, matching the backing tensor: index `0`
/// is the outermost (slowest) axis. Physical metadata (`origin`, `spacing`,
/// `direction`) is axis-major — innermost-first — matching [`ritk_spatial`].
/// Shape axis `d` therefore corresponds to spatial axis `D - 1 - d`.
#[derive(Debug)]
pub struct VoxelRegion<'a, T, const D: usize> {
    /// The parent's whole contiguous host buffer. Indexed via `offset`+strides;
    /// never sliced down, so a region keeps O(1) construction at any depth.
    pub(crate) data: &'a [T],
    pub(crate) shape: [usize; D],
    pub(crate) strides: [usize; D],
    pub(crate) offset: usize,
    pub(crate) origin: Point<D>,
    pub(crate) spacing: Spacing<D>,
    pub(crate) direction: Direction<D>,
}

impl<T, const D: usize> Clone for VoxelRegion<'_, T, D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const D: usize> Copy for VoxelRegion<'_, T, D> {}

impl<'a, T, const D: usize> VoxelRegion<'a, T, D> {
    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    #[inline]
    #[must_use]
    pub fn strides(&self) -> [usize; D] {
        self.strides
    }

    #[inline]
    #[must_use]
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    #[inline]
    #[must_use]
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    #[inline]
    #[must_use]
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for axis in (0..D).rev() {
            if self.shape[axis] > 1 && self.strides[axis] != expected {
                return false;
            }
            expected *= self.shape[axis];
        }
        true
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.is_contiguous() {
            Some(&self.data[self.offset..self.offset + self.len()])
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn physical_index(&self, index: [usize; D]) -> usize {
        let mut flat = self.offset;
        for (&component, &stride) in index.iter().zip(self.strides.iter()) {
            flat += component * stride;
        }
        flat
    }

    #[inline]
    #[must_use]
    pub fn get(&self, index: [usize; D]) -> Option<&'a T> {
        if index
            .iter()
            .zip(self.shape.iter())
            .any(|(&component, &extent)| component >= extent)
        {
            return None;
        }
        self.data.get(self.physical_index(index))
    }

    pub fn subregion(&self, bounds: [(usize, usize); D]) -> Result<Self> {
        let mut shape = [0usize; D];
        let mut offset = self.offset;
        let mut start_spatial = [0f64; D];

        for axis in 0..D {
            let (start, end) = bounds[axis];
            if start > end || end > self.shape[axis] {
                bail!(
                    "region bounds [{start}..{end}) exceed extent {} on axis {axis}",
                    self.shape[axis]
                );
            }
            shape[axis] = end - start;
            offset += start * self.strides[axis];
            start_spatial[D - 1 - axis] = start as f64;
        }

        Ok(Self {
            data: self.data,
            shape,
            strides: self.strides,
            offset,
            origin: shift_origin(&self.origin, &self.spacing, &self.direction, start_spatial),
            spacing: self.spacing,
            direction: self.direction,
        })
    }

    pub fn clipped_window(&self, centre: [usize; D], radius: [usize; D]) -> Result<Self> {
        let mut bounds = [(0usize, 0usize); D];
        for axis in 0..D {
            if centre[axis] >= self.shape[axis] {
                bail!(
                    "window centre {} exceeds extent {} on axis {axis}",
                    centre[axis],
                    self.shape[axis]
                );
            }
            let start = centre[axis].saturating_sub(radius[axis]);
            let end = (centre[axis] + radius[axis] + 1).min(self.shape[axis]);
            bounds[axis] = (start, end);
        }
        self.subregion(bounds)
    }

    /// Iterate the region's voxels as `(index, value)` pairs in row-major
    /// region order.
    ///
    /// Provided as an inherent method for discoverability; [`IntoIterator`]
    /// on `&VoxelRegion` forwards here so `for` loops and collection
    /// adapters work directly.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> VoxelIter<'a, T, D> {
        VoxelIter {
            region: *self,
            index: [0usize; D],
            remaining: if self.is_empty() { 0 } else { self.len() },
        }
    }

    #[must_use]
    pub fn rows(&self) -> RegionRows<'a, T, D> {
        let contiguous_rows = D == 0 || self.strides[D - 1] == 1;
        let row_len = if D == 0 { 1 } else { self.shape[D - 1] };
        RegionRows {
            region: *self,
            index: [0usize; D],
            remaining: if self.is_empty() {
                0
            } else {
                self.len() / row_len.max(1)
            },
            row_len,
            contiguous_rows,
            scratch: Vec::new(),
        }
    }

    pub fn subregions(&self, extent: [usize; D]) -> Result<Tiles<'a, T, D>> {
        let mut counts = [0usize; D];
        for axis in 0..D {
            if extent[axis] == 0 {
                bail!("tile extent must be non-zero on every axis, got 0 on axis {axis}");
            }
            counts[axis] = self.shape[axis] / extent[axis];
        }
        Ok(Tiles {
            region: *self,
            extent,
            counts,
            index: [0usize; D],
            remaining: counts.iter().product(),
        })
    }
}

pub(crate) fn shift_origin<const D: usize>(
    origin: &Point<D>,
    spacing: &Spacing<D>,
    direction: &Direction<D>,
    start: [f64; D],
) -> Point<D> {
    let spacing = spacing.to_array();
    Point::new(std::array::from_fn(|row| {
        origin[row]
            + (0..D)
                .map(|axis| direction[(row, axis)] * spacing[axis] * start[axis])
                .sum::<f64>()
    }))
}

#[inline]
pub(crate) fn advance<const D: usize>(index: &mut [usize; D], shape: &[usize; D]) {
    for axis in (0..D).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

impl<'a, T, const D: usize> IntoIterator for &'a VoxelRegion<'a, T, D> {
    type Item = <VoxelIter<'a, T, D> as Iterator>::Item;
    type IntoIter = VoxelIter<'a, T, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
