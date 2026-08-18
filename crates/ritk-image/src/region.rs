//! Borrowed, strided views into an [`Image`]'s voxels.
//!
//! [`crate::access`] answers "give me the whole buffer, and what does it cost".
//! This module answers the question that has no answer there: "give me *part* of
//! it, without paying for the rest". A [`VoxelRegion`] is a borrow — shape,
//! strides, offset and corrected physical metadata in fixed-size arrays, so
//! constructing one and narrowing it further allocates nothing at any rank.
//!
//! # Why a region is not a tensor slice
//!
//! `coeus_tensor::Tensor::slice` already produces a cheap strided view, but two
//! things stop it from being the answer here. It returns an owned `Tensor`
//! whose `Shape`/`Strides` are `SmallVec<[usize; 4]>` — inline only to rank 4 —
//! and, decisively, a strided `Tensor` cannot be *read*: `as_slice` and `iter`
//! both assert contiguity, so the only route from a tensor view to its values
//! is `to_contiguous`/`to_vec`, which copies the region. A region view is
//! therefore free to make and useless to read.
//!
//! The second half is domain, not layout: a sub-region of a medical image is
//! not just a sub-array. Its origin moves by `direction · (spacing ⊙ start)`,
//! so a view that forgets to shift it silently reports the parent's physical
//! position for every voxel. That correction is this crate's concern, which is
//! why the region type lives here rather than upstream.
//!
//! # The lending seam
//!
//! [`RowWalker`] carries `type Item<'a>` because its yielded rows have two
//! different owners. When a region's innermost axis is unit-stride — every
//! axis-aligned sub-region of an ordinary volume — a row is a direct borrow of
//! the source buffer, copying nothing. When it is not — a transposed or
//! permuted view — the row is gathered into a scratch buffer the walker owns
//! and reuses, so a full traversal costs one allocation rather than one per
//! row. An item that borrows the walker in one case and the source in the other
//! cannot be an [`Iterator`]: `Iterator::Item` has no lifetime to tie to
//! `&mut self`. Hence the generic associated type.
//!
//! Views whose items borrow only the source stay plain iterators
//! ([`VoxelRegion::iter`], [`VoxelRegion::subregions`]); a GAT buys them
//! nothing and would cost them the whole `Iterator` ecosystem.

use anyhow::{bail, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use ritk_spatial::{Direction, Point, Spacing};

use crate::types::Image;

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
    data: &'a [T],
    shape: [usize; D],
    strides: [usize; D],
    offset: usize,
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
}

// Hand-written rather than derived: `#[derive(Copy, Clone)]` would infer a
// `T: Copy` bound from the element type, but a region owns only a shared
// reference to the elements, so it is unconditionally copyable. The derive's
// structural default is wrong here, not merely conservative — it would make
// every region method requiring `Copy` unusable for a non-`Copy` element type.
impl<T, const D: usize> Clone for VoxelRegion<'_, T, D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const D: usize> Copy for VoxelRegion<'_, T, D> {}

impl<'a, T, const D: usize> VoxelRegion<'a, T, D> {
    /// The region's logical shape, outermost axis first.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// The strides that walk this region inside the parent buffer.
    #[inline]
    #[must_use]
    pub fn strides(&self) -> [usize; D] {
        self.strides
    }

    /// Physical position of the region's first voxel.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Physical distance between neighbouring voxels (inherited unchanged).
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Direction cosines of the image axes (inherited unchanged).
    #[inline]
    #[must_use]
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    /// Number of voxels in the region.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the region contains no voxels.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    /// Whether the region's voxels form one unbroken row-major run.
    ///
    /// True exactly when [`Self::as_slice`] can return the region's values
    /// directly. A region with any extent of zero or one along an outer axis
    /// can be contiguous even with non-row-major strides, because those axes
    /// contribute no stride steps.
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

    /// The region's values as one slice, when its layout permits.
    ///
    /// Returns `None` for a strided region rather than materialising it: a
    /// silent copy behind a borrow-shaped signature is exactly what this module
    /// exists to remove. Use [`Self::rows`] or [`Self::iter`] instead.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.is_contiguous() {
            Some(&self.data[self.offset..self.offset + self.len()])
        } else {
            None
        }
    }

    /// Physical buffer index of a logical region index, unchecked for bounds.
    #[inline]
    fn physical_index(&self, index: [usize; D]) -> usize {
        let mut flat = self.offset;
        for (&component, &stride) in index.iter().zip(self.strides.iter()) {
            flat += component * stride;
        }
        flat
    }

    /// Borrow one voxel by logical region index, or `None` when out of bounds.
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

    /// Narrow to a sub-region given per-axis half-open `[start, end)` bounds in
    /// row-major axis order.
    ///
    /// The returned region's origin is shifted to its own first voxel, so it
    /// reports correct physical positions independently of its parent.
    /// Allocates nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when any bound is inverted or exceeds the current
    /// extent along its axis.
    pub fn subregion(&self, bounds: [(usize, usize); D]) -> Result<Self> {
        let mut shape = [0usize; D];
        let mut offset = self.offset;
        // Row-major shape axis `d` is spatial axis `D - 1 - d`; the origin
        // shift is accumulated in spatial axis order below.
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

    /// Narrow to a window centred on `centre`, clipped to the region's bounds.
    ///
    /// The clipped-window (shrinking-boundary) convention ITK's box filters use:
    /// near an edge the window is truncated rather than padded, so the caller
    /// sees only real voxels and the sample count shrinks accordingly.
    /// Allocates nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when `centre` lies outside the region.
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

    /// Iterate the region's voxels in logical row-major order.
    ///
    /// Items borrow the source buffer, not the iterator, so this is a plain
    /// [`Iterator`] and composes with the whole adaptor ecosystem. Allocates
    /// nothing.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> VoxelIter<'a, T, D> {
        VoxelIter {
            region: *self,
            index: [0usize; D],
            remaining: if self.is_empty() { 0 } else { self.len() },
        }
    }

    /// A lending walker over the region's innermost rows.
    ///
    /// Yields one `&[T]` per innermost run: a direct borrow of the source when
    /// the innermost stride is `1`, and otherwise a gather into a scratch
    /// buffer the walker owns and reuses. See the module documentation for why
    /// this is a [`RowWalker`] rather than an [`Iterator`].
    ///
    /// Allocates once — and only when the innermost axis is strided.
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

    /// Iterate fixed-extent sub-regions tiling this region, dropping any ragged
    /// tail along each axis.
    ///
    /// Items borrow the source, so this is a plain [`Iterator`]. Allocates
    /// nothing — each yielded tile is a `VoxelRegion` of fixed-size arrays.
    ///
    /// # Errors
    ///
    /// Returns an error when any extent is zero.
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

/// Shift an origin to the voxel at `start` (spatial, innermost-first) index.
///
/// `origin' = origin + direction · (spacing ⊙ start)` — the single forward
/// affine this module needs. Kept private: the canonical public forward
/// transform is [`Image::continuous_index_to_physical_point`].
fn shift_origin<const D: usize>(
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

/// Row-major voxel iterator over a [`VoxelRegion`]. See [`VoxelRegion::iter`].
#[derive(Debug, Clone)]
pub struct VoxelIter<'a, T, const D: usize> {
    region: VoxelRegion<'a, T, D>,
    index: [usize; D],
    remaining: usize,
}

impl<'a, T, const D: usize> Iterator for VoxelIter<'a, T, D> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let value = &self.region.data[self.region.physical_index(self.index)];
        self.remaining -= 1;
        advance(&mut self.index, &self.region.shape);
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, const D: usize> ExactSizeIterator for VoxelIter<'_, T, D> {}
impl<T, const D: usize> std::iter::FusedIterator for VoxelIter<'_, T, D> {}

/// Fixed-extent tiling iterator. See [`VoxelRegion::subregions`].
#[derive(Debug, Clone)]
pub struct Tiles<'a, T, const D: usize> {
    region: VoxelRegion<'a, T, D>,
    extent: [usize; D],
    counts: [usize; D],
    index: [usize; D],
    remaining: usize,
}

impl<'a, T, const D: usize> Iterator for Tiles<'a, T, D> {
    type Item = VoxelRegion<'a, T, D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let mut bounds = [(0usize, 0usize); D];
        for ((slot, &tile), &extent) in bounds
            .iter_mut()
            .zip(self.index.iter())
            .zip(self.extent.iter())
        {
            let start = tile * extent;
            *slot = (start, start + extent);
        }
        self.remaining -= 1;
        advance(&mut self.index, &self.counts);
        // The bounds are derived from `counts`, which floor-divides the extent,
        // so every tile is in range.
        Some(
            self.region
                .subregion(bounds)
                .expect("invariant: tile bounds derive from floor-divided extents"),
        )
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, const D: usize> ExactSizeIterator for Tiles<'_, T, D> {}
impl<T, const D: usize> std::iter::FusedIterator for Tiles<'_, T, D> {}

/// Odometer step over a row-major logical index.
#[inline]
fn advance<const D: usize>(index: &mut [usize; D], shape: &[usize; D]) {
    for axis in (0..D).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

/// A walker whose yielded item may borrow the walker itself.
///
/// The lending seam. `Item<'a>` is tied to the `&'a mut self` borrow, so an
/// implementor is free to hand back either a borrow of an external buffer or a
/// borrow of its own reused scratch, and callers cannot hold two items at once
/// — which is precisely what makes scratch reuse sound. [`Iterator`] cannot
/// express this: its `Item` is a single type fixed independently of `next`'s
/// borrow.
///
/// Implemented by [`RegionRows`]; the documented next implementor is a
/// device-backed region reader, which must lend from a host staging buffer
/// because non-CPU-addressable storage has nothing to borrow directly.
pub trait RowWalker {
    /// The lent item, borrowing the walker for `'a`.
    type Item<'a>
    where
        Self: 'a;

    /// Advance and lend the next item, or `None` when exhausted.
    fn next_row(&mut self) -> Option<Self::Item<'_>>;
}

/// Lending walker over a region's innermost rows. See [`VoxelRegion::rows`].
#[derive(Debug, Clone)]
pub struct RegionRows<'a, T, const D: usize> {
    region: VoxelRegion<'a, T, D>,
    index: [usize; D],
    remaining: usize,
    row_len: usize,
    contiguous_rows: bool,
    /// Reused across the whole traversal; stays empty on the unit-stride path.
    scratch: Vec<T>,
}

impl<T, const D: usize> RegionRows<'_, T, D> {
    /// Rows not yet yielded.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    /// Whether rows are lent as direct borrows of the source buffer.
    ///
    /// `false` means each row is gathered into the walker's scratch buffer —
    /// still one allocation for the traversal, but a copy per row.
    #[inline]
    #[must_use]
    pub fn is_zero_copy(&self) -> bool {
        self.contiguous_rows
    }
}

impl<T: Copy, const D: usize> RowWalker for RegionRows<'_, T, D> {
    type Item<'b>
        = &'b [T]
    where
        Self: 'b;

    fn next_row(&mut self) -> Option<Self::Item<'_>> {
        if self.remaining == 0 {
            return None;
        }
        let start = self.region.physical_index(self.index);
        self.remaining -= 1;
        // Step the odometer over the outer axes only; the innermost axis is the
        // row this call yields.
        if D > 0 {
            let mut outer = self.index;
            let outer_shape = self.region.shape;
            for axis in (0..D - 1).rev() {
                outer[axis] += 1;
                if outer[axis] < outer_shape[axis] {
                    break;
                }
                outer[axis] = 0;
            }
            self.index = outer;
        }

        if self.contiguous_rows {
            return Some(&self.region.data[start..start + self.row_len]);
        }

        let stride = self.region.strides[D - 1];
        self.scratch.clear();
        self.scratch.reserve(self.row_len);
        self.scratch
            .extend((0..self.row_len).map(|step| self.region.data[start + step * stride]));
        Some(&self.scratch)
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Borrow the whole image as a [`VoxelRegion`].
    ///
    /// The zero-copy entry point to the region seam: no voxel data is read,
    /// copied, or allocated, and the returned region carries the image's own
    /// physical metadata.
    ///
    /// Unlike [`Self::data_slice`], this accepts a strided image. A permuted or
    /// sliced tensor has no flat host slice, so the only previous route to its
    /// values was [`Self::data_cow`], which materialises the whole volume; a
    /// region reads the same values in place through the layout's strides.
    /// [`VoxelRegion::as_slice`] still returns `None` for such a region, so the
    /// distinction stays visible where it matters.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing tensor's rank does not match `D`.
    /// Construction validates this, so a well-formed image never fails here.
    pub fn region(&self) -> Result<VoxelRegion<'_, T, D>> {
        let tensor = self.data();
        let layout = tensor.layout();
        let shape: [usize; D] = layout.shape().try_into().map_err(|_| {
            anyhow::anyhow!(
                "image tensor rank {} does not match D={D}",
                layout.shape().len()
            )
        })?;
        let strides: [usize; D] = layout
            .strides()
            .try_into()
            .map_err(|_| anyhow::anyhow!("image tensor strides do not match D={D}"))?;

        Ok(VoxelRegion {
            data: tensor.storage().as_slice(),
            shape,
            strides,
            offset: layout.offset(),
            origin: *self.origin(),
            spacing: *self.spacing(),
            direction: *self.direction(),
        })
    }
}

#[cfg(test)]
#[path = "tests_region.rs"]
mod tests;
