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

pub mod image;
pub mod iter;
pub mod rows;
pub mod voxel;

pub use iter::{Tiles, VoxelIter};
pub use rows::{RegionRows, RowWalker};
pub use voxel::VoxelRegion;

#[cfg(test)]
#[path = "../tests_region.rs"]
mod tests;
