use super::voxel::VoxelRegion;

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
    type Item<'a>
    where
        Self: 'a;

    fn next_row(&mut self) -> Option<Self::Item<'_>>;
}

/// Lending walker over a region's innermost rows. See [`VoxelRegion::rows`].
#[derive(Debug, Clone)]
pub struct RegionRows<'a, T, const D: usize> {
    pub(crate) region: VoxelRegion<'a, T, D>,
    pub(crate) index: [usize; D],
    pub(crate) remaining: usize,
    pub(crate) row_len: usize,
    pub(crate) contiguous_rows: bool,
    pub(crate) scratch: Vec<T>,
}

impl<T, const D: usize> RegionRows<'_, T, D> {
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.remaining
    }

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
