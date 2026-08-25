use super::voxel::{advance, VoxelRegion};

/// Row-major voxel iterator over a [`VoxelRegion`]. See [`VoxelRegion::iter`].
#[derive(Debug, Clone)]
pub struct VoxelIter<'a, T, const D: usize> {
    pub(crate) region: VoxelRegion<'a, T, D>,
    pub(crate) index: [usize; D],
    pub(crate) remaining: usize,
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
    pub(crate) region: VoxelRegion<'a, T, D>,
    pub(crate) extent: [usize; D],
    pub(crate) counts: [usize; D],
    pub(crate) index: [usize; D],
    pub(crate) remaining: usize,
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
