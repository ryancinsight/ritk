//! Voxel index newtype for 3-D image grids.

/// Index of a voxel in a 3-D image grid: `[iz, iy, ix]`.
///
/// Semantically distinct from shape/dimensions — a `VoxelIndex` identifies a
/// single position, not a volume extent. Prevents accidental interchange
/// with `[usize; 3]` shape arrays.
///
/// # Representation
///
/// `#[repr(transparent)]` over `[usize; 3]` — no layout or ABI change
/// relative to the raw array. Construction performs no bounds checking;
/// validity against a specific image shape is the caller's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VoxelIndex(pub [usize; 3]);

impl VoxelIndex {
    /// Create a `VoxelIndex` from individual z, y, x components.
    #[inline]
    pub fn new(iz: usize, iy: usize, ix: usize) -> Self {
        Self([iz, iy, ix])
    }

    /// Return the z (depth) component.
    #[inline]
    pub fn z(&self) -> usize {
        self.0[0]
    }

    /// Return the y (height) component.
    #[inline]
    pub fn y(&self) -> usize {
        self.0[1]
    }

    /// Return the x (width) component.
    #[inline]
    pub fn x(&self) -> usize {
        self.0[2]
    }

    /// Return a reference to the underlying `[usize; 3]` array.
    #[inline]
    pub fn as_array(&self) -> &[usize; 3] {
        &self.0
    }
}

impl Default for VoxelIndex {
    #[inline]
    fn default() -> Self {
        Self([0, 0, 0])
    }
}

impl From<[usize; 3]> for VoxelIndex {
    #[inline]
    fn from(arr: [usize; 3]) -> Self {
        Self(arr)
    }
}

impl From<VoxelIndex> for [usize; 3] {
    #[inline]
    fn from(idx: VoxelIndex) -> Self {
        idx.0
    }
}

impl std::ops::Index<usize> for VoxelIndex {
    type Output = usize;

    #[inline]
    fn index(&self, axis: usize) -> &Self::Output {
        &self.0[axis]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matches_array_construction() {
        let a = VoxelIndex::new(1, 2, 3);
        let b = VoxelIndex::from([1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn accessors() {
        let idx = VoxelIndex::new(4, 5, 6);
        assert_eq!(idx.z(), 4);
        assert_eq!(idx.y(), 5);
        assert_eq!(idx.x(), 6);
    }

    #[test]
    fn index_trait() {
        let idx = VoxelIndex::new(7, 8, 9);
        assert_eq!(idx[0], 7);
        assert_eq!(idx[1], 8);
        assert_eq!(idx[2], 9);
    }

    #[test]
    fn default_is_origin() {
        assert_eq!(VoxelIndex::default(), VoxelIndex::new(0, 0, 0));
    }

    #[test]
    fn from_roundtrip() {
        let arr: [usize; 3] = [10, 20, 30];
        let idx = VoxelIndex::from(arr);
        let back: [usize; 3] = idx.into();
        assert_eq!(arr, back);
    }

    #[test]
    fn as_array_returns_reference() {
        let idx = VoxelIndex::new(1, 2, 3);
        assert_eq!(idx.as_array(), &[1, 2, 3]);
    }

    #[test]
    fn repr_transparent_layout() {
        let idx = VoxelIndex::new(1, 2, 3);
        let ptr_core = &idx.0 as *const [usize; 3];
        let ptr_newtype = &idx as *const VoxelIndex;
        assert_eq!(
            ptr_core as *const u8, ptr_newtype as *const u8,
            "repr(transparent) must share address with inner array"
        );
    }
}
