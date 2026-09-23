//! Raster geometry for the sparse-field band: scalar-free and `Copy`, so it can
//! be passed into a per-voxel loop for nothing.

/// Face-neighbour offsets in ITK `m_NeighborList` order: in-plane first so the
/// 2-D case matches the validated 2-D ordering, `z` last (out of bounds and
/// skipped when the image is a single slice). The 2-D list is this list's first
/// four entries.
const FACE_OFFSETS: [(isize, isize, isize); 6] = [
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, 0),
    (-1, 0, 0),
    (1, 0, 0),
];

/// Raster geometry of a `[nz, ny, nx]` volume: flat-index arithmetic, neighbours,
/// and the face-offset table. Independent of the scalar a filter evolves.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GridTopology {
    nz: usize,
    ny: usize,
    nx: usize,
}

impl GridTopology {
    #[inline]
    pub(crate) fn new(dims: [usize; 3]) -> Self {
        let [nz, ny, nx] = dims;
        Self { nz, ny, nx }
    }

    /// Voxel count.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.nz * self.ny * self.nx
    }

    /// Effective dimensionality: a single slice is 2-D (ITK's rule).
    #[inline]
    pub(crate) fn ndim(&self) -> usize {
        if self.nz == 1 {
            2
        } else {
            3
        }
    }

    #[inline]
    pub(crate) fn nz(&self) -> usize {
        self.nz
    }

    #[inline]
    pub(crate) fn ny(&self) -> usize {
        self.ny
    }

    #[inline]
    pub(crate) fn nx(&self) -> usize {
        self.nx
    }

    #[inline]
    pub(crate) fn idx(&self, iz: usize, iy: usize, ix: usize) -> usize {
        iz * self.ny * self.nx + iy * self.nx + ix
    }

    #[inline]
    pub(crate) fn decode(&self, f: usize) -> (usize, usize, usize) {
        let iz = f / (self.ny * self.nx);
        let r = f % (self.ny * self.nx);
        (iz, r / self.nx, r % self.nx)
    }

    /// Flat index of the face neighbour, or `None` outside the volume.
    #[inline]
    pub(crate) fn neighbor(&self, f: usize, off: (isize, isize, isize)) -> Option<usize> {
        let (iz, iy, ix) = self.decode(f);
        let (z, y, x) = (
            iz as isize + off.0,
            iy as isize + off.1,
            ix as isize + off.2,
        );
        if z >= 0
            && y >= 0
            && x >= 0
            && z < self.nz as isize
            && y < self.ny as isize
            && x < self.nx as isize
        {
            Some(self.idx(z as usize, y as usize, x as usize))
        } else {
            None
        }
    }

    /// Flat index of integer coordinates, each axis clamped into the volume —
    /// the Neumann sample the difference and interpolation stencils need at the
    /// boundary.
    #[inline]
    pub(crate) fn clamped_index(&self, iz: isize, iy: isize, ix: isize) -> usize {
        let z = iz.clamp(0, self.nz as isize - 1) as usize;
        let y = iy.clamp(0, self.ny as isize - 1) as usize;
        let x = ix.clamp(0, self.nx as isize - 1) as usize;
        self.idx(z, y, x)
    }

    /// Face-neighbour offsets for this volume's dimensionality.
    #[inline]
    pub(crate) fn face_offsets(&self) -> &'static [(isize, isize, isize)] {
        if self.ndim() == 3 {
            &FACE_OFFSETS
        } else {
            &FACE_OFFSETS[..4]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_is_row_major_with_single_slice_as_2d() {
        let topo = GridTopology::new([1, 2, 3]);
        assert_eq!(topo.len(), 6);
        assert_eq!(topo.ndim(), 2);
        assert_eq!(topo.idx(0, 1, 2), 5);
        assert_eq!(topo.decode(5), (0, 1, 2));
        assert_eq!(topo.face_offsets().len(), 4);

        let volume = GridTopology::new([4, 5, 6]);
        assert_eq!(volume.ndim(), 3);
        assert_eq!(volume.len(), 120);
        assert_eq!(volume.face_offsets().len(), 6);
    }

    #[test]
    fn neighbors_are_bounds_checked_and_clamped_on_request() {
        let topo = GridTopology::new([2, 3, 4]);
        let interior = topo.idx(1, 1, 1);
        assert_eq!(topo.neighbor(interior, (0, 0, -1)), Some(topo.idx(1, 1, 0)));
        assert_eq!(topo.neighbor(interior, (-1, 0, 0)), Some(topo.idx(0, 1, 1)));
        let corner = topo.idx(0, 0, 0);
        // Every axis-negative offset leaves the volume there.
        assert_eq!(topo.neighbor(corner, (0, 0, -1)), None);
        assert_eq!(topo.neighbor(corner, (0, -1, 0)), None);
        assert_eq!(topo.neighbor(corner, (-1, 0, 0)), None);
        // Clamped sampling stays inside.
        assert_eq!(topo.clamped_index(-1, 0, 0), corner);
        assert_eq!(topo.clamped_index(1, 1, 1), topo.idx(1, 1, 1));
        assert_eq!(
            topo.clamped_index(0, 0, 1),
            topo.idx(0, 0, 1),
            "clamping must leave in-range coordinates alone"
        );
    }
}
