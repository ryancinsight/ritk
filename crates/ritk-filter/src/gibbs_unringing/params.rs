//! Validated parameters: the total-variation window and the slice axis.

use super::GibbsError;

/// Neighbour offsets `K = [start, end]` over which the oscillation of a voxel
/// is measured (Kellner et al. 2016, Methods, the `TV±` definition).
///
/// For a voxel `x` and a shifted line `I`, the right-hand measure sums
/// `|I(x + t) − I(x + t + 1)|` and the left-hand measure
/// `|I(x − t) − I(x − t − 1)|` over `t ∈ [start, end]`. With `start ≥ 1` the
/// central voxel takes no part in its own measure, which the paper recommends
/// for stability at the edge itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TvWindow {
    start: u16,
    end: u16,
}

impl TvWindow {
    /// Build a window, rejecting `start > end`.
    ///
    /// # Errors
    ///
    /// [`GibbsError::InvalidWindow`] when `start > end`.
    pub fn new(start: u16, end: u16) -> Result<Self, GibbsError> {
        if start > end {
            return Err(GibbsError::InvalidWindow { start, end });
        }
        Ok(Self { start, end })
    }

    /// First neighbour offset.
    #[must_use]
    pub fn start(self) -> u16 {
        self.start
    }

    /// Last neighbour offset.
    #[must_use]
    pub fn end(self) -> u16 {
        self.end
    }

    /// Shortest line on which the left and right windows of a voxel, plus the
    /// voxel, cover distinct samples: `2 · (end + 1) + 1`.
    #[must_use]
    pub fn minimum_line(self) -> usize {
        2 * (usize::from(self.end) + 1) + 1
    }
}

impl Default for TvWindow {
    /// `K = [1, 3]`, the setting the paper selects for MRI data (Results,
    /// "Numerical Phantoms") and the `mrdegibbs` default.
    fn default() -> Self {
        Self { start: 1, end: 3 }
    }
}

/// The through-plane axis of the acquisition: slices are the 2-D planes
/// spanned by the two other axes of the `[usize; 3]` shape, and each slice is
/// corrected independently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SliceAxis {
    /// Slices stack along axis 0; axes 1 and 2 are in-plane. For a
    /// `[nz, ny, nx]` volume these are axial slices.
    #[default]
    Axis0,
    /// Slices stack along axis 1; axes 0 and 2 are in-plane.
    Axis1,
    /// Slices stack along axis 2; axes 0 and 1 are in-plane.
    Axis2,
}

impl SliceAxis {
    /// The through-plane axis index.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Axis0 => 0,
            Self::Axis1 => 1,
            Self::Axis2 => 2,
        }
    }

    /// The two in-plane axis indices, ascending.
    #[must_use]
    pub fn plane(self) -> [usize; 2] {
        match self {
            Self::Axis0 => [1, 2],
            Self::Axis1 => [0, 2],
            Self::Axis2 => [0, 1],
        }
    }
}
