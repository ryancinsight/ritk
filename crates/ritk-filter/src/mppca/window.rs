//! Patch geometry of the MP-PCA sliding window.

use super::MpPcaError;

/// Extent of the MP-PCA window in voxels along each of the three axes.
///
/// The window holds `V = e₀·e₁·e₂` voxels. Veraart et al. (2016, §2.2) pick
/// `V` comparable to the volume count `D`, so both Casorati dimensions carry
/// the Marchenko-Pastur ratio `γ = min(V, D)/max(V, D)` near one;
/// [`PatchExtent::for_volume_count`] derives that default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchExtent([usize; 3]);

impl PatchExtent {
    /// Validate an explicit extent.
    ///
    /// # Errors
    ///
    /// [`MpPcaError::InvalidPatch`] when an axis is zero or the window holds
    /// fewer than two voxels (a one-voxel window has one eigenvalue, which the
    /// law cannot split).
    pub fn new(extent: [usize; 3]) -> Result<Self, MpPcaError> {
        let voxels = extent
            .iter()
            .try_fold(1_usize, |acc, &e| acc.checked_mul(e));
        match voxels {
            Some(v) if v >= 2 && extent.iter().all(|&e| e >= 1) => Ok(Self(extent)),
            _ => Err(MpPcaError::InvalidPatch { extent }),
        }
    }

    /// The smallest isotropic cube holding at least `volumes` voxels.
    ///
    /// Side `e = ⌈∛D⌉` (at least 2), so `V = e³ ≥ D` — the `M ≈ N` geometry of
    /// Veraart et al. (2016, §2.2), rounded up so the voxel dimension never
    /// becomes the smaller one by accident of truncation.
    #[must_use]
    pub fn for_volume_count(volumes: usize) -> Self {
        let mut side = 2_usize;
        while side.saturating_mul(side).saturating_mul(side) < volumes {
            side += 1;
        }
        Self([side; 3])
    }

    /// Extent along each axis.
    #[must_use]
    pub fn extent(self) -> [usize; 3] {
        self.0
    }

    /// Number of voxels in the window.
    #[must_use]
    pub fn voxels(self) -> usize {
        self.0.iter().product()
    }

    /// Check the window fits in `shape` on every axis.
    pub(crate) fn fit(self, shape: [usize; 3]) -> Result<(), MpPcaError> {
        if self.0.iter().zip(shape).all(|(&e, s)| e <= s) {
            Ok(())
        } else {
            Err(MpPcaError::PatchExceedsImage {
                extent: self.0,
                shape,
            })
        }
    }

    /// First voxel index of the window owned by `center`, per axis.
    ///
    /// The window is centred where it fits and shifted inward at borders, so
    /// it always spans exactly `extent` voxels (requires [`Self::fit`]).
    pub(crate) fn origin(self, center: [usize; 3], shape: [usize; 3]) -> [usize; 3] {
        std::array::from_fn(|axis| {
            let extent = self.0[axis];
            center[axis]
                .saturating_sub(extent / 2)
                .min(shape[axis] - extent)
        })
    }
}
