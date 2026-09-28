use super::*;

impl DiffusionMaps {
    /// Assemble maps directly, for cases a fit cannot produce.
    ///
    /// A diffusion tensor is sign-invariant, so fitting can never yield two
    /// voxels holding `v` and `−v` for the same fibre — the solver picks one
    /// sign deterministically per tensor. That case still arises across
    /// neighbouring voxels with different tensors, and it is exactly what
    /// sign-invariant interpolation exists to survive, so the test that proves
    /// it has to build the state directly.
    pub(crate) fn from_parts(
        eigenvalues: Vec<[f64; 3]>,
        principal: Vec<[f64; 3]>,
        mask: Vec<bool>,
        frame: GradientFrame,
    ) -> Self {
        assert!(
            eigenvalues.len() == principal.len() && principal.len() == mask.len(),
            "invariant: every per-voxel field covers the same voxels"
        );
        Self {
            eigenvalues,
            principal,
            mask,
            frame,
        }
    }
}
