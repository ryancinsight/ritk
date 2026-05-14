//! Conditional Mutual Information and Interaction Information.
//!
//! Delegates to `ritk_core::statistics::information`:
//! - I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)   (conditional MI)
//! - II(X;Y;Z) = I(X;Y) − I(X;Y|Z)                     (interaction info, McGill 1954)

use anyhow::Result;
use ritk_core::statistics::information::{
    conditional_mutual_information as core_cmi, interaction_information as core_ii,
};

/// I(X;Y|Z) via `ritk_core::statistics::information::conditional_mutual_information`.
pub(super) fn cmi_slices(x: &[f32], y: &[f32], z: &[f32], num_bins: usize) -> Result<f64> {
    core_cmi(x, y, z, num_bins)
}

/// II(X;Y;Z) via `ritk_core::statistics::information::interaction_information`.
pub(super) fn ii_slices(x: &[f32], y: &[f32], z: &[f32], num_bins: usize) -> Result<f64> {
    core_ii(x, y, z, num_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmi_constant_z_equals_mi_slice() {
        // I(X;Y|const) = I(X;Y) (validated analytically in ritk-core; cross-check slice path).
        use ritk_core::statistics::information::mutual_information as core_mi;
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i / 8) % 8) as f32).collect();
        let z_const = vec![2.0_f32; 64];
        let cmi = cmi_slices(&x, &y, &z_const, 8).unwrap();
        let mi = core_mi(&x, &y, 8).unwrap();
        assert!(
            (cmi - mi).abs() < 1e-9,
            "CMI(X,Y|const)={cmi:.9} must equal MI(X,Y)={mi:.9}"
        );
    }

    #[test]
    fn ii_constant_z_is_zero_slice() {
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i / 8) % 8) as f32).collect();
        let z_const = vec![2.0_f32; 64];
        let ii = ii_slices(&x, &y, &z_const, 8).unwrap();
        assert!(ii.abs() < 1e-9, "II(X;Y;const)={ii:.10} must be 0");
    }
}
