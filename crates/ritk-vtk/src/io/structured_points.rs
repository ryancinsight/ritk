//! Shared validation for VTK legacy structured-points geometry.

use anyhow::{Context, Result};

/// Validate dimensions and return their product in VTK XYZ order.
pub(crate) fn voxel_count([nx, ny, nz]: [usize; 3]) -> Result<usize> {
    anyhow::ensure!(
        nx > 0 && ny > 0 && nz > 0,
        "legacy VTK structured points dimensions must be positive, got [{nx}, {ny}, {nz}]"
    );
    nx.checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .with_context(|| format!("VTK DIMENSIONS product overflows usize: {nx}×{ny}×{nz}"))
}

/// Validate spacing in VTK XYZ order.
pub(crate) fn validate_spacing([sx, sy, sz]: [f64; 3]) -> Result<()> {
    for (axis, value) in [sx, sy, sz].into_iter().enumerate() {
        anyhow::ensure!(
            value.is_finite() && value > 0.0,
            "legacy VTK structured points requires finite, positive SPACING at axis {axis}; got {value}"
        );
    }
    Ok(())
}
