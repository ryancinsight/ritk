//! Voxel counts and window extents, checked against `usize` overflow.
//!
//! Every buffer-size check in the crate goes through [`check_buffer_lengths`],
//! so a `dims` whose voxel count overflows is an error on every entry point
//! instead of a panic in debug builds and a wrapped, wrong count in release.

use anyhow::{anyhow, bail, Result};

/// Voxel count of a `dims` grid.
///
/// # Errors
///
/// Returns an error naming `label` when the product overflows `usize`.
pub(crate) fn voxel_count(dims: [usize; 3], label: &str) -> Result<usize> {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .ok_or_else(|| anyhow!("{label} dimensions {dims:?} overflow"))
}

/// Per-axis extent `2 * radius + 1` of a window centred on a voxel.
///
/// # Errors
///
/// Returns an error naming `label` and the axis when an extent overflows.
pub(crate) fn window_extents(radius: [usize; 3], label: &str) -> Result<[usize; 3]> {
    let mut extents = [0; 3];
    for (axis, (extent, &axis_radius)) in extents.iter_mut().zip(&radius).enumerate() {
        *extent = axis_radius
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_add(1))
            .ok_or_else(|| anyhow!("{label} extent overflows on axis {axis}"))?;
    }
    Ok(extents)
}

/// Check that the fixed and moving buffers each hold one sample per voxel of
/// `dims`, and return that voxel count.
///
/// # Errors
///
/// Returns an error when the voxel count overflows or either length differs.
pub(crate) fn check_buffer_lengths(fixed: usize, moving: usize, dims: [usize; 3]) -> Result<usize> {
    let expected = voxel_count(dims, "image")?;
    if fixed != expected || moving != expected {
        bail!(
            "fixed ({fixed}) and moving ({moving}) buffers must both hold {expected} voxels for dims {dims:?}"
        );
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::{check_buffer_lengths, voxel_count, window_extents};

    #[test]
    fn voxel_count_multiplies_the_axes() {
        assert_eq!(voxel_count([2, 3, 4], "image").ok(), Some(24));
        assert_eq!(voxel_count([0, 3, 4], "image").ok(), Some(0));
    }

    #[test]
    fn voxel_count_rejects_an_overflowing_grid() {
        let error = voxel_count([usize::MAX, 2, 1], "image").expect_err("the product overflows");
        assert_eq!(
            error.to_string(),
            format!("image dimensions {:?} overflow", [usize::MAX, 2, 1])
        );
    }

    #[test]
    fn window_extents_are_twice_the_radius_plus_one() {
        assert_eq!(window_extents([0, 1, 5], "block").ok(), Some([1, 3, 11]));
    }

    #[test]
    fn window_extents_name_the_overflowing_axis() {
        let error =
            window_extents([1, usize::MAX / 2 + 1, 1], "search").expect_err("axis 1 overflows");
        assert_eq!(error.to_string(), "search extent overflows on axis 1");
        let error =
            window_extents([1, 1, usize::MAX / 2 + 1], "search").expect_err("axis 2 overflows");
        assert_eq!(error.to_string(), "search extent overflows on axis 2");
    }

    #[test]
    fn buffer_lengths_must_match_the_grid() {
        assert_eq!(check_buffer_lengths(24, 24, [2, 3, 4]).ok(), Some(24));
        let error = check_buffer_lengths(24, 23, [2, 3, 4]).expect_err("moving is short");
        assert_eq!(
            error.to_string(),
            "fixed (24) and moving (23) buffers must both hold 24 voxels for dims [2, 3, 4]"
        );
        let error = check_buffer_lengths(0, 0, [usize::MAX, 2, 1]).expect_err("the grid overflows");
        assert_eq!(
            error.to_string(),
            format!("image dimensions {:?} overflow", [usize::MAX, 2, 1])
        );
    }
}
