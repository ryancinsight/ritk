//! Voxel counts and window extents, checked against `usize` overflow.
//!
//! Every buffer-size check in the crate goes through [`check_buffer_lengths`],
//! so a `dims` whose voxel count overflows is an error on every entry point
//! instead of a panic in debug builds and a wrapped, wrong count in release.

use anyhow::{bail, Result};
use std::mem::size_of;

use crate::BlockMatchingError;

/// Voxel count of a `dims` grid.
///
/// # Errors
///
/// Returns an error naming `label` when the product overflows `usize`.
pub(crate) fn voxel_count(dims: [usize; 3], label: &'static str) -> Result<usize> {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .ok_or_else(|| BlockMatchingError::VoxelCountOverflow { label, dims }.into())
}

/// Buffer capacity for a three-dimensional grid, checked against the allocator limit.
///
/// # Errors
///
/// Returns an error when the voxel count or its byte capacity cannot be represented safely.
pub(crate) fn buffer_len<T>(dims: [usize; 3], label: &'static str) -> Result<usize> {
    buffer_len_with_limit::<T>(
        dims,
        label,
        usize::try_from(isize::MAX).expect("invariant: isize::MAX fits usize"),
    )
}

/// Capacity calculation with an explicit platform limit for cross-width tests.
pub(crate) fn buffer_len_with_limit<T>(
    dims: [usize; 3],
    label: &'static str,
    byte_limit: usize,
) -> Result<usize> {
    let count = voxel_count(dims, label)?;
    let element_size = size_of::<T>();
    count
        .checked_mul(element_size)
        .filter(|bytes| *bytes <= byte_limit)
        .ok_or(BlockMatchingError::ByteCountOverflow {
            label,
            dims,
            element_size,
        })?;
    Ok(count)
}

/// Per-axis extent `2 * radius + 1` of a window centred on a voxel.
///
/// # Errors
///
/// Returns an error naming `label` and the axis when an extent overflows.
pub(crate) fn window_extents(radius: [usize; 3], label: &'static str) -> Result<[usize; 3]> {
    let mut extents = [0; 3];
    for (axis, (extent, &axis_radius)) in extents.iter_mut().zip(&radius).enumerate() {
        *extent = axis_radius
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_add(1))
            .ok_or(BlockMatchingError::WindowExtentOverflow {
                label,
                axis,
                radius: axis_radius,
            })?;
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
        bail!("fixed ({fixed}) and moving ({moving}) buffers must both hold {expected} voxels for dims {dims:?}");
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::{
        buffer_len, buffer_len_with_limit, check_buffer_lengths, voxel_count, window_extents,
    };
    use crate::BlockMatchingError;

    fn assert_error<T: std::fmt::Debug>(result: anyhow::Result<T>, expected: BlockMatchingError) {
        let error = result.expect_err("operation must reject invalid input");
        assert_eq!(error.downcast_ref::<BlockMatchingError>(), Some(&expected));
    }

    #[test]
    fn voxel_count_multiplies_the_axes() {
        assert_eq!(voxel_count([2, 3, 4], "image").ok(), Some(24));
        assert_eq!(voxel_count([0, 3, 4], "image").ok(), Some(0));
    }

    #[test]
    fn voxel_count_rejects_an_overflowing_grid() {
        assert_error(
            voxel_count([usize::MAX, 2, 1], "image"),
            BlockMatchingError::VoxelCountOverflow {
                label: "image",
                dims: [usize::MAX, 2, 1],
            },
        );
    }

    #[test]
    fn buffer_len_rejects_an_allocation_that_exceeds_isize_bytes() {
        let boundary = usize::try_from(isize::MAX).expect("usize represents isize")
            / std::mem::size_of::<f64>();
        let accepted_dims = [1, 1, boundary];
        assert_eq!(
            buffer_len::<f64>(accepted_dims, "min/max pyramid level").ok(),
            Some(boundary)
        );
        let dims = [1, 1, boundary + 1];
        assert_error(
            buffer_len::<f64>(dims, "min/max pyramid level"),
            BlockMatchingError::ByteCountOverflow {
                label: "min/max pyramid level",
                dims,
                element_size: std::mem::size_of::<f64>(),
            },
        );
    }

    #[test]
    fn displacement_capacity_fails_when_32_bit_centres_still_fit() {
        let byte_limit = usize::try_from(isize::MAX).expect("isize::MAX fits usize");
        let dims = [1, 1, byte_limit / std::mem::size_of::<[f64; 3]>() + 1];
        assert_eq!(
            buffer_len_with_limit::<[u32; 3]>(dims, "tracking centres", byte_limit).ok(),
            Some(dims[2])
        );
        assert_error(
            buffer_len_with_limit::<[f64; 3]>(dims, "tracking displacements", byte_limit),
            BlockMatchingError::ByteCountOverflow {
                label: "tracking displacements",
                dims,
                element_size: std::mem::size_of::<[f64; 3]>(),
            },
        );
    }

    #[test]
    fn window_extents_are_twice_the_radius_plus_one() {
        assert_eq!(window_extents([0, 1, 5], "block").ok(), Some([1, 3, 11]));
    }

    #[test]
    fn window_extents_name_the_overflowing_axis() {
        for (radius, axis) in [
            ([1, usize::MAX / 2 + 1, 1], 1),
            ([1, 1, usize::MAX / 2 + 1], 2),
        ] {
            assert_error(
                window_extents(radius, "search"),
                BlockMatchingError::WindowExtentOverflow {
                    label: "search",
                    axis,
                    radius: radius[axis],
                },
            );
        }
    }

    #[test]
    fn buffer_lengths_must_match_the_grid() {
        assert_eq!(check_buffer_lengths(24, 24, [2, 3, 4]).ok(), Some(24));
        let error = check_buffer_lengths(24, 23, [2, 3, 4]).expect_err("moving buffer is short");
        assert_eq!(
            error.to_string(),
            "fixed (24) and moving (23) buffers must both hold 24 voxels for dims [2, 3, 4]"
        );
        assert_error(
            check_buffer_lengths(0, 0, [usize::MAX, 2, 1]),
            BlockMatchingError::VoxelCountOverflow {
                label: "image",
                dims: [usize::MAX, 2, 1],
            },
        );
    }
}
