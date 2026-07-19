//! Deterministic MPR PNG export planning.
//!
//! This module is the SSOT for planning all per-axis slice exports for a loaded
//! `[depth, rows, cols]` volume. It is intentionally pure to support direct
//! value-semantic testing.

/// One planned PNG export entry for a specific axis/slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedSliceExport {
    /// Axis index: 0 = axial, 1 = coronal, 2 = sagittal.
    pub axis: usize,
    /// Slice index for the axis.
    pub slice_index: usize,
    /// Axis folder name used under the selected export root.
    pub axis_folder: &'static str,
    /// File name of the exported PNG.
    pub file_name: String,
}

/// Return the canonical folder label for an axis.
pub fn axis_folder_name(axis: usize) -> &'static str {
    match axis {
        0 => "axial",
        1 => "coronal",
        _ => "sagittal",
    }
}

/// Return the number of slices for an axis given volume shape `[d, r, c]`.
pub fn axis_slice_total(shape: [usize; 3], axis: usize) -> usize {
    match axis {
        0 => shape[0],
        1 => shape[1],
        _ => shape[2],
    }
}

/// Plan deterministic export entries for all axial/coronal/sagittal slices.
///
/// File name convention: `{axis_folder}_{slice_index:04}.png`.
pub fn plan_all_mpr_exports(shape: [usize; 3]) -> Vec<PlannedSliceExport> {
    let mut plan = Vec::new();
    for axis in 0..3 {
        let axis_folder = axis_folder_name(axis);
        let total = axis_slice_total(shape, axis);
        for slice_index in 0..total {
            plan.push(PlannedSliceExport {
                axis,
                slice_index,
                axis_folder,
                file_name: format!("{}_{:04}.png", axis_folder, slice_index),
            });
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_slice_total_matches_shape_dimensions() {
        let shape = [11, 13, 17];
        assert_eq!(axis_slice_total(shape, 0), 11);
        assert_eq!(axis_slice_total(shape, 1), 13);
        assert_eq!(axis_slice_total(shape, 2), 17);
    }

    #[test]
    fn axis_folder_names_are_stable() {
        assert_eq!(axis_folder_name(0), "axial");
        assert_eq!(axis_folder_name(1), "coronal");
        assert_eq!(axis_folder_name(2), "sagittal");
    }

    #[test]
    fn plan_count_equals_sum_of_axis_lengths() {
        let shape = [3, 4, 5];
        let plan = plan_all_mpr_exports(shape);
        assert_eq!(plan.len(), 3 + 4 + 5);
    }

    #[test]
    fn plan_starts_with_axial_and_ends_with_sagittal() {
        let shape = [2, 2, 2];
        let plan = plan_all_mpr_exports(shape);

        let first = plan.first().expect("first planned export");
        assert_eq!(first.axis, 0);
        assert_eq!(first.slice_index, 0);
        assert_eq!(first.axis_folder, "axial");
        assert_eq!(first.file_name, "axial_0000.png");

        let last = plan.last().expect("last planned export");
        assert_eq!(last.axis, 2);
        assert_eq!(last.slice_index, 1);
        assert_eq!(last.axis_folder, "sagittal");
        assert_eq!(last.file_name, "sagittal_0001.png");
    }
}
