//! NRRD axis layout: which axis carries the acquisition, and where it sits.
//!
//! NRRD does not fix the position of a non-spatial axis the way NIfTI does.
//! A diffusion-weighted NRRD may place the gradient axis first, as the
//! NA-MIC convention Slicer and DTIPrep emit:
//!
//! ```text
//! dimension: 4
//! sizes: 33 128 128 60
//! kinds: list domain domain domain
//! space directions: none (1.7,0,0) (0,1.7,0) (0,0,2.2)
//! ```
//!
//! or last, which other tools emit. The two differ in memory stride, not in
//! meaning: a first axis varies fastest, so volumes are interleaved
//! voxel-by-voxel, while a last axis varies slowest, so volumes are contiguous
//! blocks. Reading either correctly requires identifying the axis before
//! touching the payload.
//!
//! Two header fields declare it. `kinds` labels each axis, and `space
//! directions` carries the bare token `none` in the non-spatial slot. `kinds`
//! is authoritative here because it is the field whose purpose is exactly this
//! statement; `space directions` is the fallback for files that omit `kinds`.

use anyhow::{bail, Result};

/// NRRD axis kinds that span physical space.
///
/// Every other kind — `list`, `vector`, `time`, the colour and tensor kinds —
/// indexes something other than position, so it cannot carry a spatial
/// direction or spacing.
const SPATIAL_KINDS: [&str; 2] = ["domain", "space"];

/// Where a non-spatial acquisition axis sits among a NRRD file's axes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcquisitionAxis {
    /// Every axis is spatial: an ordinary volume.
    Absent,
    /// Axis 0, varying fastest. Volumes are interleaved with stride equal to
    /// the volume count.
    Fastest,
    /// The final axis, varying slowest. Volumes are contiguous blocks.
    Slowest,
}

/// Locate the acquisition axis of a `dimension`-axis NRRD file.
///
/// `kinds` and `space_directions_present` are the raw header field values when
/// present. A rank-3 file is required to be entirely spatial; a rank-4 file is
/// required to have exactly one non-spatial axis, first or last, because those
/// are the layouts a shared spatial grid can be recovered from.
pub(crate) fn locate_acquisition_axis(
    dimension: usize,
    kinds: Option<&str>,
    space_direction_slots: Option<&[bool]>,
) -> Result<AcquisitionAxis> {
    let non_spatial = non_spatial_indices(dimension, kinds, space_direction_slots)?;

    match (dimension, non_spatial.as_slice()) {
        (_, []) if dimension <= 3 => Ok(AcquisitionAxis::Absent),
        (4, [index]) if *index == 0 => Ok(AcquisitionAxis::Fastest),
        (4, [index]) if *index == 3 => Ok(AcquisitionAxis::Slowest),
        (4, []) => bail!(
            "4-D NRRD declares four spatial axes; RITK images are 3-D, so one axis \
             must be a non-spatial acquisition axis declared by 'kinds' (list) or by \
             'none' in 'space directions'"
        ),
        (4, [index]) => bail!(
            "4-D NRRD places its non-spatial axis at index {index}; only the first \
             (fastest) or last (slowest) axis position is supported"
        ),
        (4, many) => bail!(
            "4-D NRRD declares {} non-spatial axes at {many:?}; exactly one \
             acquisition axis is supported",
            many.len()
        ),
        (_, many) => bail!(
            "{dimension}-D NRRD declares non-spatial axes at {many:?}; a \
             {dimension}-D file must be entirely spatial"
        ),
    }
}

/// Indices of the non-spatial axes, preferring `kinds` over `space directions`.
fn non_spatial_indices(
    dimension: usize,
    kinds: Option<&str>,
    space_direction_slots: Option<&[bool]>,
) -> Result<Vec<usize>> {
    if let Some(kinds) = kinds {
        let labels: Vec<&str> = kinds.split_whitespace().collect();
        if labels.len() != dimension {
            bail!(
                "NRRD 'kinds' lists {} axes but 'dimension' is {dimension}",
                labels.len()
            );
        }
        return Ok(labels
            .iter()
            .enumerate()
            .filter(|(_, label)| !SPATIAL_KINDS.contains(&label.to_lowercase().as_str()))
            .map(|(index, _)| index)
            .collect());
    }

    // `space directions` marks a non-spatial axis with the bare token `none`,
    // which the caller has already reduced to one flag per axis.
    if let Some(slots) = space_direction_slots {
        if slots.len() != dimension {
            bail!(
                "NRRD 'space directions' lists {} axes but 'dimension' is {dimension}",
                slots.len()
            );
        }
        return Ok(slots
            .iter()
            .enumerate()
            .filter(|(_, has_direction)| !**has_direction)
            .map(|(index, _)| index)
            .collect());
    }

    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_locates_a_leading_list_axis() {
        let axis =
            locate_acquisition_axis(4, Some("list domain domain domain"), None).expect("located");
        assert_eq!(axis, AcquisitionAxis::Fastest);
    }

    #[test]
    fn kinds_locates_a_trailing_list_axis() {
        let axis =
            locate_acquisition_axis(4, Some("domain domain domain list"), None).expect("located");
        assert_eq!(axis, AcquisitionAxis::Slowest);
    }

    #[test]
    fn space_directions_none_locates_the_axis_without_kinds() {
        let axis = locate_acquisition_axis(4, None, Some(&[false, true, true, true]))
            .expect("located from space directions");
        assert_eq!(axis, AcquisitionAxis::Fastest);
    }

    #[test]
    fn kinds_outranks_space_directions() {
        // A file may carry both; `kinds` is the field whose purpose is this
        // statement, so it decides.
        let axis = locate_acquisition_axis(
            4,
            Some("domain domain domain list"),
            Some(&[false, true, true, true]),
        )
        .expect("located");
        assert_eq!(
            axis,
            AcquisitionAxis::Slowest,
            "kinds must decide when both fields are present"
        );
    }

    #[test]
    fn rank_three_is_entirely_spatial() {
        let axis = locate_acquisition_axis(3, Some("domain domain domain"), None).expect("located");
        assert_eq!(axis, AcquisitionAxis::Absent);
    }

    #[test]
    fn rank_three_with_a_list_axis_is_rejected() {
        let err = locate_acquisition_axis(3, Some("list domain domain"), None)
            .expect_err("a 3-D file has no room for a non-spatial axis");
        assert!(
            format!("{err:#}").contains("entirely spatial"),
            "error must name the 3-D contract"
        );
    }

    #[test]
    fn interior_acquisition_axis_is_rejected() {
        let err = locate_acquisition_axis(4, Some("domain list domain domain"), None)
            .expect_err("an interior acquisition axis has no supported stride");
        let message = format!("{err:#}");
        assert!(
            message.contains("index 1"),
            "error must name the offending position, got: {message}"
        );
    }

    #[test]
    fn four_spatial_axes_are_rejected() {
        let err = locate_acquisition_axis(4, Some("domain domain domain domain"), None)
            .expect_err("four spatial axes cannot reduce to a 3-D grid");
        assert!(format!("{err:#}").contains("non-spatial acquisition axis"));
    }

    #[test]
    fn kinds_length_must_match_dimension() {
        let err = locate_acquisition_axis(4, Some("list domain domain"), None)
            .expect_err("kinds must cover every axis");
        assert!(format!("{err:#}").contains("lists 3 axes"));
    }

}
