//! Anatomical-plane classification for internal axis indices.
//!
//! This module is the SSOT for mapping internal volume axes `[0,1,2]`
//! (`[depth,row,col]`) to anatomical view labels `(Axial, Coronal, Sagittal)`.
//!
//! # Theorem (deterministic plane classification)
//!
//! Let `v_i in R^3` be the unit direction vector for internal axis `i`.
//! Define absolute anatomical component scores:
//! - `z(i) = |v_i.z|`
//! - `y(i) = |v_i.y|`
//!
//! Classification is:
//! 1. `axial = argmax_i z(i)`
//! 2. `coronal = argmax_{i != axial} y(i)`
//! 3. `sagittal =` the remaining axis.
//!
//! The algorithm always returns a permutation of `{0,1,2}` and is deterministic.
//! Tie-breaking is stable by first-seen order in the remaining index list.
//!
//! Proof sketch:
//! - Step 1 selects one axis from a finite set of size 3.
//! - Step 2 selects one axis from the two unselected axes.
//! - Step 3 returns the only remaining axis.
//!   Therefore all three outputs are distinct and exhaustive, so the result is a
//!   permutation of `{0,1,2}`.

use crate::LoadedVolume;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnatomicalPlane {
    Axial,
    Coronal,
    Sagittal }

impl AnatomicalPlane {
    pub fn label(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal" }
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-10 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

fn axis_vector_from_direction(direction: &[f64; 9], axis: usize) -> [f64; 3] {
    let col = axis.min(2);
    [direction[col], direction[3 + col], direction[6 + col]]
}

pub(crate) fn axis_order_from_vectors(vectors: [[f64; 3]; 3]) -> [usize; 3] {
    let mut remaining = vec![0usize, 1, 2];

    let pick_best = |indices: &[usize], component: usize| -> usize {
        let mut best_axis = indices[0];
        let mut best_score = f64::NEG_INFINITY;
        for &axis in indices {
            let score = vectors[axis][component].abs();
            if score > best_score {
                best_score = score;
                best_axis = axis;
            }
        }
        best_axis
    };

    let axial = pick_best(&remaining, 2);
    remaining.retain(|&a| a != axial);
    let coronal = pick_best(&remaining, 1);
    remaining.retain(|&a| a != coronal);
    let sagittal = remaining[0];

    [axial, coronal, sagittal]
}

fn axis_vectors_for_volume(volume: &LoadedVolume) -> [[f64; 3]; 3] {
    if let Some(meta) = volume.metadata.as_ref() {
        if let Some(iop) = meta
            .slices
            .first()
            .and_then(|s| s.image_orientation_patient)
        {
            let row = [iop[0], iop[1], iop[2]];
            let col = [iop[3], iop[4], iop[5]];
            let normal = normalize(cross(row, col));
            // Internal axis convention: axis0=depth(normal), axis1=row-index axis,
            // axis2=col-index axis.
            return [normal, col, row];
        }
    }

    [
        axis_vector_from_direction(&volume.direction, 0),
        axis_vector_from_direction(&volume.direction, 1),
        axis_vector_from_direction(&volume.direction, 2),
    ]
}

pub fn axis_for_plane_in_volume(volume: Option<&LoadedVolume>, plane: AnatomicalPlane) -> usize {
    let default_axis = match plane {
        AnatomicalPlane::Axial => 0,
        AnatomicalPlane::Coronal => 1,
        AnatomicalPlane::Sagittal => 2 };
    let Some(volume) = volume else {
        return default_axis;
    };
    let [axial, coronal, sagittal] = axis_order_from_vectors(axis_vectors_for_volume(volume));
    match plane {
        AnatomicalPlane::Axial => axial,
        AnatomicalPlane::Coronal => coronal,
        AnatomicalPlane::Sagittal => sagittal }
}

pub fn anatomical_label_for_axis(volume: Option<&LoadedVolume>, axis: usize) -> &'static str {
    let axial = axis_for_plane_in_volume(volume, AnatomicalPlane::Axial);
    let coronal = axis_for_plane_in_volume(volume, AnatomicalPlane::Coronal);
    if axis == axial {
        "Axial"
    } else if axis == coronal {
        "Coronal"
    } else {
        "Sagittal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_order_is_permutation_of_all_axes() {
        let v = [[0.2, 0.2, 0.9], [0.1, 0.8, 0.2], [0.9, 0.1, 0.1]];
        let order = axis_order_from_vectors(v);
        let mut sorted = [order[0], order[1], order[2]];
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2], "axis order must be a full permutation");
    }

    #[test]
    fn canonical_lps_internal_basis_maps_to_axial_coronal_sagittal() {
        // Internal basis: depth->S, row->P, col->L.
        let v = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let [axial, coronal, sagittal] = axis_order_from_vectors(v);
        assert_eq!(axial, 0, "depth axis must classify as axial");
        assert_eq!(coronal, 1, "row axis must classify as coronal");
        assert_eq!(sagittal, 2, "col axis must classify as sagittal");
    }

    #[test]
    fn classification_is_stable_under_axis_permutation() {
        // Permute so axis2 is most aligned with z, axis0 with y, axis1 remainder.
        let v = [[0.0, 0.95, 0.05], [0.95, 0.0, 0.05], [0.0, 0.05, 0.95]];
        let [axial, coronal, sagittal] = axis_order_from_vectors(v);
        assert_eq!(axial, 2);
        assert_eq!(coronal, 0);
        assert_eq!(sagittal, 1);
    }

    #[test]
    fn no_volume_uses_default_axis_mapping() {
        assert_eq!(
            axis_for_plane_in_volume(None, AnatomicalPlane::Axial),
            0,
            "default axial axis must be 0"
        );
        assert_eq!(
            axis_for_plane_in_volume(None, AnatomicalPlane::Coronal),
            1,
            "default coronal axis must be 1"
        );
        assert_eq!(
            axis_for_plane_in_volume(None, AnatomicalPlane::Sagittal),
            2,
            "default sagittal axis must be 2"
        );
    }
}
