use super::*;

use crate::freesurfer::SurfaceAnnotation;

/// A grid of 1 mm voxels spanning `0..8` on every axis.
fn grid() -> ParcellationGrid {
    ParcellationGrid::axis_aligned([8, 8, 8], [1.0; 3], [0.0; 3]).expect("valid grid")
}

/// An annotation over `labels.len()` vertices, with a matching table.
fn annotation(labels: &[u32]) -> SurfaceAnnotation {
    let mut table: Vec<(u32, String)> = labels
        .iter()
        .copied()
        .filter(|label| *label != BACKGROUND)
        .map(|label| (label, format!("parcel {label}")))
        .collect();
    table.sort_unstable();
    table.dedup();
    SurfaceAnnotation {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "fixtures hold a handful of vertices"
        )]
        vertex_count: labels.len() as u32,
        label_table: table,
        vertex_labels: labels.to_vec().into_boxed_slice(),
    }
}

fn surface(points: Vec<[f64; 3]>) -> Surface {
    Surface::new(points, Vec::new()).expect("valid surface")
}

// ── The ribbon is filled between the two surfaces ────────────────────────

/// A single column running from `z = 1` to `z = 4` must label every voxel it
/// crosses and nothing beyond either end.
///
/// This is the whole claim: the ribbon is the space *between* the surfaces, so
/// a voxel outside that span must stay background however close it sits.
#[test]
fn a_column_fills_the_span_between_the_surfaces_and_no_further() {
    let white = surface(vec![[3.0, 3.0, 1.0]]);
    let pial = surface(vec![[3.0, 3.0, 4.0]]);
    let grid = grid();

    let (parcellation, report) =
        rasterise_ribbon(&white, &pial, &annotation(&[5]), &grid, 16).expect("rasterises");

    for z in 0..8 {
        let expected = if (1..=4).contains(&z) { 5 } else { BACKGROUND };
        assert_eq!(
            parcellation.label_at_index([3, 3, z]),
            Some(expected),
            "voxel z={z}"
        );
    }
    assert_eq!(report.columns, 1);
    assert_eq!(report.unfilled_columns, 0);
    assert_eq!(report.filled_voxels, 4);
}

/// Two columns of different parcels fill their own spans without bleeding.
#[test]
fn separate_columns_fill_their_own_parcels() {
    let white = surface(vec![[1.0, 1.0, 1.0], [6.0, 6.0, 1.0]]);
    let pial = surface(vec![[1.0, 1.0, 3.0], [6.0, 6.0, 3.0]]);

    let (parcellation, report) =
        rasterise_ribbon(&white, &pial, &annotation(&[7, 9]), &grid(), 16).expect("rasterises");

    assert_eq!(parcellation.label_at_index([1, 1, 2]), Some(7));
    assert_eq!(parcellation.label_at_index([6, 6, 2]), Some(9));
    assert_eq!(parcellation.region_labels(), vec![7, 9]);
    assert_eq!(report.contested_voxels, 0);
}

/// A background vertex carries no parcel, so it is skipped rather than stamping
/// zeros over the volume.
#[test]
fn background_vertices_are_skipped() {
    let white = surface(vec![[3.0, 3.0, 1.0], [4.0, 4.0, 1.0]]);
    let pial = surface(vec![[3.0, 3.0, 3.0], [4.0, 4.0, 3.0]]);

    let (parcellation, report) =
        rasterise_ribbon(&white, &pial, &annotation(&[BACKGROUND, 4]), &grid(), 16)
            .expect("rasterises");

    assert_eq!(report.columns, 1, "only the labelled vertex is a column");
    assert_eq!(parcellation.label_at_index([3, 3, 2]), Some(BACKGROUND));
    assert_eq!(parcellation.label_at_index([4, 4, 2]), Some(4));
}

/// The region names come from the annotation's own table, so the parcellation
/// is readable without a separate lookup.
#[test]
fn region_names_carry_through_from_the_annotation() {
    let white = surface(vec![[2.0, 2.0, 1.0]]);
    let pial = surface(vec![[2.0, 2.0, 3.0]]);

    let (parcellation, _) =
        rasterise_ribbon(&white, &pial, &annotation(&[11]), &grid(), 8).expect("rasterises");

    assert_eq!(parcellation.name_of(11), Some("parcel 11"));
}

// ── The report says what the rasterisation could not do ──────────────────

/// Two parcels crossing one voxel is arbitrary at the voxel scale, so it is
/// counted rather than resolved silently.
#[test]
fn a_contested_voxel_is_counted_and_the_first_claim_stands() {
    // Both columns run through exactly the same voxels.
    let white = surface(vec![[3.0, 3.0, 1.0], [3.0, 3.0, 1.0]]);
    let pial = surface(vec![[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);

    let (parcellation, report) =
        rasterise_ribbon(&white, &pial, &annotation(&[2, 8]), &grid(), 8).expect("rasterises");

    assert!(
        report.contested_voxels > 0,
        "overlapping columns must be reported: {report:?}"
    );
    assert_eq!(
        parcellation.label_at_index([3, 3, 2]),
        Some(2),
        "the first column to claim a voxel keeps it"
    );
}

/// A column entirely outside the grid claims nothing, and the count of such
/// columns is what tells a caller the surfaces and the volume disagree.
#[test]
fn a_column_outside_the_grid_is_reported_as_unfilled() {
    let white = surface(vec![[3.0, 3.0, 1.0], [900.0, 900.0, 900.0]]);
    let pial = surface(vec![[3.0, 3.0, 3.0], [901.0, 901.0, 901.0]]);

    let (_, report) =
        rasterise_ribbon(&white, &pial, &annotation(&[1, 2]), &grid(), 8).expect("rasterises");

    assert_eq!(report.columns, 2);
    assert_eq!(report.unfilled_columns, 1);
}

/// Surfaces in the wrong frame land nowhere in the volume, which is the
/// commonest mistake — surface RAS used without the `c_ras` translation. It must
/// fail rather than return an empty parcellation.
#[test]
fn surfaces_entirely_outside_the_grid_are_rejected() {
    let white = surface(vec![[900.0, 900.0, 900.0]]);
    let pial = surface(vec![[901.0, 901.0, 901.0]]);

    let error = rasterise_ribbon(&white, &pial, &annotation(&[3]), &grid(), 8).unwrap_err();
    assert!(matches!(error, RibbonError::Parcellation(_)), "got {error}");
}

// ── Inputs that do not describe one reconstruction ───────────────────────

#[test]
fn mismatched_surfaces_are_rejected() {
    let white = surface(vec![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
    let pial = surface(vec![[1.0, 1.0, 3.0]]);

    let error = rasterise_ribbon(&white, &pial, &annotation(&[1, 2]), &grid(), 8).unwrap_err();
    assert!(matches!(error, RibbonError::Surface(_)), "got {error}");
}

#[test]
fn an_annotation_of_the_wrong_length_is_rejected() {
    let white = surface(vec![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
    let pial = surface(vec![[1.0, 1.0, 3.0], [2.0, 2.0, 4.0]]);

    let error = rasterise_ribbon(&white, &pial, &annotation(&[1]), &grid(), 8).unwrap_err();
    assert!(matches!(error, RibbonError::Surface(_)), "got {error}");
}

// ── Sampling ─────────────────────────────────────────────────────────────

/// A step count below two would sample nothing or only one end, so it is
/// raised to cover both endpoints rather than silently producing a hole.
#[test]
fn a_degenerate_step_count_still_walks_the_column() {
    let white = surface(vec![[3.0, 3.0, 1.0]]);
    let pial = surface(vec![[3.0, 3.0, 2.0]]);

    for steps in [0, 1, 2] {
        let (parcellation, _) =
            rasterise_ribbon(&white, &pial, &annotation(&[6]), &grid(), steps).expect("rasterises");
        assert_eq!(
            parcellation.label_at_index([3, 3, 1]),
            Some(6),
            "steps={steps} must reach the inner end"
        );
        assert_eq!(
            parcellation.label_at_index([3, 3, 2]),
            Some(6),
            "steps={steps} must reach the outer end"
        );
    }
}

/// Walking a long column more finely fills more of it — the setting has to do
/// something, or it is decoration.
#[test]
fn more_steps_fill_more_of_a_long_column() {
    let white = surface(vec![[3.0, 3.0, 0.0]]);
    let pial = surface(vec![[3.0, 3.0, 7.0]]);

    let (_, coarse) =
        rasterise_ribbon(&white, &pial, &annotation(&[1]), &grid(), 2).expect("rasterises");
    let (_, fine) =
        rasterise_ribbon(&white, &pial, &annotation(&[1]), &grid(), 64).expect("rasterises");

    assert_eq!(coarse.filled_voxels, 2, "two steps reach only the two ends");
    assert_eq!(fine.filled_voxels, 8, "a fine walk fills the whole column");
}
