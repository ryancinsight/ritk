use super::*;

use crate::ParcellationGrid;

const TOLERANCE: f64 = 1.0e-12;

/// A 4×4×1 slab with two rectangular regions and a background margin.
///
/// ```text
/// y=3:  0 0 0 0
/// y=2:  0 2 2 0
/// y=1:  1 1 0 0
/// y=0:  1 1 0 0
/// ```
///
/// Region 1 is a 2×2 block at the origin corner; region 2 is a 2×1 strip. Both
/// are small enough that every statistic is computable by hand, which is what
/// makes the assertions independent of the implementation.
fn two_region_slab() -> Parcellation {
    let labels: Box<[u32]> = Box::new([
        1, 1, 0, 0, // y=0
        1, 1, 0, 0, // y=1
        0, 2, 2, 0, // y=2
        0, 0, 0, 0, // y=3
    ]);
    let grid =
        ParcellationGrid::axis_aligned([4, 4, 1], [2.0, 2.0, 2.0], [0.0; 3]).expect("valid grid");
    Parcellation::new(labels, grid, vec![(1, "Block".into()), (2, "Strip".into())])
        .expect("valid parcellation")
}

fn statistics_for(parcellation: &Parcellation, label: u32) -> RegionStatistics {
    parcellation
        .region_statistics()
        .into_iter()
        .find(|statistics| statistics.label() == label)
        .expect("region present")
}

#[test]
fn every_region_is_reported_once_in_label_order() {
    let parcellation = two_region_slab();
    let statistics = parcellation.region_statistics();

    let labels: Vec<u32> = statistics.iter().map(RegionStatistics::label).collect();
    assert_eq!(labels, vec![1, 2]);
}

#[test]
fn voxel_counts_match_the_labelled_voxels() {
    let parcellation = two_region_slab();
    assert_eq!(statistics_for(&parcellation, 1).voxel_count(), 4);
    assert_eq!(statistics_for(&parcellation, 2).voxel_count(), 2);
}

/// Volume is the voxel count times the grid's voxel volume, so it must scale
/// with the spacing rather than counting voxels in disguise.
#[test]
fn volume_is_voxel_count_times_voxel_volume() {
    let parcellation = two_region_slab();
    // 2 mm isotropic voxels are 8 mm³ each.
    assert!((statistics_for(&parcellation, 1).volume() - 32.0).abs() < TOLERANCE);
    assert!((statistics_for(&parcellation, 2).volume() - 16.0).abs() < TOLERANCE);
}

/// The centroid of a 2×2 block at index corner `(0,0,0)` is index `(0.5, 0.5, 0)`,
/// which at 2 mm spacing from the origin is `(1, 1, 0)` mm.
#[test]
fn centroid_is_the_mean_of_the_voxel_centres_in_physical_space() {
    let parcellation = two_region_slab();

    let block = statistics_for(&parcellation, 1);
    let [x, y, z] = block.centroid().to_array();
    assert!((x - 1.0).abs() < TOLERANCE, "got {x}");
    assert!((y - 1.0).abs() < TOLERANCE, "got {y}");
    assert!(z.abs() < TOLERANCE, "got {z}");

    // The strip spans x indices 1..=2 at y index 2, so its centre is
    // index (1.5, 2, 0) → (3, 4, 0) mm.
    let strip = statistics_for(&parcellation, 2);
    let [x, y, _] = strip.centroid().to_array();
    assert!((x - 3.0).abs() < TOLERANCE, "got {x}");
    assert!((y - 4.0).abs() < TOLERANCE, "got {y}");
}

/// A centroid must be a physical position, so it must move with the grid's
/// origin. Reporting an index-space mean would leave it unchanged.
#[test]
fn centroid_follows_the_grid_origin() {
    let shifted_grid = ParcellationGrid::axis_aligned([4, 4, 1], [2.0; 3], [100.0, -50.0, 7.0])
        .expect("valid grid");
    let parcellation = Parcellation::new(
        two_region_slab().labels().to_vec().into_boxed_slice(),
        shifted_grid,
        Vec::new(),
    )
    .expect("valid parcellation");

    let [x, y, z] = statistics_for(&parcellation, 1).centroid().to_array();
    assert!((x - 101.0).abs() < TOLERANCE, "got {x}");
    assert!((y - -49.0).abs() < TOLERANCE, "got {y}");
    assert!((z - 7.0).abs() < TOLERANCE, "got {z}");
}

#[test]
fn bounding_box_spans_exactly_the_labelled_indices() {
    let parcellation = two_region_slab();

    let block = statistics_for(&parcellation, 1);
    assert_eq!(block.lower_index(), [0, 0, 0]);
    assert_eq!(block.upper_index(), [1, 1, 0]);
    assert_eq!(block.extent(), [2, 2, 1]);

    let strip = statistics_for(&parcellation, 2);
    assert_eq!(strip.lower_index(), [1, 2, 0]);
    assert_eq!(strip.upper_index(), [2, 2, 0]);
    assert_eq!(strip.extent(), [2, 1, 1]);
}

/// The single-region query must agree exactly with the whole-volume pass. They
/// are separate code paths, and a divergence between them would be invisible to
/// a test that only exercised one.
#[test]
fn single_region_query_agrees_with_the_whole_volume_pass() {
    let parcellation = two_region_slab();
    for label in parcellation.region_labels() {
        assert_eq!(
            parcellation.statistics_of(label),
            Some(statistics_for(&parcellation, label)),
            "label {label}"
        );
    }
}

#[test]
fn an_absent_label_has_no_statistics() {
    let parcellation = two_region_slab();
    assert_eq!(parcellation.statistics_of(99), None);
    assert_eq!(parcellation.statistics_of(crate::BACKGROUND), None);
}

/// A disconnected region is one region, and its centroid can land in the gap
/// between its parts. That is a property of centroids rather than a defect, and
/// pinning it keeps a later change from quietly redefining the measure.
#[test]
fn a_disconnected_region_reports_one_entry_with_a_centroid_between_its_parts() {
    let labels: Box<[u32]> = Box::new([5, 0, 0, 5]);
    let grid = ParcellationGrid::axis_aligned([4, 1, 1], [1.0; 3], [0.0; 3]).expect("valid grid");
    let parcellation = Parcellation::new(labels, grid, Vec::new()).expect("valid parcellation");

    let statistics = parcellation.region_statistics();
    assert_eq!(statistics.len(), 1);
    assert_eq!(statistics[0].voxel_count(), 2);
    // Centres at x = 0 and x = 3 average to 1.5, which is background.
    let [x, _, _] = statistics[0].centroid().to_array();
    assert!((x - 1.5).abs() < TOLERANCE, "got {x}");
    assert_eq!(
        parcellation.label_at_index([1, 0, 0]),
        Some(crate::BACKGROUND)
    );
}
