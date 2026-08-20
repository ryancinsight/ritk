use super::*;

use leto::geometry::Point3;
use ritk_parcellation::ParcellationGrid;

const TOLERANCE: f64 = 1.0e-12;

/// A 1-D strip of eight 1 mm voxels:
///
/// ```text
/// index:  0  1  2  3  4  5  6  7
/// label:  1  1  0  0  0  0  2  2
/// ```
///
/// The background gap in the middle is the white matter a streamline is tracked
/// through; the labelled ends are the grey-matter parcels it should be
/// attributed to.
fn strip() -> Parcellation {
    let grid =
        ParcellationGrid::axis_aligned([8, 1, 1], [1.0, 1.0, 1.0], [0.0; 3]).expect("valid grid");
    Parcellation::new(
        Box::new([1, 1, 0, 0, 0, 0, 2, 2]),
        grid,
        vec![(1, "Left".into()), (2, "Right".into())],
    )
    .expect("valid parcellation")
}

fn line(from: [f64; 3], to: [f64; 3]) -> Polyline<f64> {
    Polyline::new(vec![
        Point3::new(from[0], from[1], from[2]),
        Point3::new(to[0], to[1], to[2]),
    ])
    .expect("valid polyline")
}

/// A streamline running end to end across the strip's labelled voxels.
fn spanning() -> Polyline<f64> {
    line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0])
}

/// A streamline stopping one voxel short of each parcel — the ordinary case for
/// tracking that terminates at the grey/white boundary.
fn short_of_the_parcels() -> Polyline<f64> {
    line([2.0, 0.0, 0.0], [5.0, 0.0, 0.0])
}

fn build(streamlines: &[Polyline<f64>], config: &ConnectomeConfig) -> ConnectivityMatrix {
    build_connectivity_matrix(&strip(), streamlines, config).expect("build succeeds")
}

// ── Endpoint assignment ──────────────────────────────────────────────────

#[test]
fn terminal_assignment_connects_the_regions_the_endpoints_land_in() {
    let matrix = build(&[spanning()], &ConnectomeConfig::new());

    assert_eq!(matrix.weight(1, 2), Some(1.0));
    assert_eq!(matrix.accounting().assigned, 1);
    assert_eq!(matrix.accounting().unassigned, 0);
}

/// The defect radial search exists to fix: a streamline ending in white matter
/// is dropped entirely under terminal assignment, despite ending exactly where
/// tracking should stop.
#[test]
fn terminal_assignment_drops_a_streamline_ending_in_white_matter() {
    let matrix = build(&[short_of_the_parcels()], &ConnectomeConfig::new());

    assert_eq!(matrix.edge_count(), 0);
    assert_eq!(matrix.accounting().unassigned, 1);
    assert_eq!(matrix.accounting().assigned, 0);
}

/// The same streamline, recovered. Two voxels of reach is enough to bridge the
/// one-voxel gap at each end.
#[test]
fn radial_search_recovers_a_streamline_ending_in_white_matter() {
    let config = ConnectomeConfig::new()
        .with_assignment(EndpointAssignment::RadialSearch { radius_mm: 2.0 });
    let matrix = build(&[short_of_the_parcels()], &config);

    assert_eq!(matrix.weight(1, 2), Some(1.0));
    assert_eq!(matrix.accounting().assigned, 1);
    assert_eq!(matrix.accounting().unassigned, 0);
}

/// A radius too short to reach any label leaves the streamline unassigned, so
/// the recovery is bounded by the radius rather than unconditional.
#[test]
fn a_radius_too_short_to_reach_a_label_still_drops_the_streamline() {
    let config = ConnectomeConfig::new()
        .with_assignment(EndpointAssignment::RadialSearch { radius_mm: 0.5 });
    let matrix = build(&[short_of_the_parcels()], &config);

    assert_eq!(matrix.edge_count(), 0);
    assert_eq!(matrix.accounting().unassigned, 1);
}

/// Radial search must never move an endpoint that already sat inside a region,
/// so it can only add assignments a terminal search would have dropped.
#[test]
fn radial_search_leaves_endpoints_already_inside_a_region_alone() {
    let terminal = build(&[spanning()], &ConnectomeConfig::new());
    let radial = build(
        &[spanning()],
        &ConnectomeConfig::new()
            .with_assignment(EndpointAssignment::RadialSearch { radius_mm: 4.0 }),
    );

    assert_eq!(radial.weight(1, 2), terminal.weight(1, 2));
    assert_eq!(radial.accounting().assigned, terminal.accounting().assigned);
}

#[test]
fn an_unusable_radius_is_rejected() {
    let config = ConnectomeConfig::new()
        .with_assignment(EndpointAssignment::RadialSearch { radius_mm: -1.0 });
    let error = build_connectivity_matrix(&strip(), &[], &config).unwrap_err();
    assert!(matches!(error, ConnectomeError::Parcellation(_)));
}

// ── Accounting ───────────────────────────────────────────────────────────

/// A streamline that starts and ends in the same region is real and is counted,
/// but it is not an inter-region edge.
#[test]
fn an_intra_region_streamline_is_counted_separately() {
    let matrix = build(
        &[line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        &ConnectomeConfig::new(),
    );

    assert_eq!(matrix.accounting().intra_region, 1);
    assert_eq!(matrix.accounting().assigned, 0);
    assert_eq!(matrix.edge_count(), 0);
    // The self-connection is recorded even though it is not an edge.
    assert_eq!(matrix.weight(1, 1), Some(1.0));
}

/// An endpoint outside the volume cannot be attributed, and the streamline is
/// counted as unassigned rather than causing an error — a real tractogram always
/// contains some.
#[test]
fn an_endpoint_outside_the_volume_is_unassigned_rather_than_an_error() {
    let matrix = build(
        &[line([0.0, 0.0, 0.0], [500.0, 0.0, 0.0])],
        &ConnectomeConfig::new(),
    );

    assert_eq!(matrix.accounting().unassigned, 1);
    assert_eq!(matrix.edge_count(), 0);
}

/// Every streamline lands in exactly one accounting bucket, so the parts sum to
/// the whole. A miscount here would silently misrepresent how much of the
/// tractogram the connectome rests on.
#[test]
fn the_accounting_buckets_partition_the_tractogram() {
    let streamlines = vec![
        spanning(),
        spanning(),
        short_of_the_parcels(),
        line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        line([0.0, 0.0, 0.0], [500.0, 0.0, 0.0]),
    ];
    let matrix = build(&streamlines, &ConnectomeConfig::new());
    let accounting = matrix.accounting();

    assert_eq!(accounting.total, 5);
    assert_eq!(
        accounting.assigned + accounting.intra_region + accounting.unassigned,
        accounting.total
    );
    assert_eq!(accounting.assigned, 2);
    assert_eq!(accounting.intra_region, 1);
    assert_eq!(accounting.unassigned, 2);
    assert!((accounting.assigned_fraction() - 0.4).abs() < TOLERANCE);
}

#[test]
fn an_empty_tractogram_builds_an_empty_matrix() {
    let matrix = build(&[], &ConnectomeConfig::new());

    assert_eq!(matrix.region_count(), 2);
    assert_eq!(matrix.edge_count(), 0);
    assert_eq!(matrix.accounting().total, 0);
}

// ── Edge weighting ───────────────────────────────────────────────────────

#[test]
fn streamline_count_adds_one_per_streamline() {
    let matrix = build(
        &[spanning(), spanning(), spanning()],
        &ConnectomeConfig::new(),
    );
    assert_eq!(matrix.weight(1, 2), Some(3.0));
}

/// The length normalisation: a 7 mm streamline contributes `1/7`.
#[test]
fn inverse_length_weights_by_the_reciprocal_of_the_pathway() {
    let config = ConnectomeConfig::new().with_weighting(EdgeWeighting::InverseLength);
    let matrix = build(&[spanning()], &config);

    let weight = matrix.weight(1, 2).expect("edge present");
    assert!((weight - 1.0 / 7.0).abs() < TOLERANCE, "got {weight}");
}

/// The point of the normalisation: two streamlines of different length between
/// the same regions contribute unequally, with the longer one contributing less.
#[test]
fn inverse_length_makes_a_longer_pathway_contribute_less() {
    let config = ConnectomeConfig::new().with_weighting(EdgeWeighting::InverseLength);

    // The same endpoints, but one route detours through y and so is longer.
    let direct = line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0]);
    let detour = Polyline::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(3.5, 6.0, 0.0),
        Point3::new(7.0, 0.0, 0.0),
    ])
    .expect("valid polyline");

    let direct_weight = build(&[direct], &config).weight(1, 2).expect("edge");
    let detour_weight = build(&[detour], &config).weight(1, 2).expect("edge");
    assert!(
        detour_weight < direct_weight,
        "the longer pathway must contribute less: {detour_weight} vs {direct_weight}"
    );
}

/// Mean length reports the geometry of the pathway rather than a count, so it
/// does not grow when more streamlines take the same route.
#[test]
fn mean_length_reports_the_average_pathway_and_not_a_count() {
    let config = ConnectomeConfig::new().with_weighting(EdgeWeighting::MeanLength);

    let one = build(&[spanning()], &config).weight(1, 2).expect("edge");
    let three = build(&[spanning(), spanning(), spanning()], &config)
        .weight(1, 2)
        .expect("edge");

    assert!((one - 7.0).abs() < TOLERANCE, "got {one}");
    assert!(
        (three - one).abs() < TOLERANCE,
        "repeating a route must not change its mean length: {three} vs {one}"
    );
}

/// The average is a genuine mean over the contributing streamlines, not a sum
/// wearing the wrong name.
#[test]
fn mean_length_averages_streamlines_of_different_length() {
    let config = ConnectomeConfig::new().with_weighting(EdgeWeighting::MeanLength);
    let short = line([1.0, 0.0, 0.0], [6.0, 0.0, 0.0]); // 5 mm
    let long = line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0]); // 7 mm

    let weight = build(&[short, long], &config).weight(1, 2).expect("edge");
    assert!((weight - 6.0).abs() < TOLERANCE, "got {weight}");
}

/// The region-size normalisation. Each parcel is two 1 mm³ voxels, so the summed
/// node volume is 4 mm³ and one streamline contributes `1/4`.
#[test]
fn inverse_node_volume_divides_by_the_summed_region_volumes() {
    let config = ConnectomeConfig::new().with_weighting(EdgeWeighting::InverseNodeVolume);
    let matrix = build(&[spanning()], &config);

    let weight = matrix.weight(1, 2).expect("edge present");
    assert!((weight - 0.25).abs() < TOLERANCE, "got {weight}");
}

/// The point of the normalisation: enlarging a region must reduce its edge
/// weights, since the extra weight a larger region attracts is geometric rather
/// than anatomical.
#[test]
fn inverse_node_volume_reduces_the_weight_of_a_larger_region() {
    let config = ConnectomeConfig::new().with_weighting(EdgeWeighting::InverseNodeVolume);

    // The same strip, but region 1 spans four voxels instead of two.
    let grid = ParcellationGrid::axis_aligned([8, 1, 1], [1.0; 3], [0.0; 3]).expect("valid grid");
    let larger = Parcellation::new(Box::new([1, 1, 1, 1, 0, 0, 2, 2]), grid, Vec::new())
        .expect("valid parcellation");

    let baseline = build(&[spanning()], &config).weight(1, 2).expect("edge");
    let enlarged = build_connectivity_matrix(&larger, &[spanning()], &config)
        .expect("build")
        .weight(1, 2)
        .expect("edge");

    assert!(
        enlarged < baseline,
        "a larger region must carry less weight per streamline: {enlarged} vs {baseline}"
    );
}

/// The matrix records which weighting produced it, so a consumer cannot mistake
/// a length matrix for a count matrix.
#[test]
fn the_matrix_records_its_weighting() {
    for weighting in [
        EdgeWeighting::StreamlineCount,
        EdgeWeighting::InverseLength,
        EdgeWeighting::InverseNodeVolume,
        EdgeWeighting::MeanLength,
    ] {
        let matrix = build(
            &[spanning()],
            &ConnectomeConfig::new().with_weighting(weighting),
        );
        assert_eq!(matrix.weighting(), weighting);
    }
}

// ── Symmetry ─────────────────────────────────────────────────────────────

/// The graph is undirected, so the direction a streamline was tracked in must
/// not change the matrix.
#[test]
fn reversing_a_streamline_does_not_change_the_matrix() {
    let forward = build(
        &[line([0.0, 0.0, 0.0], [7.0, 0.0, 0.0])],
        &ConnectomeConfig::new(),
    );
    let backward = build(
        &[line([7.0, 0.0, 0.0], [0.0, 0.0, 0.0])],
        &ConnectomeConfig::new(),
    );

    assert_eq!(forward.weight(1, 2), backward.weight(1, 2));
    assert_eq!(forward.weight(2, 1), backward.weight(2, 1));
    assert_eq!(forward.weight(1, 2), forward.weight(2, 1));
}
