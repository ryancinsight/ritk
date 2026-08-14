#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;

/// A 2×2×2 parcellation with three regions in a diagonal pattern:
///
/// ```text
/// z=0:   [1, 0]     z=1:   [0, 2]
///        [0, 3]            [3, 0]
/// ```
fn three_region_2x2x2() -> Parcellation {
    let labels: Box<[u32]> = Box::new([
        // z=0, y=0: x=0,1
        1, 0, // z=0, y=1: x=0,1
        0, 3, // z=1, y=0: x=0,1
        0, 2, // z=1, y=1: x=0,1
        3, 0,
    ]);
    Parcellation::new(
        labels,
        [2, 2, 2],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        vec![
            (1, "Region A".into()),
            (2, "Region B".into()),
            (3, "Region C".into()),
        ],
    )
    .expect("valid parcellation")
}

fn polyline_from_points(points: &[[f64; 3]]) -> Polyline<f64> {
    let pts: Vec<leto::geometry::Point3<f64>> = points
        .iter()
        .map(|&[x, y, z]| leto::geometry::Point3::new(x, y, z))
        .collect();
    Polyline::new(pts).expect("valid polyline")
}

// ── Parcellation construction ────────────────────────────────────────────

#[test]
fn parcellation_rejects_empty_labels() {
    let err = Parcellation::new(
        Box::new([]),
        [2, 2, 2],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(err, ConnectomeError::RegionCountMismatch { .. }));
}

#[test]
fn parcellation_rejects_all_background() {
    let labels: Box<[u32]> = vec![0u32; 8].into_boxed_slice();
    let err =
        Parcellation::new(labels, [2, 2, 2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], vec![]).unwrap_err();
    assert!(matches!(err, ConnectomeError::EmptyParcellation(0)));
}

#[test]
fn parcellation_region_labels_are_sorted_and_deduplicated() {
    let p = three_region_2x2x2();
    assert_eq!(p.region_labels(), vec![1, 2, 3]);
    assert_eq!(p.region_count(), 3);
}

#[test]
fn label_at_voxel_centre() {
    let p = three_region_2x2x2();
    // Voxel (0,0,0) centre at [0,0,0] → label 1
    assert_eq!(p.label_at(&Point::new([0.0, 0.0, 0.0])), Some(1));
    // Voxel (1,1,0) centre at [2,2,0] → label 3
    assert_eq!(p.label_at(&Point::new([2.0, 2.0, 0.0])), Some(3));
    // Voxel (1,0,1) centre at [2,0,2] → label 2
    assert_eq!(p.label_at(&Point::new([2.0, 0.0, 2.0])), Some(2));
}

#[test]
fn label_at_outside_volume_returns_none() {
    let p = three_region_2x2x2();
    assert_eq!(p.label_at(&Point::new([-1.0, 0.0, 0.0])), None);
    assert_eq!(p.label_at(&Point::new([0.0, 0.0, 4.0])), None);
    assert_eq!(p.label_at(&Point::new([f64::NAN, 0.0, 0.0])), None);
}

// ── Connectivity matrix construction ─────────────────────────────────────

#[test]
fn single_streamline_connects_two_regions() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    // Streamline from region 1 (0,0,0) to region 2 (2,0,2).
    let sl = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]);
    let matrix = build_connectivity_matrix(&p, &[sl])?;

    assert_eq!(matrix.region_count(), 3);
    assert_eq!(matrix.total_streamlines(), 1);
    assert_eq!(matrix.skipped_count(), 0);
    assert_eq!(matrix.intra_region_count(), 0);
    assert_eq!(matrix.weight(1, 2), Some(1.0));
    assert_eq!(matrix.weight(2, 1), Some(1.0));
    assert_eq!(matrix.weight(1, 3), Some(0.0));
    assert_eq!(matrix.edge_count(), 1);
    Ok(())
}

#[test]
fn multiple_streamlines_accumulate_weight() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    let sl_a = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]); // 1→2
    let sl_b = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]); // 1→3
    let sl_c = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]); // 1→2
    let matrix = build_connectivity_matrix(&p, &[sl_a, sl_b, sl_c])?;

    assert_eq!(matrix.total_streamlines(), 3);
    assert_eq!(matrix.weight(1, 2), Some(2.0));
    assert_eq!(matrix.weight(1, 3), Some(1.0));
    assert_eq!(matrix.weight(2, 3), Some(0.0));
    assert_eq!(matrix.edge_count(), 2);
    Ok(())
}

#[test]
fn intra_region_streamline_counts_but_not_as_edge() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    // Both endpoints in region 1 → self-edge only.
    let sl = polyline_from_points(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    let matrix = build_connectivity_matrix(&p, &[sl])?;

    assert_eq!(matrix.total_streamlines(), 1);
    assert_eq!(matrix.intra_region_count(), 1);
    assert_eq!(matrix.skipped_count(), 0);
    // Self-weight is recorded but not an inter-region edge.
    assert_eq!(matrix.weight(1, 1), Some(1.0));
    assert_eq!(matrix.edge_count(), 0);
    Ok(())
}

#[test]
fn out_of_bounds_endpoint_is_skipped() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    // Endpoint outside the volume.
    let sl = polyline_from_points(&[[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]);
    let matrix = build_connectivity_matrix(&p, &[sl])?;

    assert_eq!(matrix.total_streamlines(), 1);
    assert_eq!(matrix.skipped_count(), 1);
    assert_eq!(matrix.edge_count(), 0);
    Ok(())
}

#[test]
fn background_endpoint_is_skipped() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    // Endpoint at (2,0,0) is background (label 0) in this parcellation.
    let sl = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let matrix = build_connectivity_matrix(&p, &[sl])?;

    assert_eq!(matrix.total_streamlines(), 1);
    assert_eq!(matrix.skipped_count(), 1);
    assert_eq!(matrix.edge_count(), 0);
    Ok(())
}

// ── Graph measures ───────────────────────────────────────────────────────

#[test]
fn degree_counts_distinct_neighbours() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    // Region 1 connects to 2 and 3.
    let sl_a = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]); // 1→2
    let sl_b = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]); // 1→3
    let matrix = build_connectivity_matrix(&p, &[sl_a, sl_b])?;

    assert_eq!(matrix.degree(1), Some(2)); // connected to 2 and 3
    assert_eq!(matrix.degree(2), Some(1)); // connected to 1 only
    assert_eq!(matrix.degree(3), Some(1)); // connected to 1 only
    assert_eq!(matrix.degree(99), None);
    Ok(())
}

#[test]
fn strength_sums_incident_weights() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    let sl_a = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]); // 1→2
    let sl_b = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]); // 1→2
    let sl_c = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]); // 1→3
    let matrix = build_connectivity_matrix(&p, &[sl_a, sl_b, sl_c])?;

    assert!((matrix.strength(1).unwrap() - 3.0).abs() < 1e-12); // 2+1
    assert!((matrix.strength(2).unwrap() - 2.0).abs() < 1e-12); // 2 from 1→2
    assert!((matrix.strength(3).unwrap() - 1.0).abs() < 1e-12); // 1 from 1→3
    Ok(())
}

#[test]
fn density_is_ratio_of_edges_to_possible() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    // Triangle: all edges present → density = 1.0.
    let sl_a = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]); // 1→2
    let sl_b = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]); // 1→3
    let sl_c = polyline_from_points(&[[2.0, 0.0, 2.0], [2.0, 2.0, 0.0]]); // 2→3
    let matrix = build_connectivity_matrix(&p, &[sl_a, sl_b, sl_c])?;

    assert_eq!(matrix.edge_count(), 3);
    // 3 nodes → max edges = 3*2/2 = 3 → density = 1.0
    assert!((matrix.density() - 1.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn density_is_zero_for_single_region() -> Result<(), ConnectomeError> {
    // Parcellation with only one region.
    let labels: Box<[u32]> = Box::new([1, 1, 1, 1, 1, 1, 1, 1]);
    let p = Parcellation::new(
        labels,
        [2, 2, 2],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        vec![(1, "Solo".into())],
    )
    .unwrap();
    let sl = polyline_from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let matrix = build_connectivity_matrix(&p, &[sl])?;
    assert_eq!(matrix.region_count(), 1);
    assert_eq!(matrix.density(), 0.0);
    Ok(())
}

// ── Serialisation ────────────────────────────────────────────────────────

#[test]
fn json_round_trip_preserves_weights_and_measures() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    let sl_a = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]);
    let sl_b = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]);
    let original = build_connectivity_matrix(&p, &[sl_a, sl_b])?;

    let json = original.to_json()?;
    let restored = ConnectivityMatrix::from_json(&json)?;

    assert_eq!(restored.region_count(), original.region_count());
    assert_eq!(restored.edge_count(), original.edge_count());
    assert_eq!(restored.total_streamlines(), original.total_streamlines());
    assert_eq!(restored.intra_region_count(), original.intra_region_count());
    assert_eq!(restored.skipped_count(), original.skipped_count());

    for &label in &[1, 2, 3] {
        assert_eq!(restored.degree(label), original.degree(label));
        assert!(
            (restored.strength(label).unwrap() - original.strength(label).unwrap()).abs() < 1e-12
        );
        for &other in &[1, 2, 3] {
            assert_eq!(restored.weight(label, other), original.weight(label, other));
        }
    }
    Ok(())
}

#[test]
fn empty_streamline_set_produces_zero_matrix() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    let matrix = build_connectivity_matrix(&p, &[])?;
    assert_eq!(matrix.region_count(), 3);
    assert_eq!(matrix.total_streamlines(), 0);
    assert_eq!(matrix.edge_count(), 0);
    assert_eq!(matrix.density(), 0.0);
    for &label in &[1, 2, 3] {
        assert_eq!(matrix.degree(label), Some(0));
        assert!((matrix.strength(label).unwrap() - 0.0).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn edges_iter_returns_only_nonzero_weights() -> Result<(), ConnectomeError> {
    let p = three_region_2x2x2();
    let sl = polyline_from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]]);
    let matrix = build_connectivity_matrix(&p, &[sl])?;

    let edges: Vec<ConnectivityEdge> = matrix.edges().collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].source, 1);
    assert_eq!(edges[0].target, 2);
    assert!((edges[0].weight - 1.0).abs() < 1e-12);
    Ok(())
}
