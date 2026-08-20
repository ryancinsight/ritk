use super::*;

const TOLERANCE: f64 = 1.0e-12;

/// A 5×5×1 slab with one labelled voxel at index `(0, 0, 0)` and another at
/// `(4, 0, 0)`, everything else background.
///
/// Two isolated labels separated by a wide background gap is the configuration
/// that makes "nearest" a real question rather than a formality.
fn two_islands() -> Parcellation {
    let mut labels = vec![BACKGROUND; 25];
    labels[0] = 3;
    labels[4] = 8;
    let grid =
        ParcellationGrid::axis_aligned([5, 5, 1], [1.0, 1.0, 1.0], [0.0; 3]).expect("valid grid");
    Parcellation::new(labels.into_boxed_slice(), grid, Vec::new()).expect("valid parcellation")
}

fn search(parcellation: &Parcellation, radius: f64) -> NearestLabelSearch {
    NearestLabelSearch::new(parcellation.grid(), radius).expect("valid radius")
}

// ── Construction ─────────────────────────────────────────────────────────

#[test]
fn an_unusable_radius_is_rejected() {
    let parcellation = two_islands();
    for radius in [-1.0, f64::NAN, f64::INFINITY] {
        let error = NearestLabelSearch::new(parcellation.grid(), radius).unwrap_err();
        assert!(
            matches!(error, ParcellationError::InvalidRadius { .. }),
            "radius {radius} must be rejected"
        );
    }
}

/// A zero radius is the degenerate search that inspects only the voxel under the
/// point — the plain endpoint lookup, expressed through the same API.
#[test]
fn a_zero_radius_inspects_only_the_containing_voxel() {
    let parcellation = two_islands();
    let search = search(&parcellation, 0.0);

    assert_eq!(search.neighbourhood_size(), 1);
    assert_eq!(
        search
            .find(&parcellation, &Point::new([0.0, 0.0, 0.0]))
            .map(|found| found.label),
        Some(3)
    );
    // One voxel away is background, and a zero radius cannot reach past it.
    assert_eq!(
        search.find(&parcellation, &Point::new([1.0, 0.0, 0.0])),
        None
    );
}

// ── Nearest-first ordering ───────────────────────────────────────────────

/// The point sits between the two islands but closer to the second, so the
/// search must return that one. Returning either would satisfy a test that only
/// checked "some label was found", which is why the asymmetry matters.
#[test]
fn the_nearer_of_two_labels_is_returned() {
    let parcellation = two_islands();
    let search = search(&parcellation, 4.0);

    let found = search
        .find(&parcellation, &Point::new([3.0, 0.0, 0.0]))
        .expect("a label within 4 mm");
    assert_eq!(found.label, 8);
    assert_eq!(found.index, [4, 0, 0]);
    assert!((found.distance - 1.0).abs() < TOLERANCE, "{found:?}");

    let found = search
        .find(&parcellation, &Point::new([1.0, 0.0, 0.0]))
        .expect("a label within 4 mm");
    assert_eq!(found.label, 3);
    assert!((found.distance - 1.0).abs() < TOLERANCE, "{found:?}");
}

/// A point already inside a labelled voxel is assigned to it at zero distance,
/// whatever the radius. Widening the search must never pull an endpoint out of
/// the region it is already in.
#[test]
fn a_point_inside_a_region_is_assigned_to_it_at_zero_distance() {
    let parcellation = two_islands();
    for radius in [0.0, 1.0, 5.0] {
        let found = search(&parcellation, radius)
            .find(&parcellation, &Point::new([0.0, 0.0, 0.0]))
            .expect("the containing voxel is labelled");
        assert_eq!(found.label, 3, "radius {radius}");
        assert!(found.distance.abs() < TOLERANCE, "radius {radius}");
    }
}

// ── The radius bounds the assignment ─────────────────────────────────────

/// The radius is a hard limit, not a hint: a label just outside it must not be
/// returned. This is what stops a widened search from reaching across a sulcus.
#[test]
fn a_label_beyond_the_radius_is_not_returned() {
    let parcellation = two_islands();

    // From (2, 0, 0) both islands are exactly 2 mm away.
    assert_eq!(
        search(&parcellation, 1.9).find(&parcellation, &Point::new([2.0, 0.0, 0.0])),
        None
    );
    assert!(
        search(&parcellation, 2.0)
            .find(&parcellation, &Point::new([2.0, 0.0, 0.0]))
            .is_some(),
        "a label exactly at the radius is inside it"
    );
}

/// Growing the radius can only find more, never fewer, endpoints — and never
/// change one that was already assigned at a shorter radius.
#[test]
fn widening_the_radius_is_monotone() {
    let parcellation = two_islands();
    let probes = [
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 1.0, 0.0]),
        Point::new([2.0, 2.0, 0.0]),
        Point::new([3.0, 1.0, 0.0]),
    ];

    for probe in &probes {
        let mut assigned_at: Option<u32> = None;
        for radius in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0] {
            let found = search(&parcellation, radius).find(&parcellation, probe);
            if let Some(previous) = assigned_at {
                let found = found.expect("a wider radius cannot lose an assignment");
                assert_eq!(
                    found.label, previous,
                    "widening to {radius} mm changed the assignment for {probe:?}"
                );
            }
            if let Some(found) = found {
                assigned_at = Some(found.label);
            }
        }
    }
}

// ── Geometry is respected ────────────────────────────────────────────────

/// Distances are physical, so an anisotropic grid must not treat one voxel step
/// on a coarse axis as equal to one on a fine axis.
#[test]
fn distances_are_physical_rather_than_voxel_counts() {
    // 1 mm on x, 10 mm on y. The label one step away on y is 10 mm off.
    let mut labels = vec![BACKGROUND; 9];
    labels[3] = 4; // index (0, 1, 0)
    let grid =
        ParcellationGrid::axis_aligned([3, 3, 1], [1.0, 10.0, 1.0], [0.0; 3]).expect("valid grid");
    let parcellation =
        Parcellation::new(labels.into_boxed_slice(), grid, Vec::new()).expect("valid parcellation");

    // A 2 mm radius spans two voxels on x but cannot reach the next y row.
    assert_eq!(
        search(&parcellation, 2.0).find(&parcellation, &Point::new([0.0, 0.0, 0.0])),
        None
    );
    let found = search(&parcellation, 10.0)
        .find(&parcellation, &Point::new([0.0, 0.0, 0.0]))
        .expect("10 mm reaches the next row");
    assert_eq!(found.label, 4);
    assert!((found.distance - 10.0).abs() < TOLERANCE, "{found:?}");
}

/// A search over an oblique grid must find the same anatomical voxel an
/// axis-aligned search would, because a rotation is an isometry and cannot
/// change which voxel is nearest.
#[test]
fn an_oblique_grid_finds_the_same_neighbour_as_an_aligned_one() {
    let angle = 0.6_f64;
    let (sin, cos) = angle.sin_cos();
    let rotation = [cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0];

    let mut labels = vec![BACKGROUND; 25];
    labels[0] = 3;
    labels[4] = 8;

    let aligned = Parcellation::new(
        labels.clone().into_boxed_slice(),
        ParcellationGrid::axis_aligned([5, 5, 1], [1.0; 3], [0.0; 3]).expect("valid grid"),
        Vec::new(),
    )
    .expect("valid parcellation");
    let oblique = Parcellation::new(
        labels.into_boxed_slice(),
        ParcellationGrid::new([5, 5, 1], [1.0; 3], [0.0; 3], rotation).expect("valid grid"),
        Vec::new(),
    )
    .expect("valid parcellation");

    // The same *index-space* probe, mapped through each grid's own affine.
    let probe_index = [3.0, 0.0, 0.0];
    let aligned_found = search(&aligned, 4.0)
        .find(
            &aligned,
            &aligned.grid().physical_point_of_continuous(probe_index),
        )
        .expect("assignment");
    let oblique_found = search(&oblique, 4.0)
        .find(
            &oblique,
            &oblique.grid().physical_point_of_continuous(probe_index),
        )
        .expect("assignment");

    assert_eq!(aligned_found.label, oblique_found.label);
    assert_eq!(aligned_found.index, oblique_found.index);
    assert!(
        (aligned_found.distance - oblique_found.distance).abs() < TOLERANCE,
        "a rotation is an isometry, so the distance must be unchanged: \
         {aligned_found:?} vs {oblique_found:?}"
    );
}

// ── Out of range ─────────────────────────────────────────────────────────

#[test]
fn a_point_outside_the_volume_has_no_assignment() {
    let parcellation = two_islands();
    let search = search(&parcellation, 5.0);

    assert_eq!(
        search.find(&parcellation, &Point::new([-50.0, 0.0, 0.0])),
        None
    );
    assert_eq!(
        search.find(&parcellation, &Point::new([f64::NAN, 0.0, 0.0])),
        None
    );
}

/// The search must not run off the edge of the volume when the query sits on a
/// boundary voxel; the neighbourhood is clipped, not wrapped.
#[test]
fn the_neighbourhood_is_clipped_at_the_volume_boundary() {
    let parcellation = two_islands();
    let found = search(&parcellation, 3.0)
        .find(&parcellation, &Point::new([0.0, 0.0, 0.0]))
        .expect("the corner voxel is labelled");
    assert_eq!(found.index, [0, 0, 0]);
}

/// A point off the centre of its voxel, with labels on both sides at different
/// true distances but the same offset distance.
#[test]
fn probe_off_centre_returns_the_truly_nearest_label() {
    let mut labels = vec![BACKGROUND; 3];
    labels[0] = 7; // voxel centre at x = 0
    labels[2] = 9; // voxel centre at x = 2
    let grid = ParcellationGrid::axis_aligned([3, 1, 1], [1.0; 3], [0.0; 3]).expect("valid grid");
    let parcellation =
        Parcellation::new(labels.into_boxed_slice(), grid, Vec::new()).expect("valid parcellation");

    // The point sits in the middle voxel but well towards region 9.
    let probe = Point::new([1.4, 0.0, 0.0]);
    let found = search(&parcellation, 2.0)
        .find(&parcellation, &probe)
        .expect("a label within 2 mm");
    assert_eq!(
        found.label, 9,
        "region 9 is 0.6 mm away and region 7 is 1.4 mm; got {found:?}"
    );
}

/// A voxel within the radius of the *point* but further than the radius from
/// the containing voxel's *centre* must still be found.
///
/// Enumerating offsets only out to the radius would never consider it: the two
/// measurements differ by how far the point sits from its own voxel's centre,
/// so the neighbourhood has to reach half a voxel diagonal further than the
/// radius it serves.
#[test]
fn a_label_inside_the_radius_of_the_point_is_found_past_the_centres_radius() {
    let mut labels = vec![BACKGROUND; 3];
    labels[1] = 5; // voxel centre at x = 1
    let grid = ParcellationGrid::axis_aligned([3, 1, 1], [1.0; 3], [0.0; 3]).expect("valid grid");
    let parcellation =
        Parcellation::new(labels.into_boxed_slice(), grid, Vec::new()).expect("valid parcellation");

    // The point is in voxel 0, 0.6 mm from the label — inside a 0.7 mm radius —
    // while the two voxel centres are 1.0 mm apart, outside it.
    let probe = Point::new([0.4, 0.0, 0.0]);
    let found = search(&parcellation, 0.7)
        .find(&parcellation, &probe)
        .expect("the label is 0.6 mm from the point");
    assert_eq!(found.label, 5);
    assert!((found.distance - 0.6).abs() < 1.0e-12, "{found:?}");
}
