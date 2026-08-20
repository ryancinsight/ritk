use super::*;

const TOLERANCE: f64 = 1.0e-12;

/// Rotation by `angle` about the z axis, row-major.
fn rotation_z(angle: f64) -> [f64; 9] {
    let (sin, cos) = angle.sin_cos();
    [cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0]
}

fn assert_close(actual: [f64; 3], expected: [f64; 3], context: &str) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < TOLERANCE,
            "{context}: axis {axis} expected {}, got {}",
            expected[axis],
            actual[axis]
        );
    }
}

// ── Construction ─────────────────────────────────────────────────────────

#[test]
fn zero_extent_is_rejected() {
    let error = ParcellationGrid::axis_aligned([2, 0, 2], [1.0; 3], [0.0; 3]).unwrap_err();
    assert!(matches!(error, ParcellationError::DegenerateGrid { .. }));
}

#[test]
fn nonpositive_spacing_is_rejected() {
    for spacing in [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, f64::NAN, 1.0]] {
        let error = ParcellationGrid::axis_aligned([2, 2, 2], spacing, [0.0; 3]).unwrap_err();
        assert!(
            matches!(error, ParcellationError::DegenerateGrid { .. }),
            "spacing {spacing:?} must be rejected"
        );
    }
}

/// A singular direction matrix has no inverse, so no physical point maps to an
/// index and every lookup would be meaningless. Rejecting at construction is
/// what keeps that from surfacing as silently wrong labels.
#[test]
fn singular_direction_is_rejected() {
    // Two identical rows: rank 2, determinant 0.
    let singular = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let error = ParcellationGrid::new([2, 2, 2], [1.0; 3], [0.0; 3], singular).unwrap_err();
    assert!(matches!(error, ParcellationError::DegenerateGrid { .. }));
}

/// Singularity must be judged relative to the matrix's own scale, so the same
/// geometry expressed in different units classifies the same way.
#[test]
fn a_valid_direction_at_small_scale_is_not_mistaken_for_singular() {
    let scaled = [1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6];
    assert!(ParcellationGrid::new([2, 2, 2], [1.0; 3], [0.0; 3], scaled).is_ok());
}

// ── The affine round-trips ───────────────────────────────────────────────

#[test]
fn axis_aligned_index_maps_to_spacing_times_index_plus_origin() {
    let grid = ParcellationGrid::axis_aligned([4, 4, 4], [2.0, 3.0, 4.0], [10.0, 20.0, 30.0])
        .expect("valid grid");

    assert_close(
        grid.physical_point_of([1, 2, 3]).to_array(),
        [12.0, 26.0, 42.0],
        "voxel centre",
    );
    assert_eq!(
        grid.voxel_of(&Point::new([12.0, 26.0, 42.0])),
        Some([1, 2, 3])
    );
}

/// The affine and its inverse must compose to the identity for every voxel,
/// including under an oblique direction matrix. This is the round-trip that
/// every label lookup depends on.
#[test]
fn every_voxel_round_trips_through_an_oblique_affine() {
    let grid = ParcellationGrid::new(
        [5, 6, 7],
        [1.5, 2.5, 3.5],
        [-4.0, 11.0, 2.5],
        rotation_z(0.4),
    )
    .expect("valid grid");

    for iz in 0..7 {
        for iy in 0..6 {
            for ix in 0..5 {
                let index = [ix, iy, iz];
                let point = grid.physical_point_of(index);
                assert_eq!(
                    grid.voxel_of(&point),
                    Some(index),
                    "voxel {index:?} must round-trip"
                );
            }
        }
    }
}

/// The whole reason the direction matrix is carried: an oblique grid must not
/// resolve to the same voxel an axis-aligned one would.
///
/// A point placed at the axis-aligned position of voxel `(3, 0, 0)` is, under a
/// 40-degree rotation, nowhere near that voxel. If obliquity were dropped the
/// lookup would return `(3, 0, 0)` regardless — the silent misassignment this
/// test exists to forbid.
#[test]
fn obliquity_changes_which_voxel_a_point_falls_in() {
    let spacing = [2.0, 2.0, 2.0];
    let aligned = ParcellationGrid::axis_aligned([8, 8, 8], spacing, [0.0; 3]).expect("valid grid");
    let oblique =
        ParcellationGrid::new([8, 8, 8], spacing, [0.0; 3], rotation_z(0.7)).expect("valid grid");

    let probe = aligned.physical_point_of([3, 0, 0]);
    assert_eq!(aligned.voxel_of(&probe), Some([3, 0, 0]));
    assert_ne!(
        oblique.voxel_of(&probe),
        Some([3, 0, 0]),
        "an oblique grid must not place the point where an axis-aligned one does"
    );
}

/// Half a voxel beyond the outermost centre is still inside the volume: the
/// voxel occupies that space. Excluding it would carve a shell off every
/// surface, which is exactly where cortical labels live.
#[test]
fn the_outer_half_voxel_is_inside_the_volume() {
    let grid = ParcellationGrid::axis_aligned([4, 4, 4], [2.0; 3], [0.0; 3]).expect("valid grid");

    // Last centre on x is 6.0; the voxel extends to 7.0.
    assert_eq!(grid.voxel_of(&Point::new([6.9, 0.0, 0.0])), Some([3, 0, 0]));
    assert_eq!(grid.voxel_of(&Point::new([7.1, 0.0, 0.0])), None);
    // Symmetrically at the low edge: the first voxel extends to -1.0.
    assert_eq!(
        grid.voxel_of(&Point::new([-0.9, 0.0, 0.0])),
        Some([0, 0, 0])
    );
    assert_eq!(grid.voxel_of(&Point::new([-1.1, 0.0, 0.0])), None);
}

#[test]
fn non_finite_coordinates_map_to_no_voxel() {
    let grid = ParcellationGrid::axis_aligned([4, 4, 4], [1.0; 3], [0.0; 3]).expect("valid grid");
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(grid.voxel_of(&Point::new([bad, 0.0, 0.0])), None);
        assert_eq!(grid.continuous_index_of(&Point::new([0.0, bad, 0.0])), None);
    }
}

// ── Storage order ────────────────────────────────────────────────────────

/// The flat offset and the index must be exact inverses, or a label read lands
/// on the wrong voxel.
#[test]
fn offset_and_index_are_inverse_over_the_whole_volume() {
    let grid = ParcellationGrid::axis_aligned([3, 4, 5], [1.0; 3], [0.0; 3]).expect("valid grid");

    for offset in 0..grid.voxel_count() {
        let index = grid.index_of_offset(offset).expect("in-range offset");
        assert_eq!(grid.offset_of(index), Some(offset), "offset {offset}");
    }
    assert_eq!(grid.index_of_offset(grid.voxel_count()), None);
    assert_eq!(grid.offset_of([3, 0, 0]), None);
}

/// Storage is z-major: x is the fastest axis, matching what every volumetric
/// format writes.
#[test]
fn storage_order_advances_x_fastest() {
    let grid = ParcellationGrid::axis_aligned([3, 4, 5], [1.0; 3], [0.0; 3]).expect("valid grid");

    assert_eq!(grid.offset_of([1, 0, 0]), Some(1));
    assert_eq!(grid.offset_of([0, 1, 0]), Some(3));
    assert_eq!(grid.offset_of([0, 0, 1]), Some(12));
}

// ── Volume ───────────────────────────────────────────────────────────────

#[test]
fn voxel_volume_is_the_product_of_the_spacings_for_an_orthonormal_frame() {
    let grid = ParcellationGrid::new([2, 2, 2], [1.5, 2.0, 4.0], [0.0; 3], rotation_z(1.1))
        .expect("valid grid");
    assert!((grid.voxel_volume() - 12.0).abs() < TOLERANCE);
}

// ── Neighbourhood bounds ─────────────────────────────────────────────────

/// For an orthonormal direction matrix the bound reduces to `radius/spacing`,
/// which is the value a hand-written search would use.
#[test]
fn index_bounds_reduce_to_radius_over_spacing_when_axis_aligned() {
    let grid =
        ParcellationGrid::axis_aligned([64; 3], [1.0, 2.0, 4.0], [0.0; 3]).expect("valid grid");
    assert_eq!(grid.index_radius_bounds(4.0), [4, 2, 1]);
}

/// The bound must never *under*-count, or a neighbourhood search silently
/// misses voxels. Checked by brute force: no offset outside the bound may lie
/// within the radius.
#[test]
fn index_bounds_cover_every_offset_within_the_radius() {
    let grid = ParcellationGrid::new([64; 3], [1.0, 2.0, 3.0], [0.0; 3], rotation_z(0.3))
        .expect("valid grid");
    let radius = 5.0;
    let bounds = grid.index_radius_bounds(radius);

    let span = 12_isize;
    for dz in -span..=span {
        for dy in -span..=span {
            for dx in -span..=span {
                if grid.physical_displacement_of([dx, dy, dz]).sqrt() > radius {
                    continue;
                }
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "bounds are small grid-derived counts"
                )]
                let limits = bounds.map(|bound| bound as isize);
                assert!(
                    dx.abs() <= limits[0] && dy.abs() <= limits[1] && dz.abs() <= limits[2],
                    "offset ({dx}, {dy}, {dz}) is within {radius} mm but outside the bound {bounds:?}"
                );
            }
        }
    }
}

#[test]
fn an_unusable_radius_yields_no_neighbourhood() {
    let grid = ParcellationGrid::axis_aligned([8; 3], [1.0; 3], [0.0; 3]).expect("valid grid");
    for radius in [-1.0, f64::NAN] {
        assert_eq!(grid.index_radius_bounds(radius), [0; 3]);
    }
    assert_eq!(grid.index_radius_bounds(0.0), [0; 3]);
}

/// Displacement is measured through the affine, so an anisotropic grid reports
/// true millimetres rather than a voxel count.
#[test]
fn physical_displacement_uses_the_grid_spacing() {
    let grid =
        ParcellationGrid::axis_aligned([8; 3], [1.0, 2.0, 4.0], [0.0; 3]).expect("valid grid");
    assert!((grid.physical_displacement_of([1, 0, 0]) - 1.0).abs() < TOLERANCE);
    assert!((grid.physical_displacement_of([0, 1, 0]) - 4.0).abs() < TOLERANCE);
    assert!((grid.physical_displacement_of([0, 0, 1]) - 16.0).abs() < TOLERANCE);
    // A rotation is an isometry, so it leaves lengths alone.
    let rotated = ParcellationGrid::new([8; 3], [1.0, 2.0, 4.0], [0.0; 3], rotation_z(0.9))
        .expect("valid grid");
    assert!((rotated.physical_displacement_of([1, 1, 1]) - 21.0).abs() < TOLERANCE);
}
