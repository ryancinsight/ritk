use super::*;

// ── Adversarial tests ─────────────────────────────────────────────────────

/// Two isolated high-intensity cubes separated by zero-intensity background.
/// Seeding from each cube must not cross-contaminate the other.
///
/// Image: 1×1×8 = [100, 100, 100, 0, 0, 200, 200, 200]
/// Seed A at [0,0,0] with bounds [50, 150], radius [0,0,0] → 3 voxels.
/// Seed B at [0,0,5] with bounds [150, 250], radius [0,0,0] → 3 voxels.
#[test]
fn test_multi_seed_two_cubes_no_cross_contamination() {
    let values = vec![100.0_f32, 100.0, 100.0, 0.0, 0.0, 200.0, 200.0, 200.0];
    let image = make_image(values, [1, 1, 8]);

    let result_a = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [0, 0, 0]);
    assert_eq!(
        count_foreground(&result_a),
        3,
        "seed in cube A must grow exactly 3 voxels"
    );
    let vals_a = get_values(&result_a);
    assert_eq!(
        vals_a[5], 0.0,
        "cube B position 5 must not be reached from seed A"
    );
    assert_eq!(
        vals_a[7], 0.0,
        "cube B position 7 must not be reached from seed A"
    );

    let result_b = neighborhood_connected(&image, [0, 0, 5], 150.0, 250.0, [0, 0, 0]);
    assert_eq!(
        count_foreground(&result_b),
        3,
        "seed in cube B must grow exactly 3 voxels"
    );
    let vals_b = get_values(&result_b);
    assert_eq!(
        vals_b[0], 0.0,
        "cube A position 0 must not be reached from seed B"
    );
    assert_eq!(
        vals_b[2], 0.0,
        "cube A position 2 must not be reached from seed B"
    );
}

/// Seed at volume corner [0,0,0] with radius [2,2,2] on a 3×3×3 uniform image.
/// The neighborhood is clamped to [0, min(2, nDim-1)] in each axis.
/// Since all voxels are uniform and within bounds, the entire 27-voxel volume
/// must be grown (clamping does not prevent growth; it restricts neighborhood check).
/// Analytical proof: for any voxel v with all neighbors in [0, 150], P(v) holds
/// for any radius as long as the domain-clamped neighborhood stays in [50, 150].
#[test]
fn test_boundary_seed_radius_overflow_clamped_to_domain() {
    // 3×3×3 uniform intensity 100; bounds [50, 150]; radius [2,2,2].
    // Radius [2,2,2] on a 3×3×3 image: neighborhood clamped to [0,2] × [0,2] × [0,2]
    // = entire image. All voxels are 100 ∈ [50, 150] → all admissible → all grown.
    let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
    let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [2, 2, 2]);
    assert_eq!(
        count_foreground(&result),
        27,
        "radius [2,2,2] clamped to 3×3×3 domain must grow all 27 voxels"
    );
}

/// Large uniform image with large radius: every voxel's neighborhood is
/// fully within bounds, so the entire volume is grown from any seed.
///
/// 6×6×6 image, all intensity 100, bounds [50, 150], radius [1,1,1].
/// Every voxel's 3×3×3 (clamped) neighborhood is all-100 ∈ [50, 150].
/// All 216 voxels must be grown.
#[test]
fn test_large_uniform_image_large_radius_grows_all_voxels() {
    let image = make_image(vec![100.0_f32; 216], [6, 6, 6]);
    let result = neighborhood_connected(&image, [3, 3, 3], 50.0, 150.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result),
        216,
        "6×6×6 uniform image with radius [1,1,1] must grow all 216 voxels"
    );
}

/// Adversarial: noisy background at image boundary; seed deep inside uniform
/// interior; large radius must still reject boundary voxels that touch noise.
///
/// 5×5×5 image: interior 3×3×3 cube (indices 1..4) = 200; boundary shell = 5 (noise).
/// Bounds [150, 255], radius [1,1,1].
///
/// With radius [1,1,1]: voxels on the surface of the 3×3×3 cube have neighborhoods
/// that extend into the shell (intensity 5 < 150) → rejected.
/// Only the center voxel (2,2,2) whose 3×3×3 neighborhood is entirely within the
/// high-intensity cube is admissible → 1 voxel grown.
/// This is the adversarial equivalent of test_neighborhood_radius_restricts_boundary
/// but with noise intensity ≪ lower bound (5 vs 50), stressing the boundary rejection.
#[test]
fn test_adversarial_noisy_boundary_large_radius_rejects_surface_voxels() {
    let (nz, ny, nx) = (5, 5, 5);
    let mut values = vec![5.0_f32; nz * ny * nx]; // extreme noise in boundary shell
    for iz in 1..4 {
        for iy in 1..4 {
            for ix in 1..4 {
                values[iz * ny * nx + iy * nx + ix] = 200.0;
            }
        }
    }
    let image = make_image(values, [nz, ny, nx]);
    let result = neighborhood_connected(&image, [2, 2, 2], 150.0, 255.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result),
        1,
        "only center voxel (2,2,2) must be grown when boundary shell contains noise"
    );
    // Verify the center voxel is foreground.
    let vals = get_values(&result);
    let center_flat = 2 * ny * nx + 2 * nx + 2;
    assert_eq!(
        vals[center_flat], 1.0,
        "center voxel (2,2,2) must be foreground"
    );
}
