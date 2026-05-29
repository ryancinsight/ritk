use super::*;

// ── Positive tests ────────────────────────────────────────────────────────

#[test]
fn test_uniform_image_grows_entire_volume() {
    // All voxels intensity 100; bounds [50, 150]; radius [1,1,1].
    // Every voxel's 3×3×3 neighborhood is uniform → all admissible.
    // Entire 4×4×4 volume is grown.
    let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
    let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result),
        64,
        "uniform region must grow entire volume"
    );
}

#[test]
fn test_radius_zero_equivalent_to_connected_threshold() {
    // With radius [0,0,0], the neighborhood is the voxel itself.
    // This degenerates to the connected-threshold predicate.
    let values = vec![
        100.0, 100.0, 100.0, //
        100.0, 100.0, 100.0, //
        100.0, 100.0, 100.0, //
        50.0, 50.0, 50.0, //
        50.0, 50.0, 50.0, //
        50.0, 50.0, 50.0, //
        100.0, 100.0, 100.0, //
        100.0, 100.0, 100.0, //
        100.0, 100.0, 100.0, //
    ];
    let image = make_image(values, [3, 3, 3]);
    // Seed at (0,0,0) intensity=100, bounds [80,120]. Region z=0 and z=2
    // have intensity 100; z=1 has intensity 50 (outside bounds).
    // With radius=0, single-voxel predicate: z=0 plane (9 voxels) grows,
    // z=1 blocks, z=2 unreachable.
    let result = neighborhood_connected(&image, [0, 0, 0], 80.0, 120.0, [0, 0, 0]);
    assert_eq!(
        count_foreground(&result),
        9,
        "radius=0 must behave as connected-threshold"
    );
}

#[test]
fn test_two_regions_seed_selects_one() {
    // 1×1×6 volume: [100, 100, 100, 10, 10, 10].
    // Seed at (0,0,0) with bounds [50, 200], radius [0,0,0].
    // Only first 3 voxels qualify.
    let values = vec![100.0, 100.0, 100.0, 10.0, 10.0, 10.0];
    let image = make_image(values, [1, 1, 6]);
    let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 200.0, [0, 0, 0]);
    let vals = get_values(&result);
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 1.0);
    assert_eq!(vals[2], 1.0);
    assert_eq!(vals[3], 0.0);
    assert_eq!(vals[4], 0.0);
    assert_eq!(vals[5], 0.0);
}

#[test]
fn test_neighborhood_rejects_noisy_voxel() {
    // 3×3×3 uniform intensity 100, except one voxel at (0,1,1)=5.0 (noise).
    // Seed at (1,1,1) center. Bounds [80, 120], radius [1,1,1].
    //
    // Voxel (0,1,1) itself is out of bounds → it fails the neighborhood check.
    // Additionally, any voxel whose 3×3×3 neighborhood includes (0,1,1) also
    // fails, because that neighbor has intensity 5.0 < 80.
    //
    // The noisy voxel at (0,1,1) affects neighbors at offsets within radius 1
    // in each axis. The affected voxels are those whose neighborhood includes
    // (0,1,1), i.e., voxels v with |vz-0|≤1, |vy-1|≤1, |vx-1|≤1.
    // That is z∈{0,1}, y∈{0,1,2}, x∈{0,1,2} → 2×3×3 = 18 voxels affected.
    // Total admissible = 27 - 18 = 9 voxels (the z=2 plane).
    // But the seed (1,1,1) is in the affected zone → seed itself is NOT admissible.
    let mut values = vec![100.0_f32; 27];
    values[3 + 1] = 5.0; // (0,1,1) = noise
    let image = make_image(values, [3, 3, 3]);
    let result = neighborhood_connected(&image, [1, 1, 1], 80.0, 120.0, [1, 1, 1]);
    // Seed's neighborhood includes (0,1,1) which has intensity 5.0 < 80.
    // Therefore seed is not admissible → empty mask.
    assert_eq!(
        count_foreground(&result),
        0,
        "noisy voxel in seed's neighborhood must cause empty result"
    );
}

#[test]
fn test_neighborhood_radius_restricts_boundary() {
    // 5×5×5 image: interior 3×3×3 cube at center has intensity 200,
    // outer shell has intensity 50. Bounds [150, 255].
    //
    // With radius [0,0,0]: the 3×3×3 interior cube (27 voxels) is grown.
    // With radius [1,1,1]: voxels on the surface of the 3×3×3 cube have
    // neighborhoods that extend into the shell (intensity 50 < 150),
    // so they fail the neighborhood check. Only the single center voxel
    // (2,2,2) has a 3×3×3 neighborhood entirely within the 3×3×3 cube.
    let (nz, ny, nx) = (5, 5, 5);
    let mut values = vec![50.0_f32; nz * ny * nx];
    for iz in 1..4 {
        for iy in 1..4 {
            for ix in 1..4 {
                values[iz * ny * nx + iy * nx + ix] = 200.0;
            }
        }
    }
    let image = make_image(values, [nz, ny, nx]);

    // radius [0,0,0]: connected-threshold behavior, captures entire cube.
    let result_r0 = neighborhood_connected(&image, [2, 2, 2], 150.0, 255.0, [0, 0, 0]);
    assert_eq!(
        count_foreground(&result_r0),
        27,
        "radius=0 must capture entire 3×3×3 cube"
    );

    // radius [1,1,1]: only the center voxel (2,2,2) has a full 3×3×3
    // neighborhood inside the high-intensity cube.
    let result_r1 = neighborhood_connected(&image, [2, 2, 2], 150.0, 255.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result_r1),
        1,
        "radius=1 must capture only center voxel of 3×3×3 cube in 5×5×5 image"
    );
}

#[test]
fn test_larger_interior_with_radius() {
    // 7×7×7 image: interior 5×5×5 cube (indices 1..6) has intensity 200,
    // outer shell has intensity 50. Bounds [150, 255].
    //
    // With radius [1,1,1]: voxels whose 3×3×3 neighborhoods are entirely
    // within the 5×5×5 cube are those at indices 2..5 in each axis → 3×3×3 = 27.
    let (nz, ny, nx) = (7, 7, 7);
    let mut values = vec![50.0_f32; nz * ny * nx];
    for iz in 1..6 {
        for iy in 1..6 {
            for ix in 1..6 {
                values[iz * ny * nx + iy * nx + ix] = 200.0;
            }
        }
    }
    let image = make_image(values, [nz, ny, nx]);
    let result = neighborhood_connected(&image, [3, 3, 3], 150.0, 255.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result),
        27,
        "radius=1 on 5×5×5 interior in 7×7×7 must capture 3×3×3 = 27 voxels"
    );
}
