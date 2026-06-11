use super::*;

// ── Predicate correctness test ────────────────────────────────────────────

#[test]
fn test_is_neighborhood_admissible_direct() {
    // Direct unit test of the internal predicate function.
    // 3×3×3 uniform intensity 100.
    let data = vec![100.0_f32; 27];
    let dims = [3, 3, 3];

    // Center voxel, radius [1,1,1]: all 27 voxels are 100.0 ∈ [50, 150].
    assert!(is_neighborhood_admissible(
        &data,
        dims,
        [1, 1, 1],
        50.0,
        150.0,
        [1, 1, 1]
    ));

    // Corner voxel (0,0,0), radius [1,1,1]: neighborhood clamped to
    // z∈[0,1], y∈[0,1], x∈[0,1] → 8 voxels, all 100.0.
    assert!(is_neighborhood_admissible(
        &data,
        dims,
        [0, 0, 0],
        50.0,
        150.0,
        [1, 1, 1]
    ));

    // Bounds exclude all voxels: lower=200 > all intensities.
    assert!(!is_neighborhood_admissible(
        &data,
        dims,
        [1, 1, 1],
        200.0,
        300.0,
        [1, 1, 1]
    ));

    // Single bad voxel in neighborhood.
    let mut data_bad = vec![100.0_f32; 27];
    data_bad[0] = 5.0; // (0,0,0) is bad
                       // Voxel (1,1,1) with radius [1,1,1] includes (0,0,0) → not admissible.
    assert!(!is_neighborhood_admissible(
        &data_bad,
        dims,
        [1, 1, 1],
        50.0,
        150.0,
        [1, 1, 1]
    ));
    // Voxel (2,2,2) with radius [1,1,1]: neighborhood is z∈[1,2], y∈[1,2], x∈[1,2].
    // Does NOT include (0,0,0) → admissible.
    assert!(is_neighborhood_admissible(
        &data_bad,
        dims,
        [2, 2, 2],
        50.0,
        150.0,
        [1, 1, 1]
    ));
}
