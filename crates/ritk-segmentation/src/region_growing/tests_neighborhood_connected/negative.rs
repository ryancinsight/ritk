use super::*;

// ── Negative / boundary tests ─────────────────────────────────────────────

#[test]
fn test_seed_neighborhood_outside_range_returns_empty() {
    // Seed intensity = 100, but neighbor at seed-1 has intensity 5 < 50.
    // With radius [1,1,1], seed neighborhood includes the bad voxel → empty.
    let mut values = vec![100.0_f32; 8]; // 2×2×2
    values[0] = 5.0; // (0,0,0) low intensity
    let image = make_image(values, [2, 2, 2]);
    // Seed at (0,0,1): neighborhood includes (0,0,0) which has val 5.0 < 50.
    let result = neighborhood_connected(&image, [0, 0, 1], 50.0, 150.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result),
        0,
        "seed with bad neighborhood must produce empty region"
    );
}

#[test]
fn test_seed_voxel_itself_below_lower_returns_empty() {
    // Even with radius [0,0,0], if the seed intensity is below lower → empty.
    let image = make_image(vec![5.0_f32; 8], [2, 2, 2]);
    let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 200.0, [0, 0, 0]);
    assert_eq!(count_foreground(&result), 0);
}

#[test]
fn test_seed_voxel_itself_above_upper_returns_empty() {
    let image = make_image(vec![201.0_f32; 1], [1, 1, 1]);
    let result = neighborhood_connected(&image, [0, 0, 0], 0.0, 200.0, [0, 0, 0]);
    assert_eq!(count_foreground(&result), 0);
}

#[test]
fn test_strictly_binary_output() {
    let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
    let result = neighborhood_connected(&image, [1, 1, 1], 50.0, 150.0, [1, 1, 1]);
    for &v in get_values(&result).iter() {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be strictly binary, got {v}"
        );
    }
}

#[test]
fn test_connectivity_is_6_not_diagonal() {
    // 3×3×1 slice:
    //   A 0 0
    //   0 0 0
    //   0 0 B
    // A and B have high intensity; all others low.
    // With 6-connectivity, seeding from A cannot reach B diagonally.
    let mut values = vec![0.0_f32; 9];
    values[0] = 200.0; // A at (0,0,0)
    values[8] = 200.0; // B at (0,2,2)
    let image = make_image(values, [1, 3, 3]);
    let result = neighborhood_connected(&image, [0, 0, 0], 100.0, 255.0, [0, 0, 0]);
    let vals = get_values(&result);
    assert_eq!(vals[0], 1.0, "seed voxel A must be foreground");
    assert_eq!(vals[8], 0.0, "diagonal voxel B must not be reached");
    assert_eq!(count_foreground(&result), 1);
}
