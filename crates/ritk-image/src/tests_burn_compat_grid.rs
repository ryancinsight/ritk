use super::{generate_grid_burn, generate_random_points_burn};

type B = burn_ndarray::NdArray<f32>;

/// The legacy grid preserves the native row-major, innermost-first contract.
#[test]
fn burn_grid_has_deterministic_innermost_first_coordinates() {
    let device = Default::default();
    let values = generate_grid_burn::<B, 3>([2, 2, 3], &device)
        .into_data()
        .as_slice::<f32>()
        .expect("NdArray tensor data must expose f32 values")
        .to_vec();

    assert_eq!(
        values,
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            1.0, 1.0,
        ],
        "each row must be [x, y, z] with x varying first"
    );
}

/// Random points remain inside the same per-column extent as grid coordinates.
#[test]
fn burn_random_points_respect_innermost_first_bounds() {
    let device = Default::default();
    let values = generate_random_points_burn::<B, 3>([2, 3, 4], 64, &device)
        .into_data()
        .as_slice::<f32>()
        .expect("NdArray tensor data must expose f32 values")
        .to_vec();

    for point in values.chunks_exact(3) {
        assert!((0.0..3.0).contains(&point[0]), "x bound");
        assert!((0.0..2.0).contains(&point[1]), "y bound");
        assert!((0.0..1.0).contains(&point[2]), "z bound");
    }
}
