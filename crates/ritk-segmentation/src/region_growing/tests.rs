// ── Tests ─────────────────────────────────────────────────────────────────────

use super::{connected_threshold, ConnectedThresholdFilter};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support::burn_compat::make_image;

type TestBackend = NdArray<f32>;

fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn count_foreground(image: &Image<TestBackend, 3>) -> usize {
    get_values(image).iter().filter(|&&v| v > 0.5).count()
}

// ── Positive tests ────────────────────────────────────────────────────────────

#[test]
fn test_uniform_image_grows_entire_volume() {
    // All voxels have intensity 100; lower=50, upper=150 → entire 4×4×4 grown.
    let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
    let result = connected_threshold(&image, [0, 0, 0], 50.0, 150.0);
    assert_eq!(count_foreground(&result), 64);
}

#[test]
fn test_single_voxel_exactly_on_lower_bound() {
    // Seed has intensity exactly equal to lower → should be included.
    let image = make_image(vec![50.0_f32; 8], [2, 2, 2]);
    let result = connected_threshold(&image, [0, 0, 0], 50.0, 100.0);
    assert_eq!(count_foreground(&result), 8);
}

#[test]
fn test_single_voxel_exactly_on_upper_bound() {
    let image = make_image(vec![100.0_f32; 8], [2, 2, 2]);
    let result = connected_threshold(&image, [1, 1, 1], 50.0, 100.0);
    assert_eq!(count_foreground(&result), 8);
}

#[test]
fn test_two_regions_seed_selects_one() {
    // 1×1×6 volume: [100, 100, 100, 10, 10, 10].
    // Seed at (0,0,0) with lower=50, upper=200 → only first 3 voxels.
    let values = vec![100.0, 100.0, 100.0, 10.0, 10.0, 10.0];
    let image = make_image(values, [1, 1, 6]);
    let result = connected_threshold(&image, [0, 0, 0], 50.0, 200.0);
    let vals = get_values(&result);
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 1.0);
    assert_eq!(vals[2], 1.0);
    assert_eq!(vals[3], 0.0);
    assert_eq!(vals[4], 0.0);
    assert_eq!(vals[5], 0.0);
}

#[test]
fn test_connectivity_is_6_not_diagonal() {
    // 3×3×1 slice:
    //   A 0 0
    //   0 0 0
    //   0 0 B
    // A and B are high-intensity; all other voxels are low.
    // With 6-connectivity, seeding from A cannot reach B.
    let mut values = vec![0.0_f32; 9];
    values[0] = 200.0; // A at (0,0,0)
    values[8] = 200.0; // B at (0,2,2)
    let image = make_image(values, [1, 3, 3]);
    let result = connected_threshold(&image, [0, 0, 0], 100.0, 255.0);
    let vals = get_values(&result);
    // Only A should be foreground; B is not 6-connected to A.
    assert_eq!(vals[0], 1.0, "seed voxel A must be foreground");
    assert_eq!(vals[8], 0.0, "diagonal voxel B must not be reached");
    assert_eq!(count_foreground(&result), 1);
}

#[test]
fn test_filter_struct_matches_function() {
    let values: Vec<f32> = (0..27).map(|i| i as f32 * 10.0).collect();
    let image = make_image(values, [3, 3, 3]);

    let via_fn = connected_threshold(&image, [1, 1, 1], 50.0, 150.0);
    let via_struct = ConnectedThresholdFilter::new([1, 1, 1], 50.0, 150.0).apply(&image);

    let fn_vals = get_values(&via_fn);
    let struct_vals = get_values(&via_struct);
    assert_eq!(
        fn_vals, struct_vals,
        "function and struct must produce identical results"
    );
}

// ── Negative / boundary tests ─────────────────────────────────────────────────

#[test]
fn test_seed_outside_range_returns_all_zero() {
    // Seed intensity = 5.0, range [50, 200] → seed excluded → empty mask.
    let image = make_image(vec![5.0_f32; 8], [2, 2, 2]);
    let result = connected_threshold(&image, [0, 0, 0], 50.0, 200.0);
    assert_eq!(count_foreground(&result), 0);
}

#[test]
fn test_intensity_just_above_upper_bound_excluded() {
    // Single voxel with intensity 201 when upper = 200.
    let image = make_image(vec![201.0_f32; 1], [1, 1, 1]);
    let result = connected_threshold(&image, [0, 0, 0], 0.0, 200.0);
    assert_eq!(count_foreground(&result), 0);
}

#[test]
fn test_intensity_just_below_lower_bound_excluded() {
    let image = make_image(vec![49.0_f32; 1], [1, 1, 1]);
    let result = connected_threshold(&image, [0, 0, 0], 50.0, 200.0);
    assert_eq!(count_foreground(&result), 0);
}

#[test]
fn test_fully_grown_region_output_is_strictly_binary() {
    let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
    let result = connected_threshold(&image, [1, 1, 1], 50.0, 150.0);
    for &v in get_values(&result).iter() {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be strictly binary, got {v}"
        );
    }
}

#[test]
fn test_spatial_metadata_preserved() {
    use ritk_core::spatial::{Direction, Point, Spacing};
    let device: <TestBackend as ritk_image::tensor::Backend>::Device = Default::default();
    let td = TensorData::new(vec![100.0_f32; 27], Shape::new([3, 3, 3]));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = Image::new(tensor, origin, spacing, direction);

    let result = connected_threshold(&image, [0, 0, 0], 50.0, 150.0);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
}

// ── 3-D volumetric test ───────────────────────────────────────────────────────

#[test]
fn test_3d_sphere_region_growing() {
    // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
    // background intensity 50; lower=150, upper=255.
    // Region growing from center should capture exactly the sphere.
    let (nz, ny, nx) = (9, 9, 9);
    let mut values = vec![50.0_f32; nz * ny * nx];
    let (cz, cy, cx) = (4isize, 4isize, 4isize);
    let r2 = 9isize; // radius 3

    let mut sphere_count = 0;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as isize - cz;
                let dy = iy as isize - cy;
                let dx = ix as isize - cx;
                if dz * dz + dy * dy + dx * dx <= r2 {
                    values[iz * ny * nx + iy * nx + ix] = 200.0;
                    sphere_count += 1;
                }
            }
        }
    }

    let image = make_image(values, [nz, ny, nx]);
    let result = connected_threshold(&image, [4, 4, 4], 150.0, 255.0);
    assert_eq!(
        count_foreground(&result),
        sphere_count,
        "grown region must match sphere voxel count exactly"
    );
}
