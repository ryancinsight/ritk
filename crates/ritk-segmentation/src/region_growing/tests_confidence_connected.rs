//! Tests for confidence_connected
//! Extracted to keep the 500-line structural limit.
#![allow(clippy::identity_op, clippy::erasing_op)]
use super::*;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::tensor::Tensor;
use ritk_image::test_support::make_image;
use ritk_image::Image;

type TestBackend = SequentialBackend;

fn get_values(image: &Image<f32, TestBackend, 3>) -> Vec<f32> {
    image.data().to_vec()
}

fn count_foreground(image: &Image<f32, TestBackend, 3>) -> usize {
    get_values(image).iter().filter(|&&v| v > 0.5).count()
}

// â”€â”€ Positive tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_seed_in_uniform_region_grows_entire_volume() {
    // All voxels have intensity 100; seed intensity 100 within [50, 150].
    // With Î¼=100, Ïƒ=0 initially, then Ïƒ=0 for uniform region,
    // all voxels qualify on first iteration.
    let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
    let result = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result),
        64,
        "uniform region grows entire volume"
    );
}

#[test]
fn test_iterative_update_converges_to_stable_region() {
    // Two-region image: center 2x2x2 = 200, surrounding = 50.
    // Seed in center; initial bounds [150, 255] only capture center.
    // After first iteration, Î¼=200, Ïƒ=0, so same bounds â†’ stable.
    let mut values = vec![50.0_f32; 64]; // 4x4x4
                                         // Center 2x2x2 at [1..3, 1..3, 1..3].
    for z in 1..3 {
        for y in 1..3 {
            for x in 1..3 {
                values[z * 16 + y * 4 + x] = 200.0;
            }
        }
    }
    let image = make_image(values, [4, 4, 4]);
    let result = confidence_connected(&image, [1, 1, 1], 150.0, 255.0, 2.5, 15);
    // Should converge to exactly the 2x2x2 center region (8 voxels).
    assert_eq!(
        count_foreground(&result),
        8,
        "iterative update must converge to stable region"
    );
}

#[test]
fn test_multiplier_affects_region_size() {
    // Gradient sphere: center intensity 200, linear falloff to edge.
    // Larger multiplier should produce larger regions.
    let (nz, ny, nx) = (9, 9, 9);
    let mut values = vec![0.0_f32; nz * ny * nx];
    let (cz, cy, cx) = (4isize, 4isize, 4isize);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as isize - cz;
                let dy = iy as isize - cy;
                let dx = ix as isize - cx;
                let dist_sq = dz * dz + dy * dy + dx * dx;
                // Intensity = 200 - 10*distanceÂ², clipped at 50.
                let intensity = (200.0 - 10.0 * (dist_sq as f32)).max(50.0);
                values[iz * ny * nx + iy * nx + ix] = intensity;
            }
        }
    }
    let image = make_image(values, [nz, ny, nx]);
    let result_small = confidence_connected(&image, [4, 4, 4], 150.0, 250.0, 1.0, 15);
    let result_large = confidence_connected(&image, [4, 4, 4], 150.0, 250.0, 5.0, 15);
    let count_small = count_foreground(&result_small);
    let count_large = count_foreground(&result_large);
    assert!(
        count_large > count_small,
        "larger k={} (count={}) must produce larger region than k={} (count={})",
        5.0,
        count_large,
        1.0,
        count_small
    );
}

#[test]
fn test_max_iteration_limit_respected() {
    // Gradual gradient from center; without limit would grow indefinitely.
    // With max_iterations=1, should stop after first expansion.
    let mut values = vec![100.0_f32; 27]; // 3x3x3
    values[13] = 200.0; // center
    let image = make_image(values, [3, 3, 3]);
    let result = confidence_connected(&image, [1, 1, 1], 150.0, 250.0, 2.5, 1);
    // First iteration: center only (Ïƒ=0). No growth, so just 1 voxel.
    assert_eq!(
        count_foreground(&result),
        1,
        "max_iterations=1 must limit growth"
    );
}

#[test]
fn test_spatial_metadata_preserved() {
    let tensor = Tensor::<f32, TestBackend>::from_slice([3, 3, 3], &[100.0_f32; 27]);
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = Image::new(tensor, origin, spacing, direction)
        .expect("invariant: fixture tensor has the declared rank");
    let result = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 2.5, 15);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
}

#[test]
fn test_binary_output_verification() {
    let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
    let result = confidence_connected(&image, [1, 1, 1], 50.0, 150.0, 2.5, 15);
    for &v in get_values(&result).iter() {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be strictly binary, got {v}"
        );
    }
}

// â”€â”€ 3-D volumetric test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_3d_sphere_region_growing() {
    // 9Ã—9Ã—9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
    // background intensity 50; initial bounds [150, 255].
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
    // Sphere voxels are uniform (200), so once entered, Î¼=200, Ïƒ=0,
    // initial bounds continue to apply, sphere captured completely.
    let result = confidence_connected(&image, [4, 4, 4], 150.0, 255.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result),
        sphere_count,
        "grown region must match sphere voxel count exactly"
    );
}

// â”€â”€ Negative tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_seed_outside_initial_range_returns_empty() {
    // Seed intensity = 5.0, initial range [50, 200] â†’ seed excluded â†’ empty mask.
    let image = make_image(vec![5.0_f32; 8], [2, 2, 2]);
    let result = confidence_connected(&image, [0, 0, 0], 50.0, 200.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result),
        0,
        "seed outside initial range must produce empty region"
    );
}

#[test]
fn test_filter_struct_builder_pattern() {
    let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
    let via_fn = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 3.0, 10);
    let via_struct = ConfidenceConnectedFilter::new([0, 0, 0], 50.0, 150.0)
        .with_multiplier(3.0)
        .expect("test multiplier is valid")
        .with_max_iterations(10)
        .apply(&image);
    let fn_vals = get_values(&via_fn);
    let struct_vals = get_values(&via_struct);
    assert_eq!(
        fn_vals, struct_vals,
        "function and filter struct must produce identical results"
    );
}

// â”€â”€ Adversarial tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Two isolated uniform-intensity cubes separated by zero-intensity background.
/// Seeding from cube A must not bleed to cube B, and vice versa.
///
/// Image: 1Ã—1Ã—8 = [100, 100, 100, 0, 0, 200, 200, 200]
///   A = positions 0..3 (intensity 100), background = 4..5 (intensity 0)
///   B = positions 5..8 (intensity 200)
/// Seed A at [0,0,0] with initial=[50,150]: grows exactly 3 voxels (cube A).
/// Seed B at [0,0,5] with initial=[150,250]: grows exactly 3 voxels (cube B).
#[test]
fn test_multi_seed_two_cubes_no_cross_contamination() {
    let values = vec![100.0_f32, 100.0, 100.0, 0.0, 0.0, 200.0, 200.0, 200.0];
    let image = make_image(values, [1, 1, 8]);

    // Seed in cube A: bounds [50, 150], k=2.5.
    // Iter 1: Ïƒ=0, bounds=[50,150]. Grows x=1 (100) and then x=2 (100).
    // x=3 has intensity 0.0 < 50 â†’ rejected. Result: 3 voxels.
    let result_a = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result_a),
        3,
        "seed in cube A must grow exactly cube A (3 voxels)"
    );
    // Verify cube B voxels remain zero.
    let vals_a = get_values(&result_a);
    assert_eq!(
        vals_a[5], 0.0,
        "cube B voxel 5 must not be reached from seed A"
    );
    assert_eq!(
        vals_a[6], 0.0,
        "cube B voxel 6 must not be reached from seed A"
    );
    assert_eq!(
        vals_a[7], 0.0,
        "cube B voxel 7 must not be reached from seed A"
    );

    // Seed in cube B: bounds [150, 250], k=2.5.
    let result_b = confidence_connected(&image, [0, 0, 5], 150.0, 250.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result_b),
        3,
        "seed in cube B must grow exactly cube B (3 voxels)"
    );
    let vals_b = get_values(&result_b);
    assert_eq!(
        vals_b[0], 0.0,
        "cube A voxel 0 must not be reached from seed B"
    );
    assert_eq!(
        vals_b[1], 0.0,
        "cube A voxel 1 must not be reached from seed B"
    );
    assert_eq!(
        vals_b[2], 0.0,
        "cube A voxel 2 must not be reached from seed B"
    );
}

/// Large multiplier expands the confidence interval to include more voxels
/// in a gradient image than a small multiplier.
///
/// Image: 1Ã—1Ã—3 = [100, 130, 10]. Seed at [0,0,0], initial=[50, 200].
/// First flood [50,200] captures {100, 130} (10 âˆ‰ [50,200]); sample stats over
/// those two are Î¼=115, Ïƒ=âˆš(((100Â²+130Â²) âˆ’ 115Â²Â·2)/(2âˆ’1))=âˆš450â‰ˆ21.2.
/// k=2.0: boundsâ‰ˆ[115âˆ’42, 115+42]=[73, 157]; 10 âˆ‰ â†’ fixed point at 2 voxels.
/// k=10.0: boundsâ‰ˆ[115âˆ’212, 115+212]=[âˆ’97, 327]; re-flood now reaches 10 â†’ 3 voxels.
#[test]
fn test_large_multiplier_expands_region_over_gradient() {
    let values = vec![100.0_f32, 130.0, 10.0];
    let image = make_image(values, [1, 1, 3]);

    let result_small_k = confidence_connected(&image, [0, 0, 0], 50.0, 200.0, 2.0, 15);
    let result_large_k = confidence_connected(&image, [0, 0, 0], 50.0, 200.0, 10.0, 15);

    let count_small = count_foreground(&result_small_k);
    let count_large = count_foreground(&result_large_k);

    // k=10.0 must capture all 3 voxels; k=2.0 must stop at 2.
    assert_eq!(
        count_large, 3,
        "k=10.0 must expand to all 3 voxels (got {count_large})"
    );
    assert_eq!(
        count_small, 2,
        "k=2.0 must stop at 2 voxels (got {count_small})"
    );
    assert!(
        count_large > count_small,
        "large k must always produce region â‰¥ small k"
    );
}

/// Seed placed at volume corner [0,0,0] of a 4Ã—4Ã—4 uniform image must still
/// grow the entire 64-voxel volume via BFS without out-of-bounds access.
#[test]
fn test_seed_at_volume_corner_grows_full_uniform_volume() {
    let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
    let result = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result),
        64,
        "corner seed on uniform 4Ã—4Ã—4 image must grow all 64 voxels"
    );
}

/// `max_iterations = 0` means the iteration loop does not execute at all.
/// The algorithm adds the seed voxel to the output before the loop, so
/// the result must contain exactly 1 foreground voxel (the seed itself).
#[test]
fn test_zero_max_iterations_returns_only_seed_voxel() {
    // 3Ã—3Ã—3 uniform image; seed at center with max_iterations=0.
    let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
    let result = confidence_connected(&image, [1, 1, 1], 50.0, 150.0, 2.5, 0);
    assert_eq!(
        count_foreground(&result),
        1,
        "max_iterations=0 must return only the seed voxel"
    );
    // Verify the seed voxel itself is the foreground voxel.
    let vals = get_values(&result);
    let seed_flat = 1 * 3 * 3 + 1 * 3 + 1;
    assert_eq!(
        vals[seed_flat], 1.0,
        "the single foreground voxel must be the seed voxel"
    );
}

/// Voxels with intensity exactly equal to `initial_lower` or `initial_upper`
/// must be included (inclusive bounds: L â‰¤ I(v) â‰¤ U).
///
/// Image: 1Ã—1Ã—3 = [50, 100, 200]. initial_lower=50, initial_upper=200.
/// Seed at [0,0,0] (value 50 == initial_lower): must be included.
///
/// The first flood uses the full initial interval [50, 200], and 50, 100, 200 are
/// all in it and connected, so the *entire* line is captured in one pass (this is
/// ITK's behaviour: each iteration floods the whole connected region within the
/// current interval). Recomputed stats over {50,100,200} are Î¼=116.67,
/// Ïƒ=âˆš(((2500+10000+40000) âˆ’ Î¼Â²Â·3)/(3âˆ’1))â‰ˆ76.38, so with k=2.5 the interval widens
/// to â‰ˆ[âˆ’74, 308] and the region is a fixed point at all 3 voxels.
///
/// (The earlier 2-voxel expectation came from a defect that advanced one BFS ring
/// per iteration and narrowed the band before voxel 200 was ever tested.)
#[test]
fn test_initial_bound_exact_values_are_inclusive() {
    let values = vec![50.0_f32, 100.0, 200.0];
    let image = make_image(values, [1, 1, 3]);

    let result = confidence_connected(&image, [0, 0, 0], 50.0, 200.0, 2.5, 15);
    let vals = get_values(&result);

    // Seed at exact initial_lower must be included.
    assert_eq!(
        vals[0], 1.0,
        "voxel at exact initial_lower (50.0) must be foreground"
    );
    // Adjacent voxel 100 âˆˆ [50, 200] must be included.
    assert_eq!(vals[1], 1.0, "voxel 100 âˆˆ [50, 200] must be foreground");
    // Voxel 200 == initial_upper is in the first flood's interval, so it is
    // captured in the initial pass and the widened bounds keep it.
    assert_eq!(
        vals[2], 1.0,
        "voxel at exact initial_upper (200.0) âˆˆ [50, 200] must be foreground"
    );

    // Separate test: seed exactly at initial_upper (1Ã—1Ã—1 single voxel).
    let image_upper = make_image(vec![200.0_f32], [1, 1, 1]);
    let result_upper = confidence_connected(&image_upper, [0, 0, 0], 100.0, 200.0, 2.5, 15);
    assert_eq!(
        count_foreground(&result_upper),
        1,
        "single voxel at exact initial_upper (200.0) must be foreground"
    );
}
