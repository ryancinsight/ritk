use super::*;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::make_image;

type B = NdArray<f32>;

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    make_image(data, dims)
}

fn get_labels(image: &Image<B, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Seeds preserved ───────────────────────────────────────────────────────

#[test]
fn test_seed_labels_preserved() {
    // 1×1×5 uniform gradient; seed at index 0 = label 1, seed at index 4 = label 2.
    let gradient = make_image_3d(vec![1.0_f32; 5], [1, 1, 5]);
    let mut markers_data = vec![0.0_f32; 5];
    markers_data[0] = 1.0;
    markers_data[4] = 2.0;
    let markers = make_image_3d(markers_data, [1, 1, 5]);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    let labels = get_labels(&result);

    assert_eq!(labels[0], 1.0, "seed at index 0 must retain label 1");
    assert_eq!(labels[4], 2.0, "seed at index 4 must retain label 2");
}

// ── Two seeds on uniform gradient: watershed boundary in middle ────────────

#[test]
fn test_two_seeds_uniform_gradient_boundary_in_middle() {
    // 1×1×5 uniform gradient; seed label 1 at [0], seed label 2 at [4].
    // Voxels [1,2,3] expand from both ends simultaneously.
    // The middle voxel should become a watershed boundary.
    let gradient = make_image_3d(vec![1.0_f32; 5], [1, 1, 5]);
    let markers = make_image_3d(vec![1.0, 0.0, 0.0, 0.0, 2.0], [1, 1, 5]);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    let labels = get_labels(&result);

    assert_eq!(labels[0], 1.0, "seed 1 preserved");
    assert_eq!(labels[4], 2.0, "seed 2 preserved");
    // Expansion from both ends on uniform gradient: labels[1]=1 (adjacent to 1),
    // labels[3]=2 (adjacent to 2); labels[2] is adjacent to both → boundary 0.
    assert_eq!(
        labels[1], 1.0,
        "voxel 1 adjacent to seed 1 must expand to label 1"
    );
    assert_eq!(
        labels[3], 2.0,
        "voxel 3 adjacent to seed 2 must expand to label 2"
    );
    assert_eq!(
        labels[2], 0.0,
        "middle voxel adjacent to both basins must be watershed boundary"
    );
}

// ── Gradient drives flooding order ────────────────────────────────────────

#[test]
fn test_gradient_drives_flooding_order() {
    // 1×1×6: gradient [0,1,2,2,1,0], seeds at idx 0 (label 1) and idx 5 (label 2).
    // Low gradient regions (near edges) flood first.
    // idx 1: adjacent to seed 0 (label 1), grad=1 → label 1.
    // idx 4: adjacent to seed 5 (label 2), grad=1 → label 2.
    // idx 2: adjacent to idx1 (label 1), grad=2 → label 1.
    // idx 3: adjacent to idx4 (label 2), grad=2 → label 2.
    // Actually idx 2 and idx 3 are queued with grad=2 simultaneously, and they are
    // adjacent to different labels only, so they each get their label cleanly.
    let gradient = make_image_3d(vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0], [1, 1, 6]);
    let markers = make_image_3d(vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0], [1, 1, 6]);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    let labels = get_labels(&result);

    assert_eq!(labels[0], 1.0, "seed preserved");
    assert_eq!(labels[5], 2.0, "seed preserved");
    assert_eq!(
        labels[1], 1.0,
        "voxel 1 must be labeled 1 (adjacent to seed 1 only)"
    );
    assert_eq!(
        labels[4], 2.0,
        "voxel 4 must be labeled 2 (adjacent to seed 2 only)"
    );
    // Inner voxels may form a boundary or be assigned to one basin.
    // Both labels[2] and labels[3] must be non-negative integers.
    assert!(
        labels[2] >= 0.0 && labels[2] == labels[2].floor(),
        "label[2] must be non-negative integer"
    );
    assert!(
        labels[3] >= 0.0 && labels[3] == labels[3].floor(),
        "label[3] must be non-negative integer"
    );
}

// ── Spatial metadata preserved ─────────────────────────────────────────────

#[test]
fn test_spatial_metadata_preserved() {
    let gradient = make_image_3d(vec![1.0_f32; 8], [2, 2, 2]);
    let markers = make_image_3d(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], [2, 2, 2]);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    assert_eq!(result.origin(), gradient.origin());
    assert_eq!(result.spacing(), gradient.spacing());
    assert_eq!(result.direction(), gradient.direction());
}

// ── Output shape matches input ─────────────────────────────────────────────

#[test]
fn test_output_shape_matches_input() {
    let dims = [4, 5, 6];
    let n: usize = dims.iter().product();
    let gradient = make_image_3d(vec![1.0_f32; n], dims);
    let mut markers_data = vec![0.0_f32; n];
    markers_data[0] = 1.0;
    markers_data[n - 1] = 2.0;
    let markers = make_image_3d(markers_data, dims);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    assert_eq!(result.shape(), dims, "output shape must match input shape");
}

// ── All-seeded image: every voxel retains its label ────────────────────────

#[test]
fn test_all_seeded_image_all_labels_preserved() {
    // Every voxel is a seed with label = its index + 1.
    // No unlabeled voxels → output = input.
    let n = 8;
    let gradient = make_image_3d(vec![1.0_f32; n], [2, 2, 2]);
    let markers: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
    let marker_image = make_image_3d(markers.clone(), [2, 2, 2]);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &marker_image)
        .unwrap();
    let labels = get_labels(&result);
    for (i, (&got, &expected)) in labels.iter().zip(markers.iter()).enumerate() {
        assert_eq!(
            got, expected,
            "all-seed image: voxel {} label {:.0} must be {:.0}",
            i, got, expected
        );
    }
}

// ── Shape mismatch returns an error ────────────────────────────────────────

#[test]
fn test_shape_mismatch_errors() {
    let gradient = make_image_3d(vec![1.0_f32; 8], [2, 2, 2]);
    let markers = make_image_3d(vec![1.0_f32; 4], [1, 2, 2]);
    assert_eq!(
        MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap_err()
            .to_string(),
        "gradient and marker shapes must match: [2, 2, 2] vs [1, 2, 2]"
    );
}

#[test]
fn native_legacy_and_all_policy_combinations_are_exact() {
    let dimensions = [1, 3, 5];
    let gradient_values = vec![
        0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
    ];
    let mut marker_values = vec![0.0; 15];
    marker_values[0] = 1.0;
    marker_values[14] = 2.0;
    let legacy_gradient = make_image_3d(gradient_values.clone(), dimensions);
    let legacy_markers = make_image_3d(marker_values.clone(), dimensions);
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let native_gradient = NativeImage::from_flat_on(
        gradient_values,
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap();
    let native_markers = NativeImage::from_flat_on(
        marker_values,
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap();
    for (connectivity, lines) in [
        (FloodConnectivity::Face, WatershedLinePolicy::Mark),
        (FloodConnectivity::Face, WatershedLinePolicy::Omit),
        (FloodConnectivity::Full, WatershedLinePolicy::Mark),
        (FloodConnectivity::Full, WatershedLinePolicy::Omit),
    ] {
        let filter = MarkerControlledWatershed::new()
            .with_connectivity(connectivity)
            .with_watershed_lines(lines);
        assert_eq!(filter.connectivity(), connectivity);
        assert_eq!(filter.watershed_lines(), lines);
        let legacy = filter.apply(&legacy_gradient, &legacy_markers).unwrap();
        let native = filter
            .apply_native(&native_gradient, &native_markers, &SequentialBackend)
            .unwrap();
        assert_eq!(native.data_slice().unwrap(), legacy.data_slice().as_ref());
        assert_eq!(*native.origin(), origin);
        assert_eq!(*native.spacing(), spacing);
        assert_eq!(*native.direction(), direction);
    }
}

#[test]
fn numeric_and_geometry_validation_errors_are_exact() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        let gradient = make_image_3d(vec![0.0, value], [1, 1, 2]);
        let markers = make_image_3d(vec![1.0, 0.0], [1, 1, 2]);
        assert_eq!(
            MarkerControlledWatershed::new().apply(&gradient, &markers).unwrap_err().to_string(),
            format!("marker watershed gradient at flat index 1 must be finite and nonnegative, got {value}")
        );
    }
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        1.5,
        MAX_EXACT_LABEL + 2.0,
    ] {
        let gradient = make_image_3d(vec![0.0, 1.0], [1, 1, 2]);
        let markers = make_image_3d(vec![1.0, value], [1, 1, 2]);
        assert_eq!(
            MarkerControlledWatershed::new().apply(&gradient, &markers).unwrap_err().to_string(),
            format!("marker watershed label at flat index 1 must be a finite nonnegative integer no greater than {MAX_EXACT_LABEL}, got {value}")
        );
    }

    let gradient = NativeImage::from_flat_on(
        vec![0.0, 1.0],
        [1, 1, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    let markers = NativeImage::from_flat_on(
        vec![1.0, 0.0],
        [1, 1, 2],
        Point::new([1.0, 0.0, 0.0]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        MarkerControlledWatershed::new()
            .apply_native(&gradient, &markers, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "gradient and marker origins must match"
    );

    let native_gradient = NativeImage::from_flat_on(
        vec![0.0, f32::NAN],
        [1, 1, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    let native_markers = NativeImage::from_flat_on(
        vec![1.0, 0.0],
        [1, 1, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        MarkerControlledWatershed::new()
            .apply_native(&native_gradient, &native_markers, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "marker watershed gradient at flat index 1 must be finite and nonnegative, got NaN"
    );

    let native_gradient = NativeImage::from_flat_on(
        vec![0.0, 1.0],
        [1, 1, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    let native_markers = NativeImage::from_flat_on(
        vec![1.0, 1.5],
        [1, 1, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        MarkerControlledWatershed::new()
            .apply_native(&native_gradient, &native_markers, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        format!("marker watershed label at flat index 1 must be a finite nonnegative integer no greater than {MAX_EXACT_LABEL}, got 1.5")
    );

    for (spacing, direction, expected) in [
        (
            Spacing::new([2.0, 1.0, 1.0]),
            Direction::identity(),
            "gradient and marker spacing must match",
        ),
        (
            Spacing::new([1.0; 3]),
            Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "gradient and marker directions must match",
        ),
    ] {
        let markers = NativeImage::from_flat_on(
            vec![1.0, 0.0],
            [1, 1, 2],
            Point::new([0.0; 3]),
            spacing,
            direction,
            &SequentialBackend,
        )
        .unwrap();
        assert_eq!(
            MarkerControlledWatershed::new()
                .apply_native(&native_gradient, &markers, &SequentialBackend)
                .unwrap_err()
                .to_string(),
            expected
        );
    }
}

#[test]
fn structural_validation_errors_are_exact() {
    assert_eq!(
        validate_inputs(&[], &[], [1, 0, 2], [1, 0, 2], true, true, true)
            .unwrap_err()
            .to_string(),
        "marker watershed requires nonzero dimensions, got [1, 0, 2]"
    );
    assert_eq!(
        validate_inputs(
            &[],
            &[],
            [usize::MAX, 2, 1],
            [usize::MAX, 2, 1],
            true,
            true,
            true,
        )
        .unwrap_err()
        .to_string(),
        format!(
            "marker watershed shape product overflows usize: [{}, 2, 1]",
            usize::MAX
        )
    );
    assert_eq!(
        validate_inputs(&[0.0], &[1.0, 0.0], [1, 1, 2], [1, 1, 2], true, true, true)
            .unwrap_err()
            .to_string(),
        "marker watershed shape [1, 1, 2] requires 2 samples, got gradient=1 markers=2"
    );
}

// ── Default construction ───────────────────────────────────────────────────

#[test]
fn test_default_construction() {
    let _w = MarkerControlledWatershed::default();
}

// ── No seeds → all zeros (unreachable) ────────────────────────────────────

#[test]
fn test_no_seeds_produces_all_zero_output() {
    let gradient = make_image_3d(vec![1.0_f32; 8], [2, 2, 2]);
    let markers = make_image_3d(vec![0.0_f32; 8], [2, 2, 2]);
    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    let labels = get_labels(&result);
    assert!(
        labels.iter().all(|&v| v == 0.0),
        "no seeds → all labels must be 0, got {:?}",
        labels
    );
}

// ── 3D volumetric: two sphere seeds ───────────────────────────────────────

#[test]
fn test_3d_two_sphere_seeds_produce_two_basins() {
    // 9×9×9 image; seed label 1 at center-left (4,4,2), seed label 2 at center-right (4,4,6).
    // Uniform gradient. Basins should expand and meet in the middle.
    let (nz, ny, nx) = (9, 9, 9);
    let n = nz * ny * nx;
    let gradient = make_image_3d(vec![1.0_f32; n], [nz, ny, nx]);
    let mut markers_data = vec![0.0_f32; n];
    // Seed 1: (4,4,2) = 4*81 + 4*9 + 2 = 324 + 36 + 2 = 362
    markers_data[4 * ny * nx + 4 * nx + 2] = 1.0;
    // Seed 2: (4,4,6) = 4*81 + 4*9 + 6 = 324 + 36 + 6 = 366
    markers_data[4 * ny * nx + 4 * nx + 6] = 2.0;
    let markers = make_image_3d(markers_data, [nz, ny, nx]);

    let result = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .unwrap();
    let labels = get_labels(&result);

    // Both labels must appear in the output.
    let has_label_1 = labels.contains(&1.0);
    let has_label_2 = labels.contains(&2.0);
    assert!(has_label_1, "output must contain label 1");
    assert!(has_label_2, "output must contain label 2");

    // All labels must be non-negative integers.
    for &v in &labels {
        assert!(
            v >= 0.0 && v == v.floor(),
            "label {v} must be non-negative integer"
        );
    }

    // Seed voxels must retain their labels.
    assert_eq!(
        labels[4 * ny * nx + 4 * nx + 2],
        1.0,
        "seed 1 must be preserved"
    );
    assert_eq!(
        labels[4 * ny * nx + 4 * nx + 6],
        2.0,
        "seed 2 must be preserved"
    );
}

/// `mark_watershed_line = false`: collision voxels get a basin label instead of
/// becoming a 0 line, so a two-basin relief has no label-0 voxels.
#[test]
fn test_no_watershed_line_assigns_every_voxel() {
    // W-shaped 1-D relief; markers at the two minima.
    let grad = make_image_3d(vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0], [1, 1, 9]);
    let markers = make_image_3d(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [1, 1, 9]);
    let with_line = MarkerControlledWatershed::new()
        .apply(&grad, &markers)
        .unwrap();
    let without = MarkerControlledWatershed::new()
        .with_watershed_lines(WatershedLinePolicy::Omit)
        .apply(&grad, &markers)
        .unwrap();
    let wl = get_labels(&with_line);
    let wo = get_labels(&without);
    assert!(wl.contains(&0.0), "default marks a watershed line");
    assert!(
        wo.iter().all(|&v| v == 1.0 || v == 2.0),
        "no-line mode labels every voxel into a basin"
    );
}
