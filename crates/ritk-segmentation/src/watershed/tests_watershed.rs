use super::*;
use burn_ndarray::NdArray;
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

// ── Constant / uniform image ───────────────────────────────────────────────

#[test]
fn test_constant_image_single_basin() {
    // All voxels have the same intensity → processed in index order.
    // The first voxel gets label 1; every subsequent voxel is 6-adjacent
    // to an already-labelled voxel with label 1 → all get label 1.
    let dims = [3, 3, 3];
    let n: usize = dims.iter().product();
    let data = vec![5.0_f32; n];
    let image = make_image_3d(data, dims);
    let result = WatershedSegmentation::new().apply(&image).unwrap();
    let labels = get_labels(&result);

    // Every voxel should have the same non-zero label.
    assert!(
        labels.iter().all(|&v| v == 1.0),
        "constant image must produce a single basin (label 1), got labels: {:?}",
        labels
    );
}

// ── Two separated minima → two basins + watershed boundary ─────────────────

#[test]
fn test_two_minima_produce_two_basins_with_boundary() {
    // 1×1×5 image: [0, 10, 100, 10, 0]
    // Two local minima at index 0 and 4. The ridge voxel at index 2
    // (intensity 100) should become a watershed boundary (label 0) because
    // when it is processed last, both basin labels are among its neighbours.
    let dims = [1, 1, 5];
    let data = vec![0.0, 10.0, 100.0, 10.0, 0.0];
    let image = make_image_3d(data, dims);
    let result = WatershedSegmentation::new().apply(&image).unwrap();
    let labels = get_labels(&result);

    // Voxels 0 and 4 have the lowest intensities; they get distinct labels.
    assert!(labels[0] > 0.0, "minimum voxel must have a basin label");
    assert!(labels[4] > 0.0, "minimum voxel must have a basin label");
    assert!(
        (labels[0] - labels[4]).abs() > f32::EPSILON,
        "two separated minima must get distinct labels: {} vs {}",
        labels[0],
        labels[4]
    );

    // The ridge voxel (index 2) should be a watershed boundary.
    assert!(
        labels[2] == 0.0,
        "ridge voxel between two basins must be watershed (0), got {}",
        labels[2]
    );
}

// ── Output shape matches input shape ───────────────────────────────────────

#[test]
fn test_output_shape_matches_input() {
    let dims = [4, 5, 6];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 10.0).collect();
    let image = make_image_3d(data, dims);
    let result = WatershedSegmentation::new().apply(&image).unwrap();
    assert_eq!(result.shape(), dims, "output shape must match input shape");
}

// ── Spatial metadata preserved ─────────────────────────────────────────────

#[test]
fn test_spatial_metadata_preserved() {
    let dims = [2, 2, 2];
    let data = vec![0.0_f32; 8];
    let image = make_image_3d(data, dims);
    let result = WatershedSegmentation::new().apply(&image).unwrap();

    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Labels are non-negative integers ───────────────────────────────────────

#[test]
fn test_labels_are_nonneg_integers() {
    let dims = [3, 3, 3];
    let n: usize = dims.iter().product();
    // Gradient-like image with a saddle to force watershed boundaries.
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let z = i / 9;
            let y = (i % 9) / 3;
            let x = i % 3;
            // Two minima at corners (0,0,0) and (2,2,2); ridge in between.
            let d0 = ((z * z + y * y + x * x) as f32).sqrt();
            let d1 = (((2 - z) * (2 - z) + (2 - y) * (2 - y) + (2 - x) * (2 - x)) as f32).sqrt();
            d0.min(d1) * 50.0
        })
        .collect();
    let image = make_image_3d(data, dims);
    let result = WatershedSegmentation::new().apply(&image).unwrap();
    let labels = get_labels(&result);

    for (i, &v) in labels.iter().enumerate() {
        assert!(
            v >= 0.0 && v == v.floor(),
            "label at voxel {} must be a non-negative integer, got {}",
            i,
            v
        );
    }
}

// ── Default delegates to new ───────────────────────────────────────────────

#[test]
fn test_default_construction() {
    let _ws = WatershedSegmentation::default();
}

// ── Single voxel → single basin ────────────────────────────────────────────

#[test]
fn test_single_voxel_single_basin() {
    let dims = [1, 1, 1];
    let data = vec![42.0_f32];
    let image = make_image_3d(data, dims);
    let result = WatershedSegmentation::new().apply(&image).unwrap();
    let labels = get_labels(&result);
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0], 1.0, "single voxel must be labelled 1");
}
