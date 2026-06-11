use super::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

fn make_mask_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    let n = data.len();
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
    Image::new(
        tensor,
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    )
}

fn make_mask_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image.data_slice().into_owned()
}

fn values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image.data_slice().into_owned()
}

fn count_fg_3d(image: &Image<TestBackend, 3>) -> usize {
    values_3d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── radius = 0 is identity ─────────────────────────────────────────────────

#[test]
fn test_radius0_is_identity_3d() {
    // Structuring element {p} → output = input for any binary mask.
    let data: Vec<f32> = (0u8..27)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mask = make_mask_3d(data.clone(), [3, 3, 3]);
    let result = BinaryDilation::new(0).apply(&mask);
    assert_eq!(
        values_3d(&result),
        data,
        "radius=0 dilation must be identity"
    );
}

#[test]
fn test_radius0_is_identity_1d() {
    let data = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let mask = make_mask_1d(data.clone());
    let result = BinaryDilation::new(0).apply(&mask);
    assert_eq!(
        values_1d(&result),
        data,
        "radius=0 dilation must be identity"
    );
}

// ── Single isolated voxel grows to neighbourhood ──────────────────────────

#[test]
fn test_single_center_voxel_5x5x5_dilates_to_box_r1() {
    // Center voxel (2,2,2) in 5×5×5: dilation r=1 → 3×3×3 box = 27 voxels.
    let mut values = vec![0.0_f32; 125];
    values[2 * 25 + 2 * 5 + 2] = 1.0; // index of (2,2,2) in [5,5,5]
    let mask = make_mask_3d(values, [5, 5, 5]);
    let result = BinaryDilation::new(1).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
        27,
        "single center voxel r=1 dilation must produce 3×3×3 = 27 foreground voxels"
    );
}

#[test]
fn test_single_corner_voxel_3x3x3_dilates_to_corner_box() {
    // Corner voxel (0,0,0) in 3×3×3: dilation r=1 clips to 2×2×2 = 8 voxels
    // (the full 3×3×3 neighbour box exceeds the image boundary on 3 sides).
    let mut values = vec![0.0_f32; 27];
    values[0] = 1.0;
    let mask = make_mask_3d(values, [3, 3, 3]);
    let result = BinaryDilation::new(1).apply(&mask);
    // The r=1 box around (0,0,0) in a [3,3,3] image is [0..=1, 0..=1, 0..=1] = 8 voxels.
    assert_eq!(
        count_fg_3d(&result),
        8,
        "corner voxel r=1 dilation must cover 2×2×2 = 8 voxels"
    );
}

// ── Extensivity invariant: input ⊆ dilated ────────────────────────────────

#[test]
fn test_dilation_is_extensive() {
    // Every foreground voxel in the input must remain foreground after dilation.
    let values: Vec<f32> = (0u8..27)
        .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mask = make_mask_3d(values.clone(), [3, 3, 3]);
    let result = BinaryDilation::new(1).apply(&mask);
    let result_vals = values_3d(&result);
    for (i, (&orig, &out)) in values.iter().zip(result_vals.iter()).enumerate() {
        if orig > 0.5 {
            assert!(
                out > 0.5,
                "dilation removed foreground voxel at index {}",
                i
            );
        }
    }
}

// ── All-foreground input stays all-foreground ─────────────────────────────

#[test]
fn test_all_foreground_stays_all_foreground() {
    // Dilating a fully-foreground mask changes nothing.
    let mask = make_mask_3d(vec![1.0_f32; 27], [3, 3, 3]);
    let result = BinaryDilation::new(1).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
        27,
        "all-foreground mask must remain fully foreground after dilation"
    );
}

// ── All-background stays all-background ───────────────────────────────────

#[test]
fn test_all_background_stays_all_background() {
    let mask = make_mask_3d(vec![0.0_f32; 27], [3, 3, 3]);
    let result = BinaryDilation::new(1).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
        0,
        "all-background mask must remain all-background after dilation"
    );
}

// ── 1D dilation: known cases ──────────────────────────────────────────────

#[test]
fn test_1d_dilation_r1_known_output() {
    // Input: [0, 0, 1, 0, 0]
    // r=1 neighbourhood:
    //   i=0: neighbours {0,1}        → bg → 0
    //   i=1: neighbours {0,1,2}      → i=2 is fg → 1
    //   i=2: neighbours {1,2,3}      → i=2 is fg → 1
    //   i=3: neighbours {2,3,4}      → i=2 is fg → 1
    //   i=4: neighbours {3,4}        → bg → 0
    // Expected: [0, 1, 1, 1, 0]
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let mask = make_mask_1d(data);
    let result = BinaryDilation::new(1).apply(&mask);
    let out = values_1d(&result);
    let expected = vec![0.0, 1.0, 1.0, 1.0, 0.0];
    assert_eq!(out, expected, "1D r=1 dilation output mismatch");
}

#[test]
fn test_1d_dilation_r2_known_output() {
    // Input: [0, 0, 0, 1, 0, 0, 0]
    // r=2: all voxels within distance 2 of index 3 are set.
    // Expected: [0, 1, 1, 1, 1, 1, 0]
    let data = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let mask = make_mask_1d(data);
    let result = BinaryDilation::new(2).apply(&mask);
    let out = values_1d(&result);
    let expected = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    assert_eq!(out, expected, "1D r=2 dilation output mismatch");
}

#[test]
fn test_1d_dilation_single_voxel_at_boundary() {
    // Single voxel at index 0: r=1 dilates to indices {0, 1}.
    let data = vec![1.0, 0.0, 0.0, 0.0];
    let mask = make_mask_1d(data);
    let result = BinaryDilation::new(1).apply(&mask);
    let out = values_1d(&result);
    let expected = vec![1.0, 1.0, 0.0, 0.0];
    assert_eq!(out, expected, "boundary single voxel dilation mismatch");
}

// ── Double dilation is superset of single dilation ────────────────────────

#[test]
fn test_double_dilation_superset_of_single_dilation() {
    // D(D(M)) ⊇ D(M) (monotone).
    let mut values = vec![0.0_f32; 125];
    values[62] = 1.0; // center of 5×5×5
    let mask = make_mask_3d(values, [5, 5, 5]);
    let once = BinaryDilation::new(1).apply(&mask);
    let twice = BinaryDilation::new(1).apply(&once);
    let once_vals = values_3d(&once);
    let twice_vals = values_3d(&twice);
    for (i, (&once_v, &twice_v)) in once_vals.iter().zip(twice_vals.iter()).enumerate() {
        if once_v > 0.5 {
            assert!(
                twice_v > 0.5,
                "double dilation removed voxel at index {} that was present after single dilation",
                i
            );
        }
    }
}

// ── Output strictly binary ────────────────────────────────────────────────

#[test]
fn test_output_strictly_binary_3d() {
    let values: Vec<f32> = (0u8..27)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mask = make_mask_3d(values, [3, 3, 3]);
    let result = BinaryDilation::new(1).apply(&mask);
    for &v in values_3d(&result).iter() {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be strictly binary, got {v}"
        );
    }
}

// ── Metadata preservation ─────────────────────────────────────────────────

#[test]
fn test_preserves_spatial_metadata() {
    let device: <TestBackend as Backend>::Device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; 27], Shape::new([3, 3, 3])),
        &device,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let mask: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);
    let result = BinaryDilation::new(1).apply(&mask);
    assert_eq!(result.origin(), &origin, "origin must be preserved");
    assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(
        result.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
}

// ── Dilation then erosion (closing): foreground grows then shrinks ────────

#[test]
fn test_dilation_increases_or_preserves_foreground_count() {
    // Dilation must never decrease the number of foreground voxels.
    let values: Vec<f32> = (0u8..27)
        .map(|i| if i % 4 == 0 { 1.0 } else { 0.0 })
        .collect();
    let orig_count = values.iter().filter(|&&v| v > 0.5).count();
    let mask = make_mask_3d(values, [3, 3, 3]);
    let result = BinaryDilation::new(1).apply(&mask);
    let result_count = count_fg_3d(&result);
    assert!(
        result_count >= orig_count,
        "dilation must not decrease foreground count: before={} after={}",
        orig_count,
        result_count
    );
}
