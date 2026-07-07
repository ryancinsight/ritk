use super::*;
use burn_ndarray::NdArray;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::tensor::{Shape, Tensor, TensorData};

type B = NdArray<f32>;

fn make_label_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn flat(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// Single component, no size threshold → relabeled as 1, count preserved.
#[test]
fn single_component_identity() {
    // 2×1×1 image: both voxels are component 1.
    let img = make_label_image(vec![1.0, 1.0], [2, 1, 1]);
    let (out, stats) = RelabelComponentFilter::new().apply(&img);
    assert_eq!(flat(&out), vec![1.0, 1.0]);
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].new_label, 1);
    assert_eq!(stats[0].voxel_count, 2);
    assert_eq!(stats[0].original_label, 1);
}

/// Three components with distinct sizes → sorted by descending count.
///
/// Input labels and voxel counts: {1:5, 2:15, 3:3}.
/// Expected new labels: 2→1 (15), 1→2 (5), 3→3 (3).
#[test]
fn three_components_sorted_descending() {
    // 1×1×23 flat image: label 1 appears 5×, label 2 appears 15×, label 3 appears 3×.
    let mut vals = vec![1.0_f32; 5];
    vals.extend(vec![2.0_f32; 15]);
    vals.extend(vec![3.0_f32; 3]);
    let n = vals.len();
    let img = make_label_image(vals.clone(), [1, 1, n]);

    let (out, stats) = RelabelComponentFilter::new().apply(&img);
    let out_flat = flat(&out);

    // stats should be sorted by descending count: 15, 5, 3.
    assert_eq!(stats.len(), 3);
    assert_eq!(stats[0].new_label, 1);
    assert_eq!(stats[0].voxel_count, 15);
    assert_eq!(stats[0].original_label, 2);
    assert_eq!(stats[1].new_label, 2);
    assert_eq!(stats[1].voxel_count, 5);
    assert_eq!(stats[1].original_label, 1);
    assert_eq!(stats[2].new_label, 3);
    assert_eq!(stats[2].voxel_count, 3);
    assert_eq!(stats[2].original_label, 3);

    // Voxels that were 2 should now be 1, 1→2, 3→3.
    let expected: Vec<f32> = vals
        .iter()
        .map(|&v| match v as u32 {
            2 => 1.0,
            1 => 2.0,
            3 => 3.0,
            _ => 0.0,
        })
        .collect();
    assert_eq!(out_flat, expected);
}

/// `minimum_object_size` removes components below threshold.
///
/// Components: {1: 3 voxels, 2: 10 voxels}. Threshold = 5.
/// Expected: component 1 removed (→0), component 2 relabeled to 1.
#[test]
fn minimum_object_size_removes_small() {
    let mut vals = vec![1.0_f32; 3]; // label 1, count=3 (small)
    vals.extend(vec![2.0_f32; 10]); // label 2, count=10 (large)
    let n = vals.len();
    let img = make_label_image(vals, [1, 1, n]);

    let (out, stats) = RelabelComponentFilter::with_minimum_object_size(5).apply(&img);
    let out_flat = flat(&out);

    // Only the large component survives as label 1.
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].original_label, 2);
    assert_eq!(stats[0].new_label, 1);
    assert_eq!(stats[0].voxel_count, 10);

    // First 3 voxels (label 1) → 0; last 10 voxels (label 2) → 1.
    let mut expected = vec![0.0_f32; 3];
    expected.extend(vec![1.0_f32; 10]);
    assert_eq!(out_flat, expected);
}

/// All components below minimum_object_size → all-zero output.
#[test]
fn all_below_threshold_gives_all_zero() {
    let vals: Vec<f32> = (1..=4).map(|v| v as f32).collect(); // labels 1,2,3,4 each with 1 voxel
    let img = make_label_image(vals, [1, 1, 4]);

    let (out, stats) = RelabelComponentFilter::with_minimum_object_size(2).apply(&img);

    assert!(stats.is_empty());
    assert!(flat(&out).iter().all(|&v| v == 0.0));
}

/// Background voxels (0.0) are preserved as 0.0 after relabeling.
#[test]
fn background_preserved() {
    // Pattern: bg, comp1, bg, comp1, bg
    let vals = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let img = make_label_image(vals, [1, 1, 5]);

    let (out, stats) = RelabelComponentFilter::new().apply(&img);
    let out_flat = flat(&out);

    assert_eq!(stats.len(), 1);
    assert_eq!(out_flat, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
}

/// All-background input produces all-zero output with empty statistics.
#[test]
fn all_background_produces_empty_stats() {
    let img = make_label_image(vec![0.0, 0.0, 0.0], [1, 1, 3]);
    let (out, stats) = RelabelComponentFilter::new().apply(&img);
    assert!(stats.is_empty());
    assert_eq!(flat(&out), vec![0.0, 0.0, 0.0]);
}

/// Spatial metadata (origin, spacing, direction) is preserved unchanged.
#[test]
fn spatial_metadata_preserved() {
    use ritk_core::spatial::Direction;
    let device = Default::default();
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.75, 1.25]);
    let direction = Direction::identity();

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let img = Image::new(tensor, origin, spacing, direction);

    let (out, _) = RelabelComponentFilter::new().apply(&img);

    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}

/// Two equal-size components → sorted by original label ascending (tie-break).
///
/// Both label 1 and label 2 have 4 voxels.
/// Tie-break by ascending label: 1 → new 1, 2 → new 2.
#[test]
fn equal_size_tiebreak_by_label() {
    let vals = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    let img = make_label_image(vals, [1, 1, 8]);

    let (out, stats) = RelabelComponentFilter::new().apply(&img);

    // Tie-break: label 1 comes first (ascending original label).
    assert_eq!(stats[0].original_label, 1);
    assert_eq!(stats[0].new_label, 1);
    assert_eq!(stats[1].original_label, 2);
    assert_eq!(stats[1].new_label, 2);
    assert_eq!(flat(&out), vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
}

/// minimum_object_size = 1 is equivalent to no filtering (retain all components).
///
/// This verifies the boundary condition: min_size=1 means "at least 1 voxel",
/// which matches all non-background labels by definition.
#[test]
fn minimum_object_size_one_retains_all() {
    let vals = vec![1.0, 2.0, 3.0]; // three single-voxel components
    let img = make_label_image(vals, [1, 1, 3]);

    let (_, stats) = RelabelComponentFilter::with_minimum_object_size(1).apply(&img);
    assert_eq!(stats.len(), 3);
    assert!(stats.iter().all(|s| s.new_label > 0));
}

#[test]
fn relabel_consecutive_ascending_value_order() {
    use super::relabel_consecutive;
    use ritk_image::test_support as ts;
    use ritk_tensor_ops::extract_vec_infallible;
    type B = burn_ndarray::NdArray<f32>;
    // labels 2,5,7,9 -> 1,2,3,4 (ascending original value)
    let data = vec![0.0, 5.0, 2.0, 9.0, 7.0, 0.0, 5.0, 2.0];
    let img = ts::make_image::<B, 3>(data, [1, 1, 8]);
    let out = relabel_consecutive(&img);
    let (v, _) = extract_vec_infallible(&out);
    assert_eq!(v, vec![0.0, 2.0, 1.0, 4.0, 3.0, 0.0, 2.0, 1.0]);
}
