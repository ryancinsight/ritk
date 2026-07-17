//! Tests for labeling
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support::burn_compat::make_image;

type TestBackend = NdArray<f32>;

fn make_mask(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
    make_image(values, shape)
}

fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

#[test]
fn test_single_component_6_connectivity() {
    // A 3×3×3 all-foreground cube: one component.
    let mask = make_mask(vec![1.0_f32; 27], [3, 3, 3]);
    let (_, num) = connected_components(&mask, Connectivity::Six);
    assert_eq!(num, 1, "solid cube must be a single component");
}

#[test]
fn test_two_separated_components_6_connectivity() {
    // 1×1×4 volume: two foreground voxels separated by a background gap.
    // Indices: [1,0,0,1] along X.
    let values = vec![1.0, 0.0, 0.0, 1.0];
    let mask = make_mask(values, [1, 1, 4]);
    let (_, num) = connected_components(&mask, Connectivity::Six);
    assert_eq!(num, 2, "two separated voxels must be two components");
}

#[test]
fn test_two_components_connected_by_diagonal_6_not_connected() {
    // In a 3×3×1 slice two diagonal foreground voxels are NOT connected
    // under 6-connectivity but ARE connected under 26-connectivity.
    // Layout (z=0):
    //   1 0 0
    //   0 1 0
    //   0 0 0
    let mut values = vec![0.0_f32; 9];
    values[0] = 1.0; // (0,0,0)
    values[4] = 1.0; // (0,1,1)
    let mask = make_mask(values, [1, 3, 3]);
    let (_, num6) = connected_components(&mask, Connectivity::Six);
    let (_, num26) = connected_components(&mask, Connectivity::TwentySix);
    assert_eq!(num6, 2, "diagonal voxels must be 2 components under 6-conn");
    assert_eq!(
        num26, 1,
        "diagonal voxels must be 1 component under 26-conn"
    );
}

#[test]
fn test_empty_mask_returns_zero_components() {
    let mask = make_mask(vec![0.0_f32; 8], [2, 2, 2]);
    let (_, num) = connected_components(&mask, Connectivity::Six);
    assert_eq!(num, 0);
}

#[test]
fn test_label_values_are_consecutive_integers() {
    // Two components: half-split along Z.
    let mut values = [0.0_f32; 16];
    // First 8 voxels form component 1, last 8 form component 2.
    for v in values.iter_mut().take(8) {
        *v = 1.0;
    }
    for v in values.iter_mut().skip(8) {
        *v = 1.0;
    }
    // Make a gap at z=1 boundary: set z=1 layer to 0 for a 4×2×2 split.
    // Actually just use 1×1×8 with gap in middle.
    let gap_values = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mask = make_mask(gap_values, [1, 1, 8]);
    let (label_img, num) = connected_components(&mask, Connectivity::Six);
    assert_eq!(num, 2);
    let lbls = get_values(&label_img);
    let unique: std::collections::HashSet<u32> = lbls
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| v as u32)
        .collect();
    assert_eq!(unique.len(), 2);
    assert!(unique.contains(&1) && unique.contains(&2));
}

#[test]
fn test_statistics_voxel_count() {
    // Two components: sizes 4 and 2.
    let values = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let mask = make_mask(values, [1, 1, 7]);
    let filter = ConnectedComponentsFilter::with_connectivity(Connectivity::Six);
    let (_, stats) = filter.apply(&mask);
    assert_eq!(stats.len(), 2);
    let total: usize = stats.iter().map(|s| s.voxel_count).sum();
    assert_eq!(total, 6, "total labelled voxels must equal 6");
    // Sizes must be {4, 2} in some order.
    let mut sizes: Vec<usize> = stats.iter().map(|s| s.voxel_count).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![2, 4]);
}

#[test]
fn test_statistics_centroid_single_voxel() {
    // Single foreground voxel at (2, 3, 4) in a 5×5×5 image.
    let mut values = vec![0.0_f32; 125];
    let flat = 2 * 25 + 3 * 5 + 4;
    values[flat] = 1.0;
    let mask = make_mask(values, [5, 5, 5]);
    let filter = ConnectedComponentsFilter::with_connectivity(Connectivity::Six);
    let (_, stats) = filter.apply(&mask);
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].centroid, Point::new([2.0, 3.0, 4.0]));
}

#[test]
fn test_statistics_bounding_box() {
    // A 3×3×3 cube of foreground in a 5×5×5 background.
    // Cube occupies z ∈ [1,3], y ∈ [1,3], x ∈ [1,3].
    let mut values = vec![0.0_f32; 125];
    for iz in 1..4 {
        for iy in 1..4 {
            for ix in 1..4 {
                values[iz * 25 + iy * 5 + ix] = 1.0;
            }
        }
    }
    let mask = make_mask(values, [5, 5, 5]);
    let filter = ConnectedComponentsFilter::with_connectivity(Connectivity::Six);
    let (_, stats) = filter.apply(&mask);
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].bounding_box.0, [1, 1, 1], "min corner");
    assert_eq!(stats[0].bounding_box.1, [3, 3, 3], "max corner");
}

#[test]
fn test_metadata_preserved() {
    use ritk_core::spatial::{Direction, Point, Spacing};
    let device = Default::default();
    let values = vec![1.0_f32; 8];
    let td = TensorData::new(values, Shape::new([2, 2, 2]));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    let origin = Point::new([5.0, 6.0, 7.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let mask = Image::new(tensor, origin, spacing, direction);

    let (label_img, _) = connected_components(&mask, Connectivity::Six);
    assert_eq!(label_img.origin(), &origin);
    assert_eq!(label_img.spacing(), &spacing);
}
