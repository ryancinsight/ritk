//! Tests for binary_contour
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 0.0]),
        Spacing::new([1.0_f64, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// All-background → all-zero output.
#[test]
fn all_background_zero() {
    let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    assert!(voxels(&out).iter().all(|&v| v == 0.0));
}

/// All-foreground solid block: only the outer shell is a border.
/// 3×3×3 all-fg: corner/edge/face voxels are borders; center (1,1,1) is interior (6-conn).
#[test]
fn solid_block_all_border() {
    let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    let v = voxels(&out);
    // Center voxel (1,1,1) = index 1*9+1*3+1 = 13: has all 6 face-neighbors in-bounds and fg → interior.
    assert_eq!(v[13], 0.0, "center of 3×3×3 is interior in 6-conn");
    // All other 26 voxels are on the outer shell and neighbor out-of-bounds → borders.
    assert!(
        v.iter()
            .enumerate()
            .filter(|&(i, _)| i != 13)
            .all(|(_, &x)| (x - 1.0).abs() < 1e-5),
        "all non-center voxels of 3×3×3 must be borders"
    );
}

/// 5×5×5 block: all-fg. Center voxel (2,2,2) is interior → 0 (6-connected).
#[test]
fn five_cube_center_is_interior_6conn() {
    let img = make_image(vec![1.0f32; 125], [5, 5, 5]);
    let out = BinaryContourImageFilter::new(Connectivity::Face6, 1.0)
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    // Center voxel index = 2*25+2*5+2 = 62
    assert_eq!(v[62], 0.0, "center of 5×5×5 should be interior (6-conn)");
}

/// Single foreground voxel in a background image is a border voxel.
#[test]
fn single_fg_voxel_is_border() {
    let mut data = vec![0.0f32; 27];
    data[13] = 1.0; // center of 3×3×3
    let img = make_image(data, [3, 3, 3]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    let v = voxels(&out);
    assert!((v[13] - 1.0).abs() < 1e-5, "single fg voxel must be border");
    let others: f32 = v
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != 13)
        .map(|(_, &x)| x)
        .sum();
    assert_eq!(others, 0.0);
}

/// Spatial metadata preserved.
#[test]
fn preserves_metadata() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
    assert_eq!(*out.origin(), *img.origin());
    assert_eq!(*out.spacing(), *img.spacing());
}

/// 26-connected: center of 5×5×5 all-fg block is also interior.
#[test]
fn five_cube_center_interior_26conn() {
    let img = make_image(vec![1.0f32; 125], [5, 5, 5]);
    let out = BinaryContourImageFilter::new(Connectivity::Vertex26, 1.0)
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    assert_eq!(v[62], 0.0, "center of 5×5×5 should be interior (26-conn)");
}
