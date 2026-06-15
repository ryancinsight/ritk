use super::super::surface::{
    compute_strides, coords_to_flat, directed_hausdorff, euclidean_distance, flat_to_coords,
    min_distance_to_set,
};
use super::{make_mask_1d, make_mask_2d, make_mask_3d, TestBackend};
use crate::image_comparison::{hausdorff_distance, mean_surface_distance};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

#[test]
fn test_hausdorff_identical_masks_is_zero() {
    let mask = make_mask_2d(vec![1.0f32; 9], [3, 3]);
    let spacing = [1.0f64, 1.0];
    let hd = hausdorff_distance(&mask, &mask, &spacing);
    assert!(
        hd.abs() < super::F32_TOL,
        "identical masks -> HD = 0.0, got {}",
        hd
    );
}

#[test]
fn test_hausdorff_1d_known_value() {
    let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    let spacing = [1.0f64];
    let hd = hausdorff_distance(&pred, &gt, &spacing);
    assert!((hd - 3.0).abs() < 1e-4, "1D HD expected 3.0, got {}", hd);
}

#[test]
fn test_hausdorff_scales_with_spacing() {
    let device: <TestBackend as Backend>::Device = Default::default();
    let pred_t = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0f32, 1.0, 0.0, 0.0, 0.0], Shape::new([5])),
        &device,
    );
    let gt_t = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0], Shape::new([5])),
        &device,
    );
    let pred_img: Image<TestBackend, 1> = Image::new(
        pred_t,
        Point::new([0.0]),
        Spacing::new([2.0]),
        Direction::identity(),
    );
    let gt_img: Image<TestBackend, 1> = Image::new(
        gt_t,
        Point::new([0.0]),
        Spacing::new([2.0]),
        Direction::identity(),
    );
    let spacing = [2.0f64];
    let hd = hausdorff_distance(&pred_img, &gt_img, &spacing);
    assert!(
        (hd - 6.0).abs() < 1e-4,
        "spacing=2.0 -> HD = 6.0, got {}",
        hd
    );
}

#[test]
fn test_hausdorff_symmetry() {
    let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    let spacing = [1.0f64];
    let hd_pg = hausdorff_distance(&pred, &gt, &spacing);
    let hd_gp = hausdorff_distance(&gt, &pred, &spacing);
    assert!(
        (hd_pg - hd_gp).abs() < super::F32_TOL,
        "HD not symmetric: {} vs {}",
        hd_pg,
        hd_gp
    );
}

#[test]
fn test_hausdorff_both_empty_is_zero() {
    let pred = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let gt = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let spacing = [1.0f64, 1.0, 1.0];
    let hd = hausdorff_distance(&pred, &gt, &spacing);
    assert!(
        hd.abs() < super::F32_TOL,
        "both empty -> HD = 0.0, got {}",
        hd
    );
}

#[test]
fn test_msd_identical_masks_is_zero() {
    let mask = make_mask_2d(vec![1.0f32; 9], [3, 3]);
    let spacing = [1.0f64, 1.0];
    let msd = mean_surface_distance(&mask, &mask, &spacing);
    assert!(
        msd.abs() < super::F32_TOL,
        "identical masks -> MSD = 0.0, got {}",
        msd
    );
}

#[test]
fn test_msd_1d_known_value() {
    let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    let spacing = [1.0f64];
    let msd = mean_surface_distance(&pred, &gt, &spacing);
    assert!((msd - 2.5).abs() < 1e-4, "1D MSD expected 2.5, got {}", msd);
}

#[test]
fn test_msd_leq_hausdorff() {
    let pred = make_mask_1d(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    let spacing = [1.0f64];
    let hd = hausdorff_distance(&pred, &gt, &spacing);
    let msd = mean_surface_distance(&pred, &gt, &spacing);
    assert!(
        msd <= hd + super::F32_TOL,
        "MSD ({}) must be <= HD ({})",
        msd,
        hd
    );
}

#[test]
fn test_msd_both_empty_is_zero() {
    let pred = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let gt = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let spacing = [1.0f64, 1.0, 1.0];
    let msd = mean_surface_distance(&pred, &gt, &spacing);
    assert!(
        msd.abs() < super::F32_TOL,
        "both empty -> MSD = 0.0, got {}",
        msd
    );
}

#[test]
fn test_msd_symmetry() {
    let pred = make_mask_1d(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let gt = make_mask_1d(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    let spacing = [1.0f64];
    let msd_pg = mean_surface_distance(&pred, &gt, &spacing);
    let msd_gp = mean_surface_distance(&gt, &pred, &spacing);
    assert!(
        (msd_pg - msd_gp).abs() < super::F32_TOL,
        "MSD is not symmetric: {} vs {}",
        msd_pg,
        msd_gp
    );
}

#[test]
fn test_strides_volumetric() {
    let shape = [2usize, 3, 4];
    let strides = compute_strides(&shape);
    assert_eq!(strides, vec![12, 4, 1]);
}

#[test]
fn test_flat_to_coords_round_trip() {
    let shape = [3usize, 4, 5];
    let strides = compute_strides(&shape);
    let coords = flat_to_coords(37, &strides);
    assert_eq!(coords, vec![1, 3, 2]);
    let back = coords_to_flat(&coords, &strides);
    assert_eq!(back, 37);
}

#[test]
fn test_euclidean_distance_known() {
    let a = vec![0.0f64, 0.0];
    let b = vec![3.0f64, 4.0];
    let d = euclidean_distance(&a, &b);
    assert!((d - 5.0).abs() < 1e-10, "expected 5.0, got {}", d);
}

#[test]
fn test_min_distance_to_empty_set_is_infinity() {
    let p = [1.0f64, 2.0];
    let empty: Vec<[f64; 2]> = vec![];
    let d = min_distance_to_set(&p, &empty);
    assert!(d.is_infinite() && d > 0.0, "empty set -> +inf");
}

#[test]
fn test_directed_hausdorff_empty_from_is_zero() {
    let from: Vec<[f64; 2]> = vec![];
    let to: Vec<[f64; 2]> = vec![[1.0, 0.0], [2.0, 0.0]];
    assert_eq!(directed_hausdorff(&from, &to), 0.0);
}
