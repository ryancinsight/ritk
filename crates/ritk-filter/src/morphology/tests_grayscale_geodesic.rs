//! Tests for grayscale_geodesic
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
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

/// When marker equals mask, reconstruction by dilation returns marker unchanged.
#[test]
fn geodesic_dilation_marker_equals_mask_is_identity() {
    let vals = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let marker = make_image(vals.clone(), [2, 2, 2]);
    let mask = make_image(vals.clone(), [2, 2, 2]);
    let out = GrayscaleGeodesicDilationFilter::new()
        .apply(&marker, &mask)
        .unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "voxel {}: expected {}, got {}",
            i,
            b,
            a
        );
    }
}

/// Marker ≤ mask: reconstruction expands marker but never exceeds mask.
#[test]
fn geodesic_dilation_result_bounded_by_mask() {
    // 1×1×5: marker=[0,0,5,0,0], mask=[3,3,5,3,3]
    let marker = make_image(vec![0.0f32, 0.0, 5.0, 0.0, 0.0], [1, 1, 5]);
    let mask = make_image(vec![3.0f32, 3.0, 5.0, 3.0, 3.0], [1, 1, 5]);
    let out = GrayscaleGeodesicDilationFilter::new()
        .apply(&marker, &mask)
        .unwrap();
    let v = voxels(&out);
    let mv = [3.0f32, 3.0, 5.0, 3.0, 3.0];
    for (i, (&a, &b)) in v.iter().zip(mv.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "voxel {}: expected {}, got {}",
            i,
            b,
            a
        );
    }
}

/// Spatial metadata is preserved identically.
#[test]
fn geodesic_dilation_preserves_spatial_metadata() {
    let vals = vec![1.0f32; 8];
    let marker = make_image(vals.clone(), [2, 2, 2]);
    let mask = make_image(vals, [2, 2, 2]);
    let out = GrayscaleGeodesicDilationFilter::new()
        .apply(&marker, &mask)
        .unwrap();
    assert_eq!(out.shape(), marker.shape());
    assert_eq!(out.spacing(), marker.spacing());
    assert_eq!(out.origin(), marker.origin());
}

/// When marker equals mask, reconstruction by erosion returns marker unchanged.
#[test]
fn geodesic_erosion_marker_equals_mask_is_identity() {
    let vals = vec![7.0f32, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let marker = make_image(vals.clone(), [2, 2, 2]);
    let mask = make_image(vals.clone(), [2, 2, 2]);
    let out = GrayscaleGeodesicErosionFilter::new()
        .apply(&marker, &mask)
        .unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "voxel {}: expected {}, got {}",
            i,
            b,
            a
        );
    }
}

/// Face vs full connectivity differ on diagonal-only propagation.
///
/// On a 3×3 checkerboard mask, the corner voxels (value 5) are reachable from
/// the centre seed only through diagonal (vertex) steps; the edge voxels
/// between them are 0 and block every axis-aligned path. Face connectivity
/// (ITK default) therefore leaves the corners at 0, while full connectivity
/// reconstructs them to 5.
#[test]
fn geodesic_dilation_connectivity_controls_diagonal_propagation() {
    // 1×3×3 checkerboard mask; marker = centre seed only.
    let mask_vals = vec![5.0f32, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0];
    let mut marker_vals = vec![0.0f32; 9];
    marker_vals[4] = 5.0; // centre
    let marker = make_image(marker_vals, [1, 3, 3]);
    let mask = make_image(mask_vals.clone(), [1, 3, 3]);

    let face = voxels(
        &GrayscaleGeodesicDilationFilter::new()
            .with_connectivity(Connectivity::Face6)
            .apply(&marker, &mask)
            .unwrap(),
    );
    let full = voxels(
        &GrayscaleGeodesicDilationFilter::new()
            .with_connectivity(Connectivity::Vertex26)
            .apply(&marker, &mask)
            .unwrap(),
    );

    // Face: only the centre carries the marker; corners stay 0.
    let expect_face = [0.0f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    // Full: centre plus the four diagonal corners reach 5.
    let expect_full = [5.0f32, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0];
    for (i, (&f, &e)) in face.iter().zip(expect_face.iter()).enumerate() {
        assert!((f - e).abs() < 1e-4, "face voxel {i}: got {f}, want {e}");
    }
    for (i, (&f, &e)) in full.iter().zip(expect_full.iter()).enumerate() {
        assert!((f - e).abs() < 1e-4, "full voxel {i}: got {f}, want {e}");
    }
    // Default (no with_connectivity) must equal face connectivity (ITK default).
    let default = voxels(
        &GrayscaleGeodesicDilationFilter::new()
            .apply(&marker, &mask)
            .unwrap(),
    );
    assert_eq!(
        default, face,
        "default connectivity must be face (ITK default)"
    );
}

/// Marker ≥ mask: reconstruction contracts marker but never goes below mask.
#[test]
fn geodesic_erosion_result_bounded_below_by_mask() {
    // 1×1×5: marker=[5,5,0,5,5], mask=[3,3,0,3,3]; result must ≥ mask
    let marker = make_image(vec![5.0f32, 5.0, 0.0, 5.0, 5.0], [1, 1, 5]);
    let mask = make_image(vec![3.0f32, 3.0, 0.0, 3.0, 3.0], [1, 1, 5]);
    let out = GrayscaleGeodesicErosionFilter::new()
        .apply(&marker, &mask)
        .unwrap();
    let v = voxels(&out);
    let mask_v = [3.0f32, 3.0, 0.0, 3.0, 3.0];
    for (i, (&a, &b)) in v.iter().zip(mask_v.iter()).enumerate() {
        assert!(a >= b - 1e-4, "voxel {}: result {} below mask {}", i, a, b);
    }
}
