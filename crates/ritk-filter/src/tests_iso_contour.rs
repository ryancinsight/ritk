use super::IsoContourDistanceFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// On a unit-gradient ramp `I(x) = x` with `level = 3.5`, the iso-contour lies
/// between x=3 (val −0.5) and x=4 (val +0.5). The averaged unit gradient gives
/// scale 1, so those two voxels receive their exact signed distance (−0.5, 0.5);
/// every other voxel keeps ±far_value.
#[test]
fn iso_contour_unit_ramp_exact_distance() {
    let nx = 8usize;
    let vals: Vec<f32> = (0..nx).map(|x| x as f32).collect();
    let out = IsoContourDistanceFilter::new(3.5, 10.0).apply(&make(vals, [1, 1, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    let want = [-10.0f32, -10.0, -10.0, -0.5, 0.5, 10.0, 10.0, 10.0];
    for (i, (&got, &w)) in ov.iter().zip(want.iter()).enumerate() {
        assert!((got - w).abs() < 1e-6, "voxel {i}: got {got}, want {w}");
    }
}

/// A constant image has no iso-contour, so the whole field is the far value with
/// the constant's sign relative to the level.
#[test]
fn iso_contour_constant_is_far_value() {
    let dims = [1usize, 4, 5];
    let n: usize = dims.iter().product();
    let out = IsoContourDistanceFilter::new(0.0, 7.0).apply(&make(vec![3.0; n], dims));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(ov.iter().all(|&v| v == 7.0), "above-level constant → +far");
}

/// Output geometry equals input geometry.
#[test]
fn iso_contour_preserves_geometry() {
    let dims = [1usize, 5, 6];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.5).collect();
    let img = make(vals, dims);
    let out = IsoContourDistanceFilter::default().apply(&img);
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
