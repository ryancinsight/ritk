//! Tests for permute_axes
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image_with::<f32, B, 3>(
        vals,
        shape,
        Some(Point::new([0.0, 0.0, 0.0])),
        Some(Spacing::new([1.0, 2.0, 3.0])),
        None,
    )
}

fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

#[test]
fn permute_axes_identity_order_is_noop() {
    let vals: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let img = make_image(vals.clone(), [1, 2, 3]);
    let out = PermuteAxesImageFilter::new([0, 1, 2])
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), [1, 2, 3]);
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert_eq!(a, b, "voxel {}: identity mismatch", i);
    }
}

#[test]
fn permute_axes_transpose_zx_swaps_shape_and_voxels() {
    // Input shape [2, 1, 3], order [2, 1, 0] → output shape [3, 1, 2]
    // vals layout (ZYX): val at (iz,iy,ix) = iz*3 + iy*3 + ix + 1
    let vals: Vec<f32> = (1..=6).map(|x| x as f32).collect(); // 2×1×3
    let img = make_image(vals, [2, 1, 3]);
    let out = PermuteAxesImageFilter::new([2, 1, 0])
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), [3, 1, 2], "output shape after ZX transpose");
    let v = voxels(&out);
    // out[i0][i1][i2] = in[in_idx]; in_idx[order[j]] = ij
    // order=[2,1,0]: in_coords[2]=i0, in_coords[1]=i1, in_coords[0]=i2
    // out[0][0][0] = in[0][0][0] = 1
    // out[0][0][1] = in[1][0][0] = 4
    // out[1][0][0] = in[0][0][1] = 2
    assert_eq!(v[0], 1.0, "out[0][0][0]");
    assert_eq!(v[1], 4.0, "out[0][0][1]");
    assert_eq!(v[2], 2.0, "out[1][0][0]");
}

#[test]
fn permute_axes_spacing_is_permuted() {
    // spacing = [1, 2, 3]; order = [2, 0, 1] → new spacing = [3, 1, 2]
    let img = make_image(vec![0.0; 6], [1, 2, 3]);
    let out = PermuteAxesImageFilter::new([2, 0, 1])
        .apply(&img)
        .expect("infallible: validated precondition");
    let s = out.spacing();
    assert!(
        (s[0] - 3.0).abs() < 1e-9,
        "spacing[0] expected 3, got {}",
        s[0]
    );
    assert!(
        (s[1] - 1.0).abs() < 1e-9,
        "spacing[1] expected 1, got {}",
        s[1]
    );
    assert!(
        (s[2] - 2.0).abs() < 1e-9,
        "spacing[2] expected 2, got {}",
        s[2]
    );
}

#[test]
fn permute_axes_invalid_order_returns_error() {
    let img = make_image(vec![1.0; 8], [2, 2, 2]);
    // Duplicate axis
    let r = PermuteAxesImageFilter::new([0, 0, 1]).apply(&img);
    assert!(r.is_err(), "duplicate axis should return Err");
    // Out-of-range axis
    let r2 = PermuteAxesImageFilter::new([0, 1, 3]).apply(&img);
    assert!(r2.is_err(), "out-of-range axis should return Err");
}

#[test]
fn native_permute_axes_preserves_origin_and_reorders_geometry() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;

    let image = NativeImage::from_flat_on(
        (1..=6).map(|value| value as f32).collect(),
        [2, 1, 3],
        Point::new([5.0, 7.0, 11.0]),
        Spacing::new([1.0, 2.0, 3.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = PermuteAxesImageFilter::new([2, 1, 0])
        .apply_native(&image, &SequentialBackend)
        .expect("native axis permutation succeeds");

    assert_eq!(output.shape(), [3, 1, 2]);
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [5.0, 7.0, 11.0]
    );
    assert_eq!(
        [
            output.spacing()[0],
            output.spacing()[1],
            output.spacing()[2]
        ],
        [3.0, 2.0, 1.0]
    );
}
