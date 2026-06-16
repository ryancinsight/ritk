//! Tests for flip
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

/// No-flip is identity.
#[test]
fn flip_none_is_identity() {
    let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let out = FlipImageFilter::new([FlipPolicy::Keep; 3])
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "[{}] expected {}, got {}", i, b, a);
    }
}

/// Flip X on a 1×1×4 array reverses the sequence.
#[test]
fn flip_x_reverses_x_axis() {
    let img = make_image(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 4]);
    let out = FlipImageFilter::flip_x().apply(&img).unwrap();
    let v = voxels(&out);
    let expected = [4.0f32, 3.0, 2.0, 1.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

/// Flip Z on a 4×1×1 array reverses the sequence.
#[test]
fn flip_z_reverses_z_axis() {
    let img = make_image(vec![10.0f32, 20.0, 30.0, 40.0], [4, 1, 1]);
    let out = FlipImageFilter::flip_z().apply(&img).unwrap();
    let v = voxels(&out);
    let expected = [40.0f32, 30.0, 20.0, 10.0];
    for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

/// Applying the same flip twice returns the original image (involutory property).
#[test]
fn flip_twice_returns_original() {
    let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let flip = FlipImageFilter::new([FlipPolicy::Flip, FlipPolicy::Flip, FlipPolicy::Keep]);
    let out1 = flip.apply(&img).unwrap();
    let out2 = flip.apply(&out1).unwrap();
    let v = voxels(&out2);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "[{}] double-flip expected {}, got {}",
            i,
            b,
            a
        );
    }
}

/// Flip preserves shape and spatial metadata.
#[test]
fn flip_preserves_spatial_metadata() {
    let img = make_image(vec![1.0f32; 8], [2, 2, 2]);
    let out = FlipImageFilter::flip_y().apply(&img).unwrap();
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.origin(), img.origin());
}

/// Flip all three axes on a 2×3×4 volume and verify specific voxel.
#[test]
fn flip_all_axes_2x3x4_correctness() {
    // dims [nz=2, ny=3, nx=4]; voxel (iz,iy,ix) has value iz*100 + iy*10 + ix
    let dims = [2usize, 3, 4];
    let mut vals = vec![0.0f32; 2 * 3 * 4];
    for iz in 0..2usize {
        for iy in 0..3usize {
            for ix in 0..4usize {
                vals[iz * 12 + iy * 4 + ix] = (iz * 100 + iy * 10 + ix) as f32;
            }
        }
    }
    let img = make_image(vals, dims);
    let out = FlipImageFilter::new([FlipPolicy::Flip; 3])
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    // After flipping all axes: out(iz,iy,ix) = in(nz-1-iz, ny-1-iy, nx-1-ix)
    for iz in 0..2usize {
        for iy in 0..3usize {
            for ix in 0..4usize {
                let got = v[iz * 12 + iy * 4 + ix];
                let iz_s = 1 - iz;
                let iy_s = 2 - iy;
                let ix_s = 3 - ix;
                let exp = (iz_s * 100 + iy_s * 10 + ix_s) as f32;
                assert!(
                    (got - exp).abs() < 1e-4,
                    "({},{},{}): expected {}, got {}",
                    iz,
                    iy,
                    ix,
                    exp,
                    got
                );
            }
        }
    }
}
