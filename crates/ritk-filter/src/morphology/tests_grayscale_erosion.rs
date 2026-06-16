//! Tests for grayscale_erosion
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// Erosion of a constant image returns the same constant.
///
/// **Proof**: min_{b ∈ B} c = c for all constants c. ∎
#[test]
fn test_constant_image_unchanged() {
    let dims = [8, 8, 8];
    let c = 7.0_f32;
    let vals = vec![c; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims);

    let filter = GrayscaleErosion::new(2);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - c).abs() < 1e-6,
            "constant image erosion: voxel {i} = {v}, expected {c}"
        );
    }
}

/// Erosion reduces a bright spot embedded in a dark background.
///
/// A single bright voxel surrounded by a lower background value must be
/// replaced by the background after erosion with radius ≥ 1, because the
/// minimum over the neighbourhood includes the background.
///
/// **Proof**: Let f(x₀) = h (bright), f(x) = bg for x ≠ x₀ with bg < h.
/// For any y with ‖y − x₀‖_∞ ≤ r, the neighbourhood B(y) contains at
/// least one voxel x ≠ x₀ (since |B| ≥ 27 > 1), so
/// (E_B f)(y) ≤ bg < h. In particular, (E_B f)(x₀) = bg. ∎
#[test]
fn test_bright_spot_reduced() {
    let dims = [8, 8, 8];
    let bg = 1.0_f32;
    let bright = 100.0_f32;
    let n = dims[0] * dims[1] * dims[2];
    let mut vals = vec![bg; n];

    // Place bright spot at centre (4, 4, 4)
    let centre = 4 * dims[1] * dims[2] + 4 * dims[2] + 4;
    vals[centre] = bright;

    let img = make_image(vals, dims);
    let filter = GrayscaleErosion::new(1);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // The bright spot should be replaced by the background
    assert!(
        (out[centre] - bg).abs() < 1e-6,
        "bright spot should be eroded to background: got {}, expected {bg}",
        out[centre]
    );

    // Interior voxels away from the spot should remain at background
    for iz in 1..dims[0] - 1 {
        for iy in 1..dims[1] - 1 {
            for ix in 1..dims[2] - 1 {
                let flat = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                assert!(
                    (out[flat] - bg).abs() < 1e-6,
                    "interior voxel ({iz},{iy},{ix}) = {}, expected {bg}",
                    out[flat]
                );
            }
        }
    }
}

/// Radius-0 erosion is identity.
///
/// **Proof**: B = {0}, so (E_B f)(x) = min{ f(x) } = f(x). ∎
#[test]
fn test_radius_zero_identity() {
    let dims = [6, 6, 6];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image(vals.clone(), dims);

    let filter = GrayscaleErosion::new(0);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, (&expected, &actual)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "radius-0 identity: voxel {i} = {actual}, expected {expected}"
        );
    }
}

/// Anti-extensivity: (E_B f)(x) ≤ f(x) for all x when B contains the origin.
///
/// **Proof**: 0 ∈ B ⇒ min_{b ∈ B} f(x+b) ≤ f(x+0) = f(x). ∎
#[test]
fn test_anti_extensivity() {
    let dims = [8, 8, 8];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| (i % 37) as f32 + 1.0).collect();
    let img = make_image(vals.clone(), dims);

    let filter = GrayscaleErosion::new(1);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, (&original, &eroded)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            eroded <= original + 1e-6,
            "anti-extensivity violated at voxel {i}: eroded = {eroded} > original = {original}"
        );
    }
}

/// Opening (erosion then dilation) removes small bright features.
///
/// A small bright cube (smaller than the structuring element) embedded in a
/// uniform background should be removed by opening = dilation(erosion(f)).
/// The erosion shrinks the feature below existence, then dilation cannot
/// recover it.
#[test]
fn test_opening_removes_small_bright_feature() {
    let dims = [16, 16, 16];
    let bg = 0.0_f32;
    let bright = 100.0_f32;
    let n = dims[0] * dims[1] * dims[2];
    let mut vals = vec![bg; n];

    // Place a single bright voxel at (8, 8, 8)
    let centre = 8 * dims[1] * dims[2] + 8 * dims[2] + 8;
    vals[centre] = bright;

    let img = make_image(vals, dims);

    // Erosion with radius 1
    let erosion = GrayscaleErosion::new(1);
    let eroded = erosion.apply(&img).unwrap();

    // Dilation with radius 1 (import from sibling module)
    let dilation = crate::morphology::GrayscaleDilation::new(1);
    let opened = dilation.apply(&eroded).unwrap();
    let out = extract_vals(&opened);

    // The single bright voxel should be completely removed by opening
    assert!(
        (out[centre] - bg).abs() < 1e-6,
        "opening should remove single bright voxel: got {}, expected {bg}",
        out[centre]
    );

    // All interior voxels should be at background level
    let margin = 2;
    for iz in margin..dims[0] - margin {
        for iy in margin..dims[1] - margin {
            for ix in margin..dims[2] - margin {
                let flat = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                assert!(
                    (out[flat] - bg).abs() < 1e-6,
                    "opening: voxel ({iz},{iy},{ix}) = {}, expected {bg}",
                    out[flat]
                );
            }
        }
    }
}
