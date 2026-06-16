//! Tests for grayscale_dilation
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

/// Dilation of a constant image returns the same constant.
///
/// **Proof**: max_{b ∈ B} c = c for all constants c. ∎
#[test]
fn test_constant_image_unchanged() {
    let dims = [8, 8, 8];
    let c = 7.0_f32;
    let vals = vec![c; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims);

    let filter = GrayscaleDilation::new(2);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - c).abs() < 1e-6,
            "constant image dilation: voxel {i} = {v}, expected {c}"
        );
    }
}

/// Dilation expands a bright spot: a single bright voxel at the centre
/// should propagate to all voxels within the structuring element radius.
///
/// **Proof**: Let f(x₀) = h, f(x) = bg for x ≠ x₀ with h > bg.
/// For any y with ‖y − x₀‖_∞ ≤ r, x₀ ∈ B(y), so
/// (D_B f)(y) ≥ f(x₀) = h > bg. For y with ‖y − x₀‖_∞ > r,
/// B(y) ∩ {x₀} = ∅, so (D_B f)(y) = bg. ∎
#[test]
fn test_bright_spot_expanded() {
    let dims = [16, 16, 16];
    let bg = 1.0_f32;
    let bright = 100.0_f32;
    let n = dims[0] * dims[1] * dims[2];
    let mut vals = vec![bg; n];

    // Place bright spot at centre (8, 8, 8)
    let cz = 8usize;
    let cy = 8usize;
    let cx = 8usize;
    let centre = cz * dims[1] * dims[2] + cy * dims[2] + cx;
    vals[centre] = bright;

    let img = make_image(vals, dims);
    let radius = 1usize;
    let filter = GrayscaleDilation::new(radius);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // All voxels within L∞ distance ≤ radius of the centre should be bright
    for iz in 0..dims[0] {
        for iy in 0..dims[1] {
            for ix in 0..dims[2] {
                let flat = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                let dz = (iz as isize - cz as isize).unsigned_abs();
                let dy = (iy as isize - cy as isize).unsigned_abs();
                let dx = (ix as isize - cx as isize).unsigned_abs();
                let linf = dz.max(dy).max(dx);

                if linf <= radius {
                    assert!(
                        (out[flat] - bright).abs() < 1e-6,
                        "dilation should expand bright spot: voxel ({iz},{iy},{ix}) = {}, expected {bright}",
                        out[flat]
                    );
                } else if linf > radius + 1 {
                    // Voxels far from the spot remain at background
                    assert!(
                        (out[flat] - bg).abs() < 1e-6,
                        "voxel ({iz},{iy},{ix}) should remain at background: got {}, expected {bg}",
                        out[flat]
                    );
                }
            }
        }
    }
}

/// Radius-0 dilation is identity.
///
/// **Proof**: B = {0}, so (D_B f)(x) = max{ f(x) } = f(x). ∎
#[test]
fn test_radius_zero_identity() {
    let dims = [6, 6, 6];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image(vals.clone(), dims);

    let filter = GrayscaleDilation::new(0);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, (&expected, &actual)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "radius-0 identity: voxel {i} = {actual}, expected {expected}"
        );
    }
}

/// Extensivity: (D_B f)(x) ≥ f(x) for all x when B contains the origin.
///
/// **Proof**: 0 ∈ B ⇒ max_{b ∈ B} f(x+b) ≥ f(x+0) = f(x). ∎
#[test]
fn test_extensivity() {
    let dims = [8, 8, 8];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| (i % 37) as f32 + 1.0).collect();
    let img = make_image(vals.clone(), dims);

    let filter = GrayscaleDilation::new(1);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, (&original, &dilated)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            dilated >= original - 1e-6,
            "extensivity violated at voxel {i}: dilated = {dilated} < original = {original}"
        );
    }
}

/// Duality: D_B(f) = −E_B(−f) for flat symmetric structuring elements.
///
/// Verify numerically that dilation of f equals the negation of erosion
/// applied to −f.
#[test]
fn test_duality_with_erosion() {
    let dims = [8, 8, 8];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 41) as f32).collect();
    let neg_vals: Vec<f32> = vals.iter().map(|&v| -v).collect();

    let img = make_image(vals, dims);
    let neg_img = make_image(neg_vals, dims);

    let radius = 1;
    let dilation = GrayscaleDilation::new(radius);
    let erosion = crate::morphology::GrayscaleErosion::new(radius);

    let dilated = extract_vals(&dilation.apply(&img).unwrap());
    let eroded_neg = extract_vals(&erosion.apply(&neg_img).unwrap());

    for i in 0..n {
        let expected = -eroded_neg[i];
        assert!(
            (dilated[i] - expected).abs() < 1e-5,
            "duality violated at voxel {i}: D_B(f) = {}, -E_B(-f) = {expected}",
            dilated[i]
        );
    }
}
