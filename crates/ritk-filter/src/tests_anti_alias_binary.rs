//! Unit tests for the SparseField AntiAliasBinary solver.
//!
//! Value-semantic bit-exactness against `sitk.AntiAliasBinary` is covered by the
//! Python cmake-parity suite (the iteration oracle). These Rust tests assert the
//! structural invariants of the level-set output.

use super::AntiAliasBinaryImageFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;

type B = NdArray<f32>;

fn make(binary: &[f32], dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(binary.to_vec(), dims)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// A straight binary edge yields the exact half-integer signed-distance lattice
/// (…−1.5, −0.5, 0.5, 1.5…) clamped to ±(NumberOfLayers+1), foreground positive.
#[test]
fn straight_edge_is_half_integer_signed_distance() {
    // 2-D (nz==1): left half background (0), right half foreground (1).
    let (ny, nx) = (1usize, 8usize);
    let mut b = vec![0.0f32; ny * nx];
    for x in 4..nx {
        b[x] = 1.0;
    }
    let img = make(&b, [1, ny, nx]);
    let out = AntiAliasBinaryImageFilter::default().apply(&img);
    let v = voxels(&out);
    // Boundary between x=3 (bg) and x=4 (fg): values straddle at ±0.5.
    assert!((v[3] - (-0.5)).abs() < 1e-4, "x3 = {} (expected -0.5)", v[3]);
    assert!((v[4] - 0.5).abs() < 1e-4, "x4 = {} (expected 0.5)", v[4]);
    // Foreground is positive, background negative.
    assert!(v[7] > 0.0, "interior fg must be positive, got {}", v[7]);
    assert!(v[0] < 0.0, "interior bg must be negative, got {}", v[0]);
}

/// A uniform image is degenerate: `min == max` ⇒ `iso = max` ⇒ `shifted = 0`
/// everywhere ⇒ no zero crossing ⇒ no active layer. ITK fills the whole field
/// with the outermost outside level −(NumberOfLayers+1) (shifted ≤ 0 branch).
#[test]
fn uniform_image_has_no_boundary() {
    let dims = [1usize, 4, 4];
    let out = AntiAliasBinaryImageFilter::default().apply(&make(&vec![1.0; 16], dims));
    let v = voxels(&out);
    // 2-D ⇒ NumberOfLayers = 2 ⇒ outermost background level = −3.
    for &x in &v {
        assert!((x - (-3.0)).abs() < 1e-4, "uniform voxel = {} (expected -3.0)", x);
    }
}

/// The thresholded output (φ > 0) reproduces the input foreground exactly: the
/// per-pixel sign constraint locks the segmentation to the binary input.
#[test]
fn sign_is_locked_to_input_binary() {
    let (ny, nx) = (12usize, 12usize);
    let mut b = vec![0.0f32; ny * nx];
    for y in 3..9 {
        for x in 3..9 {
            b[y * nx + x] = 1.0;
        }
    }
    let img = make(&b, [1, ny, nx]);
    let out = AntiAliasBinaryImageFilter::default().apply(&img);
    let v = voxels(&out);
    for i in 0..(ny * nx) {
        let fg = b[i] > 0.5;
        assert_eq!(
            fg,
            v[i] > 0.0,
            "voxel {} sign must match input binary (fg={}, phi={})",
            i,
            fg,
            v[i]
        );
    }
}
