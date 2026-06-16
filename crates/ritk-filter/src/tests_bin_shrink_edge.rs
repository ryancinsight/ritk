//! Edge-case and boundary tests for BinShrinkImageFilter.
//!
//! This file covers Tests 8–13: comparison with DownsampleFilter, 2D support,
//! non-uniform bin averaging, and single/middle/last dimension shrink.
//!
//! Every test asserts on computed values (not just is_ok/is_some).
//! Analytical results are derived from the BinShrink specification:
//! O(o) = (1/N) · Σ I(o·f + b) for b in [0, f)
//!
//! # Layout convention
//!
//! Burn `NdArray` tensors use row-major (C-contiguous) memory layout, matching
//! the rest of ritk. For a shape [Z, Y, X], the flat index is
//!
//! flat(z, y, x) = z·Y·X + y·X + x
//!
//! so the rightmost dimension (X) varies fastest in memory and the leftmost (Z)
//! varies slowest. (An earlier version of these tests assumed column-major
//! layout — `z + Z·y + Z·Y·x` — which only agrees with the real layout on
//! layout-symmetric bins; the bin-shrink stride walk averaged the wrong axes
//! for any genuine anisotropic case.)

use crate::bin_shrink::BinShrinkImageFilter;
use crate::downsample::DownsampleFilter;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_image_3d(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

fn make_image_2d(data: Vec<f32>, shape: [usize; 2]) -> Image<B, 2> {
    ts::make_image::<B, 2>(data, shape)
}

/// Row-major flat index for a 3D multi-index: flat(z, y, x) = z·Y·X + y·X + x
/// where Y = shape[1], X = shape[2].
#[inline]
fn flat3(shape: [usize; 3], z: usize, y: usize, x: usize) -> usize {
    z * shape[1] * shape[2] + y * shape[2] + x
}

/// Row-major flat index for a 2D multi-index: flat(r, c) = r·C + c where
/// C = shape[1].
#[inline]
fn flat2(shape: [usize; 2], r: usize, c: usize) -> usize {
    r * shape[1] + c
}

// ── Test 8: BinShrink vs DownsampleFilter ────────────────────────────────────
//
// # Derivation
// BinShrink averages all voxels in each bin.
// Downsample picks every Nth voxel (subsample).
//
// Input: [1,2,3,4,5,6,7,8] shaped as [2,2,2], factor=[2,2,2].
// BinShrink output: 4.5 (mean of all 8, layout-invariant).
// Downsample output: I[0,0,0] = first element = 1.

#[test]
fn bin_shrink_differs_from_downsample() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image_3d(data, [2, 2, 2]);
    let shrunk = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let downsampled = DownsampleFilter::<B>::new(vec![2, 2, 2]).apply(&img);
    let (shrink_vals, _) = extract_vec_infallible(&shrunk);
    let (down_vals, _) = extract_vec_infallible(&downsampled);
    // BinShrink: single voxel = 4.5
    assert_eq!(shrink_vals.len(), 1, "BinShrink output size");
    assert!(
        (shrink_vals[0] - 4.5).abs() < 1e-6,
        "BinShrink mean = 4.5, got {}",
        shrink_vals[0]
    );
    // Downsample: single voxel = I[0,0,0] (first element)
    assert_eq!(down_vals.len(), 1, "Downsample output size");
    assert!(
        (down_vals[0] - 1.0).abs() < 1e-6,
        "Downsample should pick first voxel = 1.0, got {}",
        down_vals[0]
    );
    // They must differ
    assert!(
        (shrink_vals[0] - down_vals[0]).abs() > 0.1,
        "BinShrink ({}) must differ from Downsample ({})",
        shrink_vals[0],
        down_vals[0]
    );
}

// ── Test 9: 2D support ──────────────────────────────────────────────────────
//
// # Derivation
// 2D image [4,6], factor=[2,3] → out_shape = [2, 2].
// Row-major: I[r,c] = 6·r + c.
//
// output[0,0] = mean of I[0..2, 0..3]:
//   I[0,0]=0, I[0,1]=1, I[0,2]=2, I[1,0]=6, I[1,1]=7, I[1,2]=8
//   Mean = (0+1+2+6+7+8)/6 = 24/6 = 4.0
//
// output[1,1] = mean of I[2..4, 3..6]:
//   I[2,3]=15, I[2,4]=16, I[2,5]=17, I[3,3]=21, I[3,4]=22, I[3,5]=23
//   Mean = (15+16+17+21+22+23)/6 = 114/6 = 19.0

#[test]
fn two_d_support() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image_2d(data, [4, 6]);
    let out = BinShrinkImageFilter::new(vec![2, 3]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: 4 / 2 = 2");
    assert_eq!(s[1], 2, "dim1: 6 / 3 = 2");
    let (got, out_shape) = extract_vec_infallible(&out);
    let expected_00 = (0.0 + 1.0 + 2.0 + 6.0 + 7.0 + 8.0) / 6.0;
    assert!(
        (got[flat2(out_shape, 0, 0)] - expected_00).abs() < 1e-4,
        "output[0,0] = {}, expected {}",
        got[flat2(out_shape, 0, 0)],
        expected_00
    );
    assert!(
        (expected_00 - 4.0).abs() < 1e-4,
        "analytical expected = 4.0, computed {}",
        expected_00
    );
    let expected_11 = (15.0 + 16.0 + 17.0 + 21.0 + 22.0 + 23.0) / 6.0;
    assert!(
        (got[flat2(out_shape, 1, 1)] - expected_11).abs() < 1e-4,
        "output[1,1] = {}, expected {}",
        got[flat2(out_shape, 1, 1)],
        expected_11
    );
    assert!(
        (expected_11 - 19.0).abs() < 1e-4,
        "analytical expected = 19.0, computed {}",
        expected_11
    );
}

// ── Test 10: Non-uniform bin averaging with analytical result ─────────────────
//
// # Derivation
// 4×4×4 image with sequential data, factor [2,2,2] → output shape [2,2,2].
// Row-major for shape [4,4,4]: I[z,y,x] = 16·z + 4·y + x.
//
// output[0,0,0] = mean of I[0..2, 0..2, 0..2]:
//   {0,1,4,5,16,17,20,21} → 84/8 = 10.5
// output[1,1,1] = mean of I[2..4, 2..4, 2..4]:
//   {42,43,46,47,58,59,62,63} → 420/8 = 52.5

#[test]
fn non_uniform_bin_averaging_analytical() {
    let n = 4 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [4, 4, 4]);
    let out = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let (got, out_shape) = extract_vec_infallible(&out);
    assert_eq!(out_shape, [2, 2, 2], "output shape must be [2,2,2]");
    let expected_000 = 10.5f32;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected_000).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected_000
    );
    let expected_111 = 52.5f32;
    assert!(
        (got[flat3(out_shape, 1, 1, 1)] - expected_111).abs() < 1e-4,
        "output[1,1,1] = {}, expected {}",
        got[flat3(out_shape, 1, 1, 1)],
        expected_111
    );
}

// ── Test 11: Single-dimension shrink preserves other dimensions ───────────────
//
// # Derivation
// Input shape [8,4,4], factor=[4,1,1] → out_shape = [2,4,4]; only dim 0 (z) shrinks.
// Row-major for [8,4,4]: I[z,y,x] = 16·z + 4·y + x.
//
// output[0,0,0] = mean of I[0..4, 0, 0] = {0,16,32,48} → 96/4 = 24.0
// output[1,0,0] = mean of I[4..8, 0, 0] = {64,80,96,112} → 352/4 = 88.0

#[test]
fn single_dimension_shrink() {
    let n = 8 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [8, 4, 4]);
    let out = BinShrinkImageFilter::new(vec![4, 1, 1]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: 8 / 4 = 2");
    assert_eq!(s[1], 4, "dim1: unchanged");
    assert_eq!(s[2], 4, "dim2: unchanged");
    let (got, out_shape) = extract_vec_infallible(&out);
    let expected = (0.0 + 16.0 + 32.0 + 48.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected
    );
    let expected_100 = (64.0 + 80.0 + 96.0 + 112.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 1, 0, 0)] - expected_100).abs() < 1e-4,
        "output[1,0,0] = {}, expected {}",
        got[flat3(out_shape, 1, 0, 0)],
        expected_100
    );
}

// ── Test 12: Middle-dimension shrink (non-contiguous axis) ────────────────────
//
// # Derivation
// Input shape [2,8,2], factor=[1,4,1] → out_shape = [2,2,2]; only dim 1 (y) shrinks.
// Row-major for [2,8,2]: I[z,y,x] = 16·z + 2·y + x.
//
// output[0,0,0] = mean of I[0, 0..4, 0] = {0,2,4,6} → 12/4 = 3.0
// output[0,1,0] = mean of I[0, 4..8, 0] = {8,10,12,14} → 44/4 = 11.0

#[test]
fn middle_dimension_shrink() {
    let n = 2 * 8 * 2;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [2, 8, 2]);
    let out = BinShrinkImageFilter::new(vec![1, 4, 1]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: unchanged");
    assert_eq!(s[1], 2, "dim1: 8 / 4 = 2");
    assert_eq!(s[2], 2, "dim2: unchanged");
    let (got, out_shape) = extract_vec_infallible(&out);
    let expected_000 = (0.0 + 2.0 + 4.0 + 6.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected_000).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected_000
    );
    let expected_010 = (8.0 + 10.0 + 12.0 + 14.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 1, 0)] - expected_010).abs() < 1e-4,
        "output[0,1,0] = {}, expected {}",
        got[flat3(out_shape, 0, 1, 0)],
        expected_010
    );
}

// ── Test 13: Last-dimension shrink (rightmost/innermost axis) ──────────────────
//
// # Derivation
// Input shape [2,2,8], factor=[1,1,4] → out_shape = [2,2,2]; only dim 2 (x) shrinks.
// Row-major for [2,2,8]: I[z,y,x] = 16·z + 8·y + x.
//
// output[0,0,0] = mean of I[0, 0, 0..4] = {0,1,2,3} → 6/4 = 1.5
// output[0,0,1] = mean of I[0, 0, 4..8] = {4,5,6,7} → 22/4 = 5.5

#[test]
fn last_dimension_shrink() {
    let n = 2 * 2 * 8;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [2, 2, 8]);
    let out = BinShrinkImageFilter::new(vec![1, 1, 4]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: unchanged");
    assert_eq!(s[1], 2, "dim1: unchanged");
    assert_eq!(s[2], 2, "dim2: 8 / 4 = 2");
    let (got, out_shape) = extract_vec_infallible(&out);
    let expected_000 = (0.0 + 1.0 + 2.0 + 3.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected_000).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected_000
    );
    assert!(
        (expected_000 - 1.5).abs() < 1e-4,
        "analytical expected = 1.5, computed {}",
        expected_000
    );
    let expected_001 = (4.0 + 5.0 + 6.0 + 7.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 1)] - expected_001).abs() < 1e-4,
        "output[0,0,1] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 1)],
        expected_001
    );
    assert!(
        (expected_001 - 5.5).abs() < 1e-4,
        "analytical expected = 5.5, computed {}",
        expected_001
    );
}
