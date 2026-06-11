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
//! Burn tensors use column-major (Fortran) memory layout. For a shape
//! [Z, Y, X], the flat index is computed as:
//!
//! flat(z, y, x) = z + Z·y + Z·Y·x
//!
//! where Z = shape[0], Y = shape[1]. The rightmost dimension varies
//! slowst in memory (outermost loop), the leftmost varies fastest.

use crate::bin_shrink::BinShrinkImageFilter;
use crate::downsample::DownsampleFilter;
use ritk_core::filter::ops::extract_vec_infallible;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_image_3d(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn make_image_2d(data: Vec<f32>, shape: [usize; 2]) -> Image<B, 2> {
    let device = Default::default();
    let t = Tensor::<B, 2>::from_data(TensorData::new(data, Shape::new(shape)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
}

/// Compute the flat (column-major) index for a 3D multi-index in the given shape.
/// flat(z, y, x) = z + Z*y + Z*Y*x where Z=shape[0], Y=shape[1].
#[inline]
fn flat3(shape: [usize; 3], z: usize, y: usize, x: usize) -> usize {
    z + shape[0] * y + shape[0] * shape[1] * x
}

/// Compute the flat (column-major) index for a 2D multi-index.
/// flat(r, c) = r + R*c where R=shape[0].
#[inline]
fn flat2(shape: [usize; 2], r: usize, c: usize) -> usize {
    r + shape[0] * c
}

// ── Test 8: BinShrink vs DownsampleFilter ────────────────────────────────────
//
// # Derivation
// BinShrink averages all voxels in each bin.
// Downsample picks every Nth voxel (subsample).
// For non-uniform input, these must produce different values.
//
// Input: [1,2,3,4,5,6,7,8] shaped as [2,2,2], factor=[2,2,2].
// BinShrink output: 4.5 (mean of all 8).
// Downsample output: I[0,0,0] = first element = 1 (column-major).

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
// 2D image [4,6], factor=[2,3]:
// out_shape = [2, 2]
//
// Column-major: I[r,c] = r + 4*c (shape[0]=4)
// output[0,0] = mean of I[0..2, 0..3]:
//   I[0,0]=0, I[1,0]=1, I[0,1]=4, I[1,1]=5, I[0,2]=8, I[1,2]=9
//   Mean = (0+1+4+5+8+9)/6 = 27/6 = 4.5
//
// output[1,1] = mean of I[2..4, 3..6]:
//   I[2,3]=14, I[3,3]=15, I[2,4]=18, I[3,4]=19, I[2,5]=22, I[3,5]=23
//   Mean = (14+15+18+19+22+23)/6 = 111/6 = 18.5

#[test]
fn two_d_support() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image_2d(data, [4, 6]);
    let out = BinShrinkImageFilter::new(vec![2, 3]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: 4 / 2 = 2");
    assert_eq!(s[1], 2, "dim1: 6 / 3 = 2");
    let (got, out_shape) = extract_vec_infallible(&out);
    // output[0,0] in column-major output [2,2]: flat = 0 + 2*0 = 0
    let expected_00 = (0.0 + 1.0 + 4.0 + 5.0 + 8.0 + 9.0) / 6.0;
    assert!(
        (got[flat2(out_shape, 0, 0)] - expected_00).abs() < 1e-4,
        "output[0,0] = {}, expected {}",
        got[flat2(out_shape, 0, 0)],
        expected_00
    );
    assert!(
        (expected_00 - 4.5).abs() < 1e-4,
        "analytical expected = 4.5, computed {}",
        expected_00
    );
    // output[1,1] in column-major output [2,2]: flat = 1 + 2*1 = 3
    let expected_11 = (14.0 + 15.0 + 18.0 + 19.0 + 22.0 + 23.0) / 6.0;
    assert!(
        (got[flat2(out_shape, 1, 1)] - expected_11).abs() < 1e-4,
        "output[1,1] = {}, expected {}",
        got[flat2(out_shape, 1, 1)],
        expected_11
    );
    assert!(
        (expected_11 - 18.5).abs() < 1e-4,
        "analytical expected = 18.5, computed {}",
        expected_11
    );
}

// ── Test 10: Non-uniform bin averaging with analytical result ─────────────────
//
// # Derivation
// 4×4×4 image with sequential data, factor [2,2,2].
// Output shape [2,2,2].
//
// Column-major for shape [4,4,4]: I[z,y,x] = z + 4*y + 16*x
//
// output[0,0,0] = mean of I[0..2, 0..2, 0..2]:
//   I[0,0,0]=0, I[1,0,0]=1, I[0,1,0]=4, I[1,1,0]=5
//   I[0,0,1]=16, I[1,0,1]=17, I[0,1,1]=20, I[1,1,1]=21
//   Mean = (0+1+4+5+16+17+20+21)/8 = 84/8 = 10.5
//
// output[1,1,1] = mean of I[2..4, 2..4, 2..4]:
//   I[2,2,2]=42, I[3,2,2]=43, I[2,3,2]=46, I[3,3,2]=47
//   I[2,2,3]=58, I[3,2,3]=59, I[2,3,3]=62, I[3,3,3]=63
//   Mean = (42+43+46+47+58+59+62+63)/8 = 420/8 = 52.5

#[test]
fn non_uniform_bin_averaging_analytical() {
    let n = 4 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [4, 4, 4]);
    let out = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let (got, out_shape) = extract_vec_infallible(&out);
    assert_eq!(out_shape, [2, 2, 2], "output shape must be [2,2,2]");
    // output[0,0,0]: flat = 0 + 2*0 + 4*0 = 0
    let expected_000 = 10.5f32;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected_000).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected_000
    );
    // output[1,1,1]: flat = 1 + 2*1 + 4*1 = 7
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
// Input shape [8,4,4], factor=[4,1,1]:
// out_shape = [2,4,4]
// Only dim 0 (z) is shrunk.
//
// Column-major for [8,4,4]: I[z,y,x] = z + 8*y + 32*x
//
// output[0,0,0] = mean of I[0..4, 0, 0]:
//   I[0,0,0]=0, I[1,0,0]=1, I[2,0,0]=2, I[3,0,0]=3
//   Mean = (0+1+2+3)/4 = 6/4 = 1.5

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
    // output[0,0,0] = mean of I[0,1,2,3, 0, 0]
    // flat3([8,4,4], z, 0, 0) = z for z in 0..4 → values 0,1,2,3
    let expected = (0.0 + 1.0 + 2.0 + 3.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected
    );
    // output[1,0,0] = mean of I[4,5,6,7, 0, 0]
    // flat3([8,4,4], z, 0, 0) = z for z in 4..8 → values 4,5,6,7
    let expected_100 = (4.0 + 5.0 + 6.0 + 7.0) / 4.0;
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
// Input shape [2,8,2], factor=[1,4,1]:
// out_shape = [2,2,2]
// Only dim 1 (y) is shrunk.
//
// Column-major for [2,8,2]: I[z,y,x] = z + 2*y + 16*x
//
// output[0,0,0] = mean of I[0, 0..4, 0]:
//   I[0,0,0]=0, I[0,1,0]=2, I[0,2,0]=4, I[0,3,0]=6
//   Mean = (0+2+4+6)/4 = 12/4 = 3.0
//
// output[0,1,0] = mean of I[0, 4..8, 0]:
//   I[0,4,0]=8, I[0,5,0]=10, I[0,6,0]=12, I[0,7,0]=14
//   Mean = (8+10+12+14)/4 = 44/4 = 11.0

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
    // output[0,0,0] in output [2,2,2]: flat = 0 + 2*0 + 4*0 = 0
    let expected_000 = (0.0 + 2.0 + 4.0 + 6.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected_000).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected_000
    );
    // output[0,1,0] in output [2,2,2]: flat = 0 + 2*1 + 4*0 = 2
    let expected_010 = (8.0 + 10.0 + 12.0 + 14.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 1, 0)] - expected_010).abs() < 1e-4,
        "output[0,1,0] = {}, expected {}",
        got[flat3(out_shape, 0, 1, 0)],
        expected_010
    );
}

// ── Test 13: Last-dimension shrink (rightmost/outermost axis) ─────────────────
//
// # Derivation
// Input shape [2,2,8], factor=[1,1,4]:
// out_shape = [2,2,2]
// Only dim 2 (x) is shrunk.
//
// Column-major for [2,2,8]: I[z,y,x] = z + 2*y + 4*x
//
// output[0,0,0] = mean of I[0, 0, 0..4]:
//   I[0,0,0]=0, I[0,0,1]=4, I[0,0,2]=8, I[0,0,3]=12
//   Mean = (0+4+8+12)/4 = 24/4 = 6.0
//
// output[0,0,1] = mean of I[0, 0, 4..8]:
//   I[0,0,4]=16, I[0,0,5]=20, I[0,0,6]=24, I[0,0,7]=28
//   Mean = (16+20+24+28)/4 = 88/4 = 22.0

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
    // output[0,0,0] in output [2,2,2]: flat = 0 + 2*0 + 4*0 = 0
    let expected_000 = (0.0 + 4.0 + 8.0 + 12.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected_000).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected_000
    );
    assert!(
        (expected_000 - 6.0).abs() < 1e-4,
        "analytical expected = 6.0, computed {}",
        expected_000
    );
    // output[0,0,1] in output [2,2,2]: flat = 0 + 2*0 + 4*1 = 4
    let expected_001 = (16.0 + 20.0 + 24.0 + 28.0) / 4.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 1)] - expected_001).abs() < 1e-4,
        "output[0,0,1] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 1)],
        expected_001
    );
    assert!(
        (expected_001 - 22.0).abs() < 1e-4,
        "analytical expected = 22.0, computed {}",
        expected_001
    );
}
