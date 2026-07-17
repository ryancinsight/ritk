//! Tests for BinShrinkImageFilter.
//!
//! Every test asserts on computed values (not just is_ok/is_some).
//! Analytical results are derived from the BinShrink specification:
//! O(o) = (1/N) · Σ I(o·f + b) for b in [0, f)
//!
//! Extended edge-case tests (2D support, single/middle/last dimension shrink,
//! comparison with DownsampleFilter) are in `tests_bin_shrink_edge.rs`.
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
use crate::native_support::LegacyBurnBackend;
use ritk_core::image::Image;
use ritk_image::test_support as ts;
use ritk_spatial::{Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_image_3d(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(data, shape)
}

fn make_image_3d_with_metadata(
    data: Vec<f32>,
    shape: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<B, 3> {
    ts::burn_compat::make_image_with::<B, 3>(
        data,
        shape,
        Some(Point::new(origin)),
        Some(Spacing::new(spacing)),
        None,
    )
}

/// Compute the flat (column-major) index for a 3D multi-index in the given shape.
/// flat(z, y, x) = z + Z*y + Z*Y*x where Z=shape[0], Y=shape[1].
#[inline]
fn flat3(shape: [usize; 3], z: usize, y: usize, x: usize) -> usize {
    z + shape[0] * y + shape[0] * shape[1] * x
}

// ── Test 1: Factor 1 is identity ─────────────────────────────────────────────
//
// # Derivation
// factor[d] = 1 → out_shape[d] = shape[d] / 1 = shape[d]
// Each bin contains exactly 1 voxel → mean = that voxel.
// Spacing[d] *= 1 → unchanged.

#[test]
fn factor_one_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image_3d(data.clone(), [2, 3, 4]);
    let out = BinShrinkImageFilter::new(vec![1, 1, 1]).apply(&img);
    assert_eq!(
        out.shape(),
        img.shape(),
        "shape must be unchanged for factor=1"
    );
    let (got, _) = extract_vec_infallible(&out);
    assert_eq!(got, data, "voxels must be identical for factor=1");
    assert_eq!(out.origin(), img.origin(), "origin must be preserved");
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
}

// ── Test 2: Factor 2 halves shape and doubles spacing ────────────────────────
//
// # Derivation
// Input shape [8,8,8], factor=[2,2,2]:
// out_shape = [8/2, 8/2, 8/2] = [4,4,4]
// spacing = [1*2, 1*2, 1*2] = [2,2,2]

#[test]
fn factor_two_halves_shape_and_doubles_spacing() {
    let n = 8 * 8 * 8;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [8, 8, 8]);
    let out = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 4, "dim0: 8 / 2 = 4");
    assert_eq!(s[1], 4, "dim1: 8 / 2 = 4");
    assert_eq!(s[2], 4, "dim2: 8 / 2 = 4");
    let sp = out.spacing();
    assert!((sp[0] - 2.0).abs() < 1e-9, "spacing[0] must double to 2.0");
    assert!((sp[1] - 2.0).abs() < 1e-9, "spacing[1] must double to 2.0");
    assert!((sp[2] - 2.0).abs() < 1e-9, "spacing[2] must double to 2.0");
}

// ── Test 3: Mean computation — uniform input ─────────────────────────────────
//
// # Derivation
// All voxels = 2.0. Bin average = mean of 2.0s = 2.0.
// Output shape [2,2,2] with all values 2.0.

#[test]
fn mean_computation_uniform_input() {
    let n = 4 * 4 * 4;
    let data = vec![2.0f32; n];
    let img = make_image_3d(data, [4, 4, 4]);
    let out = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let (got, shape) = extract_vec_infallible(&out);
    assert_eq!(shape, [2, 2, 2], "output shape must be [2,2,2]");
    for (i, &v) in got.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-6,
            "output[{}] = {} but expected 2.0",
            i,
            v
        );
    }
}

// ── Test 3b: Mean computation — known non-uniform values ─────────────────────
//
// # Derivation
// 2×2×2 image with values [1,2,3,4,5,6,7,8] and factor=[2,2,2].
// Single output voxel = mean of all 8 values = (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5.
//
// Regardless of column-major vs row-major layout, the sum of all 8 voxels
// in the single output bin is 1+2+...+8 = 36, so mean = 4.5.

#[test]
fn mean_computation_known_values() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image_3d(data, [2, 2, 2]);
    let out = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let (got, shape) = extract_vec_infallible(&out);
    assert_eq!(shape, [1, 1, 1], "output shape must be [1,1,1]");
    assert_eq!(got.len(), 1, "output must have exactly 1 voxel");
    assert!(
        (got[0] - 4.5).abs() < 1e-6,
        "mean of [1..8] = 4.5, got {}",
        got[0]
    );
}

// ── Test 4: Asymmetric factors ───────────────────────────────────────────────
//
// # Derivation
// Input shape [6,4,9], factor=[2,1,3]:
// out_shape = [6/2, 4/1, 9/3] = [3, 4, 3]
// spacing = [1*2, 1*1, 1*3] = [2, 1, 3]

#[test]
fn asymmetric_factors() {
    let n = 6 * 4 * 9;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [6, 4, 9]);
    let out = BinShrinkImageFilter::new(vec![2, 1, 3]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 3, "dim0: 6 / 2 = 3");
    assert_eq!(s[1], 4, "dim1: 4 / 1 = 4 (unchanged)");
    assert_eq!(s[2], 3, "dim2: 9 / 3 = 3");
    let sp = out.spacing();
    assert!((sp[0] - 2.0).abs() < 1e-9, "spacing[0]: 1*2 = 2.0");
    assert!((sp[1] - 1.0).abs() < 1e-9, "spacing[1]: 1*1 = 1.0");
    assert!((sp[2] - 3.0).abs() < 1e-9, "spacing[2]: 1*3 = 3.0");
}

// ── Test 5: Truncation of remainder voxels ───────────────────────────────────
//
// # Derivation
// 3D image shape [5,5,5], factor=[2,2,2]:
// out_shape = [5/2, 5/2, 5/2] = [2,2,2]
// Remainder = 1 voxel per dimension is discarded.
//
// For output[0,0,0]: average of I[0..2, 0..2, 0..2] (8 voxels).
// Column-major for shape [5,5,5]:
// I[z,y,x] = z + 5*y + 25*x
// I[0,0,0]=0, I[0,0,1]=25, I[0,1,0]=5, I[0,1,1]=30
// I[1,0,0]=1, I[1,0,1]=26, I[1,1,0]=6, I[1,1,1]=31
// Sum = 0+25+5+30+1+26+6+31 = 124, mean = 15.5

#[test]
fn truncation_of_remainder_voxels() {
    let n = 5 * 5 * 5;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data.clone(), [5, 5, 5]);
    let out = BinShrinkImageFilter::new(vec![2, 2, 2]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: floor(5/2) = 2");
    assert_eq!(s[1], 2, "dim1: floor(5/2) = 2");
    assert_eq!(s[2], 2, "dim2: floor(5/2) = 2");
    // output[0,0,0] = flat3(out_shape, 0, 0, 0) = 0
    let (got, out_shape) = extract_vec_infallible(&out);
    // Compute expected mean of bin I[0..2, 0..2, 0..2]
    let in_shape = [5usize, 5, 5];
    let mut sum = 0.0f32;
    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                sum += data[flat3(in_shape, bz, by, bx)];
            }
        }
    }
    let expected = sum / 8.0;
    assert!(
        (got[flat3(out_shape, 0, 0, 0)] - expected).abs() < 1e-4,
        "output[0,0,0] = {}, expected {}",
        got[flat3(out_shape, 0, 0, 0)],
        expected
    );
    assert!(
        (expected - 15.5).abs() < 1e-4,
        "analytical expected = 15.5, computed {}",
        expected
    );
}

// ── Test 6: Broadcast factor ─────────────────────────────────────────────────
//
// # Derivation
// factors=[2] on 3D image → all dims use factor 2.
// Input shape [4,4,4] → output shape [2,2,2].

#[test]
fn broadcast_factor() {
    let n = 4 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image_3d(data, [4, 4, 4]);
    let out = BinShrinkImageFilter::new(vec![2]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "broadcast factor must apply to dim 0");
    assert_eq!(s[1], 2, "broadcast factor must apply to dim 1");
    assert_eq!(s[2], 2, "broadcast factor must apply to dim 2");
    let sp = out.spacing();
    assert!((sp[0] - 2.0).abs() < 1e-9, "spacing[0] doubled");
    assert!((sp[1] - 2.0).abs() < 1e-9, "spacing[1] doubled");
    assert!((sp[2] - 2.0).abs() < 1e-9, "spacing[2] doubled");
}

// ── Test 7: Metadata preservation ────────────────────────────────────────────
//
// # Derivation
// Origin: unchanged (physical location of first voxel is the same).
// Direction: unchanged.
// Spacing: scaled by factor per dimension.

#[test]
fn metadata_preservation() {
    let origin = [10.0, 20.0, 30.0];
    let spacing = [0.5, 1.0, 2.0];
    let n = 6 * 4 * 8;
    let data = vec![1.0f32; n];
    let img = make_image_3d_with_metadata(data, [6, 4, 8], origin, spacing);
    let out = BinShrinkImageFilter::new(vec![3, 2, 4]).apply(&img);
    // Origin preserved
    assert_eq!(out.origin(), img.origin(), "origin must be preserved");
    // Direction preserved
    assert_eq!(
        out.direction(),
        img.direction(),
        "direction must be preserved"
    );
    // Spacing scaled: [0.5*3, 1.0*2, 2.0*4] = [1.5, 2.0, 8.0]
    let sp = out.spacing();
    assert!((sp[0] - 1.5).abs() < 1e-9, "spacing[0]: 0.5*3 = 1.5");
    assert!((sp[1] - 2.0).abs() < 1e-9, "spacing[1]: 1.0*2 = 2.0");
    assert!((sp[2] - 8.0).abs() < 1e-9, "spacing[2]: 2.0*4 = 8.0");
}
