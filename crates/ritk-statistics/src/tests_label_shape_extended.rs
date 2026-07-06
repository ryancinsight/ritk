//! Value-semantic tests for extended label shape statistics (GAP-262-STA-03).
#![allow(clippy::identity_op, clippy::erasing_op)]
//!
//! Verifies centroid, perimeter, elongation, flatness, roundness, Feret diameter,
//! and principal moments for known geometries.

use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

use super::*;

type B = NdArray<f32>;

fn make_label_image(data: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new(spacing),
        Direction::identity(),
    )
}

// ── Centroid ─────────────────────────────────────────────────────────────────

/// Single-voxel label: centroid = voxel position.
#[test]
fn single_voxel_centroid() {
    // [1,4,4] label map; label=1 at flat index 6 = (z=0, y=1, x=2).
    let mut data = vec![0.0_f32; 16];
    data[1 * 4 + 2] = 1.0; // flat = 0*16 + 1*4 + 2 = 6 → z=0, y=1, x=2
    let img = make_label_image(data, [1, 4, 4], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].label, 1);
    assert_eq!(stats[0].count, 1);
    assert!((stats[0].centroid[0] - 0.0).abs() < 1e-6, "z");
    assert!(
        (stats[0].centroid[1] - 1.0).abs() < 1e-6,
        "y={}",
        stats[0].centroid[1]
    );
    assert!(
        (stats[0].centroid[2] - 2.0).abs() < 1e-6,
        "x={}",
        stats[0].centroid[2]
    );
}

/// Symmetric pair: centroid is midpoint.
#[test]
fn symmetric_pair_centroid() {
    // Two voxels label=1 at (0,0,0) and (0,0,2)
    let mut data = vec![0.0_f32; 27]; // [3,3,3]
    data[0] = 1.0; // (0,0,0)
    data[2] = 1.0; // (0,0,2)
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert_eq!(stats.len(), 1);
    assert!((stats[0].centroid[0] - 0.0).abs() < 1e-6);
    assert!((stats[0].centroid[1] - 0.0).abs() < 1e-6);
    assert!((stats[0].centroid[2] - 1.0).abs() < 1e-6);
}

// ── Perimeter ────────────────────────────────────────────────────────────────

/// Isolated single voxel: Crofton surface-area perimeter matches ITK/SimpleITK
/// `GetPerimeter` (3.00408..., float-exact reference).
#[test]
fn single_voxel_perimeter_matches_itk() {
    let mut data = vec![0.0_f32; 27];
    data[13] = 1.0; // center of 3×3×3
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(
        (stats[0].perimeter - 3.0040803078963907).abs() < 1e-9,
        "single-voxel perimeter {} ≠ ITK 3.00408",
        stats[0].perimeter
    );
}

/// Solid 3×3×3 block: Crofton surface-area perimeter matches ITK/SimpleITK
/// `GetPerimeter` (41.07017..., float-exact reference).
#[test]
fn solid_block_perimeter_matches_itk() {
    let data = vec![1.0_f32; 27]; // all label=1
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert_eq!(stats[0].count, 27);
    assert!(
        (stats[0].perimeter - 41.070175434096235).abs() < 1e-9,
        "3×3×3 perimeter {} ≠ ITK 41.07018",
        stats[0].perimeter
    );
}

// ── Elongation & Flatness ───────────────────────────────────────────────────

/// Sphere-like (3×3×3 block): elongation ≈ 1, flatness ≈ 1 (ITK convention,
/// both ≥ 1 with 1 = isotropic).
#[test]
fn solid_block_is_roughly_spherical() {
    let data = vec![1.0_f32; 27];
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(
        (stats[0].elongation - 1.0).abs() < 0.1,
        "elongation should be near 1, got {}",
        stats[0].elongation
    );
    assert!(
        (stats[0].flatness - 1.0).abs() < 0.1,
        "flatness should be near 1, got {}",
        stats[0].flatness
    );
}

/// Elongated box (3×3×15 along X): ITK elongation = √(λ2/λ1) ≫ 1 (long axis vs
/// square cross-section), flatness = √(λ1/λ0) ≈ 1 (square cross-section).
#[test]
fn line_component_is_highly_elongated() {
    let (nz, ny, nx) = (3usize, 3usize, 15usize);
    let data = vec![1.0_f32; nz * ny * nx];
    let img = make_label_image(data, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    // λ2 (x, length 15) ≫ λ1 ≈ λ0 (3×3 cross-section) → elongation large.
    assert!(
        stats[0].elongation > 2.5,
        "elongated box elongation should be ≫ 1, got {}",
        stats[0].elongation
    );
    assert!(
        (stats[0].flatness - 1.0).abs() < 0.1,
        "square cross-section flatness should be ≈ 1, got {}",
        stats[0].flatness
    );
}

// ── Roundness ───────────────────────────────────────────────────────────────

/// 3×3×3 block: roundness = equivSpherePerimeter/perimeter matches ITK
/// `GetRoundness` (1.05974..., float-exact). Note ITK roundness is *not* clamped
/// to ≤ 1: a small blob whose discretised surface area under-estimates the true
/// boundary can exceed 1.
#[test]
fn solid_block_roundness_matches_itk() {
    let data = vec![1.0_f32; 27];
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(
        (stats[0].roundness - 1.0597418272119552).abs() < 1e-9,
        "3×3×3 roundness {} ≠ ITK 1.05974",
        stats[0].roundness
    );
}

/// Single voxel: roundness matches ITK `GetRoundness` (1.60980...). Feret is 0
/// (a single boundary voxel has no opposing pair) but roundness is driven by the
/// perimeter, so it is non-zero — the old feret-based guard no longer applies.
#[test]
fn single_voxel_roundness_matches_itk() {
    let mut data = vec![0.0_f32; 27];
    data[13] = 1.0;
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(
        (stats[0].feret_diameter).abs() < 1e-9,
        "feret must be 0 for single voxel"
    );
    assert!(
        (stats[0].roundness - 1.6098024574568734).abs() < 1e-9,
        "single-voxel roundness {} ≠ ITK 1.60980",
        stats[0].roundness
    );
}

// ── Feret diameter ──────────────────────────────────────────────────────────

/// 1×1×5 line: Feret = max surface-voxel distance = ‖(0,0,0)−(0,0,4)‖ = 4.
#[test]
fn line_feret_diameter() {
    let data = vec![1.0_f32; 5];
    let img = make_label_image(data, [1, 1, 5], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!((stats[0].feret_diameter - 4.0).abs() < 1e-9);
}

/// Single voxel: Feret ≈ 0 (point).
#[test]
fn single_voxel_feret_is_zero() {
    let mut data = vec![0.0_f32; 27];
    data[0] = 1.0;
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!((stats[0].feret_diameter - 0.0).abs() < 1e-6);
}

// ── Multiple labels ─────────────────────────────────────────────────────────

/// Two labels: results sorted by label.
#[test]
fn multiple_labels_sorted() {
    let mut data = vec![0.0_f32; 8]; // [2,2,2]
    data[0] = 2.0;
    data[7] = 1.0;
    let img = make_label_image(data, [2, 2, 2], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert_eq!(stats.len(), 2);
    assert_eq!(stats[0].label, 1);
    assert_eq!(stats[1].label, 2);
}

// ── Empty image ─────────────────────────────────────────────────────────────

/// No foreground labels → empty result.
#[test]
fn empty_image_returns_empty() {
    let data = vec![0.0_f32; 27];
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(stats.is_empty());
}

// ── Anisotropic spacing ─────────────────────────────────────────────────────

/// Spacing affects physical attributes (perimeter is now a physical surface
/// area, not a voxel count) but not the voxel count itself.
#[test]
fn anisotropic_spacing_affects_moments() {
    let data = vec![1.0_f32; 27]; // solid 3×3×3
    let iso = make_label_image(data.clone(), [3, 3, 3], [1.0, 1.0, 1.0]);
    let aniso = make_label_image(data, [3, 3, 3], [2.0, 1.0, 0.5]);
    let iso_stats = compute_label_shape_statistics_extended(&iso);
    let aniso_stats = compute_label_shape_statistics_extended(&aniso);
    // Counts are identical (voxel-space)
    assert_eq!(iso_stats[0].count, aniso_stats[0].count);
    // Physical surface area differs under anisotropic spacing.
    assert!((iso_stats[0].perimeter - aniso_stats[0].perimeter).abs() > 0.01);
    // Physical moments differ
    assert!(iso_stats[0].principal_moments != aniso_stats[0].principal_moments);
    assert!((iso_stats[0].feret_diameter - aniso_stats[0].feret_diameter).abs() > 0.01);
}
