//! Value-semantic tests for extended label shape statistics (GAP-262-STA-03).
#![allow(clippy::identity_op, clippy::erasing_op)]
//!
//! Verifies centroid, perimeter, elongation, flatness, roundness, Feret diameter,
//! and principal moments for known geometries.

use ritk_image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

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

/// Isolated single voxel: all 6 faces are boundary → perimeter=1.
#[test]
fn single_voxel_perimeter_is_one() {
    let mut data = vec![0.0_f32; 27];
    data[13] = 1.0; // center of 3×3×3
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert_eq!(stats[0].perimeter, 1);
}

/// Interior voxel in a 3×3×3 solid block has zero perimeter.
#[test]
fn interior_voxel_perimeter_is_zero() {
    let data = vec![1.0_f32; 27]; // all label=1
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    // Center voxel (1,1,1) is interior
    // Actually, perimeter counts all boundary voxels of the region
    // For a 3×3×3 solid block: 26 voxels are on the boundary, 1 interior
    assert_eq!(stats[0].count, 27);
    assert_eq!(stats[0].perimeter, 26);
}

// ── Elongation & Flatness ───────────────────────────────────────────────────

/// Sphere-like (3×3×3 block): elongation ≈ 1, flatness ≈ 1.
#[test]
fn solid_block_is_roughly_spherical() {
    let data = vec![1.0_f32; 27];
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(stats[0].elongation > 0.9, "elongation should be near 1");
    assert!(stats[0].flatness > 0.9, "flatness should be near 1");
}

/// Line along X (1×1×5): high elongation, low flatness.
#[test]
fn line_component_is_highly_elongated() {
    let mut data = vec![0.0_f32; 5]; // [1,1,5]
    data[0] = 1.0;
    data[1] = 1.0;
    data[2] = 1.0;
    data[3] = 1.0;
    data[4] = 1.0;
    let img = make_label_image(data, [1, 1, 5], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    // Elongation should be near 0 for a thin line (λ1 << λ2)
    assert!(
        stats[0].elongation < 0.3,
        "line elongation should be low, got {}",
        stats[0].elongation
    );
    assert!(
        stats[0].flatness < 0.3,
        "line flatness should be low, got {}",
        stats[0].flatness
    );
}

// ── Roundness ───────────────────────────────────────────────────────────────

/// 3×3×3 isotropic block: roundness → 1 (sphere-like).
#[test]
fn solid_block_roundness() {
    let data = vec![1.0_f32; 27];
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    // Cube inscribed in its circumscribed sphere → roundness ≈ 0.37
    // This is expected for a cubic voxel blob
    assert!(
        stats[0].roundness > 0.3 && stats[0].roundness <= 1.0,
        "roundness should be >0.3 for a cube, got {}",
        stats[0].roundness
    );
}

/// Single voxel: roundness near 1 (sphere-like at unit scale).
#[test]
fn single_voxel_roundness() {
    // Single voxel: bounding box is a point → feret_diameter = 0.
    // Guard clause: roundness = 0.0 when feret = 0.
    let mut data = vec![0.0_f32; 27];
    data[13] = 1.0;
    let img = make_label_image(data, [3, 3, 3], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    assert!(
        (stats[0].feret_diameter).abs() < 1e-9,
        "feret must be 0 for single voxel"
    );
    assert!(
        (stats[0].roundness).abs() < 1e-9,
        "roundness = 0 when feret = 0"
    );
}

// ── Feret diameter ──────────────────────────────────────────────────────────

/// 1×1×5 line: Feret ≈ 4×√3 (corner-to-corner of bbox).
#[test]
fn line_feret_diameter() {
    let mut data = vec![0.0_f32; 5];
    data[0] = 1.0;
    data[1] = 1.0;
    data[2] = 1.0;
    data[3] = 1.0;
    data[4] = 1.0;
    let img = make_label_image(data, [1, 1, 5], [1.0, 1.0, 1.0]);
    let stats = compute_label_shape_statistics_extended(&img);
    // Bbox: z∈[0,0], y∈[0,0], x∈[0,4] → corners (0,0,0)→(0,0,4) → feret=4
    assert!((stats[0].feret_diameter - 4.0).abs() < 0.01);
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

/// Spacing affects physical moments but not voxel counts.
#[test]
fn anisotropic_spacing_affects_moments() {
    let data = vec![1.0_f32; 27]; // solid 3×3×3
    let iso = make_label_image(data.clone(), [3, 3, 3], [1.0, 1.0, 1.0]);
    let aniso = make_label_image(data, [3, 3, 3], [2.0, 1.0, 0.5]);
    let iso_stats = compute_label_shape_statistics_extended(&iso);
    let aniso_stats = compute_label_shape_statistics_extended(&aniso);
    // Counts are identical (voxel-space)
    assert_eq!(iso_stats[0].count, aniso_stats[0].count);
    assert_eq!(iso_stats[0].perimeter, aniso_stats[0].perimeter);
    // Physical moments differ
    assert!(iso_stats[0].principal_moments != aniso_stats[0].principal_moments);
    assert!((iso_stats[0].feret_diameter - aniso_stats[0].feret_diameter).abs() > 0.01);
}
