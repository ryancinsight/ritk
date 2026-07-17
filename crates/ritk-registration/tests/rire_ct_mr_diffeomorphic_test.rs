//! RIRE CT/MR T1 image integration tests.
//!
//! These tests load the actual `.mha` volumes, check spatial metadata, and
//! verify intensity range expectations. They require the RIRE test data under
//! `test_data/registration/rire/` and can be run with:
//!
//! ```shell
//! cargo test --test rire_ct_mr_diffeomorphic_test -- --ignored
//! ```
//!
//! # RIRE provenance
//!
//! Images and the standard transformation were provided as part of the project
//! *Retrospective Image Registration Evaluation* (RIRE), National Institutes
//! of Health, Project Number 8R01EB002124-03, Principal Investigator
//! J. Michael Fitzpatrick, Vanderbilt University, Nashville TN.
//! Data site: <https://rire.insight-journal.org/>
//! License: Creative Commons Attribution 3.0 United States.

mod common;

use coeus_core::SequentialBackend;
use common::find_rire_dir;
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};

// ── Group 2 — Image integration tests ────────────────────────────────────────

/// # Specification
///
/// CT `.mha` must load with correct shape [29, 512, 512] and spacing
/// (0.6536, 0.6536, 4.0 mm). Intensity range must span at least from air
/// (-1000 HU) to bone cortex (+500 HU).
///
/// Assertions:
/// - `shape == [29, 512, 512]`
/// - `|spacing()[0] - 4.0| < 0.01` (z / slice spacing)
/// - `|spacing()[1] - 0.653595| < 1e-4` (y / row spacing)
/// - `|spacing()[2] - 0.653595| < 1e-4` (x / col spacing)
/// - `min(data) <= -1000.0`
/// - `max(data) >= 500.0`
///
/// # Reference
/// RIRE training_001_ct.mha — 16-bit signed integer, HU range [-1024, 1969].
#[test]
#[ignore = "requires test_data/registration/rire/"]
fn test_rire_mha_load_ct_metadata() {
    let Some(rire_dir) = find_rire_dir() else {
        eprintln!(
            "RIRE data directory not found; \
 skipping test_rire_mha_load_ct_metadata"
        );
        return;
    };

    let ct_path = rire_dir.join("training_001_ct.mha");
    if !ct_path.exists() {
        eprintln!(
            "CT file {} not found; skipping test_rire_mha_load_ct_metadata",
            ct_path.display()
        );
        return;
    }

    let backend = SequentialBackend;
    let image = MetaImageReader::new(backend)
        .read(&ct_path)
        .expect("CT .mha must load without error");

    // ── Shape [nz, ny, nx] ────────────────────────────────────────────────
    let shape = image.shape();
    assert_eq!(
        shape,
        [29, 512, 512],
        "CT shape must be [29, 512, 512], got {:?}",
        shape
    );

    // ── Spacing: spacing()[0]=z, [1]=y, [2]=x ─────────────────────────────
    let sz = image.spacing()[0]; // z / slice
    let sy = image.spacing()[1]; // y / row
    let sx = image.spacing()[2]; // x / col

    assert!(
        (sz - 4.0).abs() < 0.01,
        "CT z spacing must be ≈ 4.0 mm (±0.01), got {:.6}",
        sz
    );
    assert!(
        (sy - 0.653595).abs() < 1e-4,
        "CT y spacing must be ≈ 0.653595 mm (±1e-4), got {:.6}",
        sy
    );
    assert!(
        (sx - 0.653595).abs() < 1e-4,
        "CT x spacing must be ≈ 0.653595 mm (±1e-4), got {:.6}",
        sx
    );

    // ── Intensity range ───────────────────────────────────────────────────
    let voxels = image.data_cow_on(&backend).into_owned();
    let vmin = voxels.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = voxels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        vmin <= -1000.0,
        "CT min intensity must be <= -1000 HU (air), got {:.2}",
        vmin
    );
    assert!(
        vmax >= 500.0,
        "CT max intensity must be >= +500 HU (bone cortex), got {:.2}",
        vmax
    );

    eprintln!(
        "CT loaded: shape={:?}, spacing=[{:.4},{:.4},{:.4}] mm, range=[{:.1},{:.1}] HU",
        shape, sz, sy, sx, vmin, vmax
    );
}

/// # Specification
///
/// MRI T1 `.mha` must load with correct shape [26, 256, 256] and spacing
/// (1.25, 1.25, 4.0 mm). Positive-only signal intensities confirm correct
/// unsigned-short decoding.
///
/// Assertions:
/// - `shape == [26, 256, 256]`
/// - `|spacing()[0] - 4.0| < 0.01` (z / slice spacing)
/// - `|spacing()[1] - 1.25| < 1e-4` (y / row spacing)
/// - `|spacing()[2] - 1.25| < 1e-4` (x / col spacing)
/// - `min(data) >= 0.0`
/// - `max(data) >= 100.0`
///
/// # Reference
/// RIRE training_001_mr_T1.mha — 16-bit signed integer, SI range [2, 1626].
#[test]
#[ignore = "requires test_data/registration/rire/"]
fn test_rire_mha_load_mri_t1_metadata() {
    let Some(rire_dir) = find_rire_dir() else {
        eprintln!(
            "RIRE data directory not found; \
 skipping test_rire_mha_load_mri_t1_metadata"
        );
        return;
    };

    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    if !mri_path.exists() {
        eprintln!(
            "MRI T1 file {} not found; skipping test_rire_mha_load_mri_t1_metadata",
            mri_path.display()
        );
        return;
    }

    let backend = SequentialBackend;
    let image = MetaImageReader::new(backend)
        .read(&mri_path)
        .expect("MRI T1 .mha must load without error");

    // ── Shape [nz, ny, nx] ────────────────────────────────────────────────
    let shape = image.shape();
    assert_eq!(
        shape,
        [26, 256, 256],
        "MRI T1 shape must be [26, 256, 256], got {:?}",
        shape
    );

    // ── Spacing: spacing()[0]=z, [1]=y, [2]=x ─────────────────────────────
    let sz = image.spacing()[0]; // z / slice
    let sy = image.spacing()[1]; // y / row
    let sx = image.spacing()[2]; // x / col

    assert!(
        (sz - 4.0).abs() < 0.01,
        "MRI T1 z spacing must be ≈ 4.0 mm (±0.01), got {:.6}",
        sz
    );
    assert!(
        (sy - 1.25).abs() < 1e-4,
        "MRI T1 y spacing must be ≈ 1.25 mm (±1e-4), got {:.6}",
        sy
    );
    assert!(
        (sx - 1.25).abs() < 1e-4,
        "MRI T1 x spacing must be ≈ 1.25 mm (±1e-4), got {:.6}",
        sx
    );

    // ── Intensity range ───────────────────────────────────────────────────
    let voxels = image.data_cow_on(&backend).into_owned();
    let vmin = voxels.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = voxels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        vmin >= 0.0,
        "MRI T1 min intensity must be >= 0.0 (positive-only signal), got {:.2}",
        vmin
    );
    assert!(
        vmax >= 100.0,
        "MRI T1 max intensity must be >= 100.0 (non-trivial signal), got {:.2}",
        vmax
    );

    eprintln!(
        "MRI T1 loaded: shape={:?}, spacing=[{:.4},{:.4},{:.4}] mm, range=[{:.1},{:.1}]",
        shape, sz, sy, sx, vmin, vmax
    );
}
