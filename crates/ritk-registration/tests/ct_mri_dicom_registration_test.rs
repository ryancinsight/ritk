//! CT+MRI DICOM registration integration tests.
//!
//! # Test Data
//! These tests require the MRI-DIR CT (3_head_ct_mridir/DICOM/) and MRI
//! (2_head_mri_t2/DICOM/) DICOM series to be present under test_data/.
//!
//! CT: 409 slices, 512×512, 0.625mm thickness, 0.390625mm pixel spacing,
//!     from TCIA MRI-DIR collection (zzmeatphantom, CC BY 4.0,
//!     Ger et al. 2018, DOI: 10.1002/mp.13090).
//! MRI: 94 slices, T2-weighted, same phantom, paired for CT↔MRI registration.
//!
//! To run these tests:
//!   cargo test --test ct_mri_dicom_registration_test -- --ignored

use burn_ndarray::NdArray;
use ritk_io::{read_dicom_series_with_metadata, DicomReadMetadata};
use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration};

type B = NdArray<f32>;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn find_test_data_dir() -> Option<std::path::PathBuf> {
    for p in &["test_data", "../test_data", "../../test_data"] {
        let p = std::path::Path::new(p);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Downsample a flat z-major volume by taking every `stride`-th voxel along
/// each axis.
///
/// # Returns
/// `(downsampled_flat_vec, [new_nz, new_ny, new_nx])`
fn downsample_stride(data: &[f32], dims: [usize; 3], stride: usize) -> (Vec<f32>, [usize; 3]) {
    let [nz, ny, nx] = dims;
    let new_nz = (nz + stride - 1) / stride;
    let new_ny = (ny + stride - 1) / stride;
    let new_nx = (nx + stride - 1) / stride;
    let mut out = Vec::with_capacity(new_nz * new_ny * new_nx);
    for iz in (0..nz).step_by(stride) {
        for iy in (0..ny).step_by(stride) {
            for ix in (0..nx).step_by(stride) {
                let idx = iz * ny * nx + iy * nx + ix;
                out.push(data[idx]);
            }
        }
    }
    (out, [new_nz, new_ny, new_nx])
}

/// Minmax-normalize a volume to `[0, 1]`.
///
/// When `max == min` the denominator is clamped to `1e-8` to avoid division
/// by zero, producing an all-zero output.
fn normalize_minmax(data: &[f32]) -> Vec<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    data.iter().map(|&v| (v - min) / range).collect()
}

/// Pearson normalized cross-correlation of two equal-length float slices.
///
/// # Mathematical basis
/// ```text
/// NCC(a, b) = Σᵢ (aᵢ − ā)(bᵢ − b̄)
///             ─────────────────────────
///             ‖a − ā‖₂ · ‖b − b̄‖₂
/// ```
/// Returns `0.0` when either input has zero standard deviation (degenerate
/// case where the denominator would be < `1e-12`).
fn ncc(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "ncc: inputs must have equal length");
    let n = a.len() as f64;
    let ma: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - ma) * (y as f64 - mb))
        .sum();
    let da: f64 = a
        .iter()
        .map(|&v| (v as f64 - ma).powi(2))
        .sum::<f64>()
        .sqrt();
    let db: f64 = b
        .iter()
        .map(|&v| (v as f64 - mb).powi(2))
        .sum::<f64>()
        .sqrt();
    if da < 1e-12 || db < 1e-12 {
        return 0.0;
    }
    num / (da * db)
}

// ── Integration tests ─────────────────────────────────────────────────────────

/// # Specification
///
/// The MRI-DIR CT series (TCIA, zzmeatphantom, CC BY 4.0) must load with:
/// - Modality == `"CT"`
/// - Shape: 409 slices × 512 rows × 512 columns (±4 tolerance for DICOM
///   padding artefacts introduced during multi-frame reconstruction)
/// - z spacing (slice thickness) ≈ 0.625 mm (±0.15)
/// - y spacing (row pixel spacing) ≈ 0.390625 mm (±0.01)
/// - x spacing (col pixel spacing) ≈ 0.390625 mm (±0.01)
///
/// # Reference
/// Ger et al., Medical Physics 2018, DOI: 10.1002/mp.13090
#[test]
#[ignore = "requires test data"]
fn test_ct_dicom_series_metadata() {
    let Some(root) = find_test_data_dir() else {
        eprintln!("test_data directory not found; skipping test_ct_dicom_series_metadata");
        return;
    };
    let ct_dir = root.join("3_head_ct_mridir").join("DICOM");
    if !ct_dir.exists() {
        eprintln!(
            "CT DICOM path {} not found; skipping test_ct_dicom_series_metadata",
            ct_dir.display()
        );
        return;
    }

    let device = Default::default();
    let (image, metadata): (_, DicomReadMetadata) =
        read_dicom_series_with_metadata::<B, _>(&ct_dir, &device)
            .expect("CT DICOM series must load without error");

    // ── Modality ──────────────────────────────────────────────────────────
    assert_eq!(
        metadata.modality.as_deref(),
        Some("CT"),
        "modality must be CT, got {:?}",
        metadata.modality
    );

    // ── Shape [nz, ny, nx] ────────────────────────────────────────────────
    let shape = image.shape();
    assert!(
        (405..=413).contains(&shape[0]),
        "expected 409 ± 4 slices, got {}",
        shape[0]
    );
    assert_eq!(shape[1], 512, "expected 512 rows, got {}", shape[1]);
    assert_eq!(shape[2], 512, "expected 512 columns, got {}", shape[2]);

    // ── Physical spacing [sz, sy, sx] ─────────────────────────────────────
    // spacing()[0] = z (slice thickness); spacing()[1] = y (row); spacing()[2] = x (col).
    let sz = image.spacing()[0];
    let sy = image.spacing()[1];
    let sx = image.spacing()[2];

    assert!(
        (sz - 0.625_f64).abs() <= 0.15,
        "z spacing (slice thickness) expected ≈ 0.625 mm ± 0.15, got {:.6}",
        sz
    );
    assert!(
        (sy - 0.390_625_f64).abs() <= 0.01,
        "y spacing (row pixel spacing) expected ≈ 0.390625 mm ± 0.01, got {:.6}",
        sy
    );
    assert!(
        (sx - 0.390_625_f64).abs() <= 0.01,
        "x spacing (col pixel spacing) expected ≈ 0.390625 mm ± 0.01, got {:.6}",
        sx
    );
}

/// # Specification
///
/// The MRI-DIR T2 MRI series must load with:
/// - Modality == `"MR"`
/// - Shape: 94 slices ± 2 tolerance
/// - In-plane dimensions: rows ≥ 64, columns ≥ 64
/// - Non-trivial intensity range: `min < 0.01 × max` (rules out constant,
///   all-zero, and binary volumes)
///
/// # Reference
/// Ger et al., Medical Physics 2018, DOI: 10.1002/mp.13090
#[test]
#[ignore = "requires test data"]
fn test_mri_dir_mri_series_metadata() {
    let Some(root) = find_test_data_dir() else {
        eprintln!("test_data directory not found; skipping test_mri_dir_mri_series_metadata");
        return;
    };
    let mri_dir = root.join("2_head_mri_t2").join("DICOM");
    if !mri_dir.exists() {
        eprintln!(
            "MRI DICOM path {} not found; skipping test_mri_dir_mri_series_metadata",
            mri_dir.display()
        );
        return;
    }

    let device = Default::default();
    let (image, metadata): (_, DicomReadMetadata) =
        read_dicom_series_with_metadata::<B, _>(&mri_dir, &device)
            .expect("MRI DICOM series must load without error");

    // ── Modality ──────────────────────────────────────────────────────────
    assert_eq!(
        metadata.modality.as_deref(),
        Some("MR"),
        "modality must be MR, got {:?}",
        metadata.modality
    );

    // ── Shape [nz, ny, nx] ────────────────────────────────────────────────
    let shape = image.shape();
    assert!(
        (92..=96).contains(&shape[0]),
        "expected 94 ± 2 slices, got {}",
        shape[0]
    );
    assert!(shape[1] >= 64, "expected ≥64 rows, got {}", shape[1]);
    assert!(shape[2] >= 64, "expected ≥64 columns, got {}", shape[2]);

    // ── Non-trivial intensity range ───────────────────────────────────────
    let voxels: Vec<f32> = image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("MRI tensor must contain f32 data")
        .to_vec();
    let vmin = voxels.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = voxels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        vmax > 0.0,
        "MRI maximum intensity must be positive, got {}",
        vmax
    );
    assert!(
        vmin < 0.01 * vmax,
        "MRI min ({:.4}) must be < 0.01 × max ({:.4}) — data must have dynamic range",
        vmin,
        vmax
    );
}

/// # Specification
///
/// Given a downsampled CT sub-volume (stride = 16, yielding ≈ 26×32×32 voxels)
/// minmax-normalized to `[0, 1]`, and a second volume derived by a cyclic
/// 2-voxel shift in x:
///
/// ```text
/// NCC(fixed, warped_after) > NCC(fixed, moving_before) − 0.001
/// NCC(fixed, warped_after) ≥ 0.80
/// ```
///
/// # Mathematical basis
///
/// NCC ∈ `[−1, 1]`. A cyclic 2-voxel translation in x on a ≈26×32×32
/// normalized CT volume reduces NCC from 1.0 to approximately 0.7–0.9
/// depending on the spatial frequency content of the sub-volume. A single-level
/// BSpline FFD with `initial_control_spacing = [4, 4, 4]` and 50 gradient-
/// descent iterations maximizes the NCC objective, recovering the shift.
/// The assertion `NCC_after ≥ 0.80` bounds the minimum acceptable recovery.
///
/// # Reference
/// Rueckert et al., IEEE TMI 18(8):712–721, 1999 (B-spline FFD).
#[test]
#[ignore = "requires test data"]
fn test_bspline_ffd_mridir_ct_synthetic_shift_recovery() {
    let Some(root) = find_test_data_dir() else {
        eprintln!(
            "test_data directory not found; \
             skipping test_bspline_ffd_mridir_ct_synthetic_shift_recovery"
        );
        return;
    };
    let ct_dir = root.join("3_head_ct_mridir").join("DICOM");
    if !ct_dir.exists() {
        eprintln!(
            "CT DICOM path {} not found; \
             skipping test_bspline_ffd_mridir_ct_synthetic_shift_recovery",
            ct_dir.display()
        );
        return;
    }

    let device = Default::default();
    let (image, _metadata) = read_dicom_series_with_metadata::<B, _>(&ct_dir, &device)
        .expect("CT DICOM series must load for registration test");

    let full_shape = image.shape();
    let raw_data: Vec<f32> = image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("CT tensor must contain f32 data")
        .to_vec();

    // Downsample to ≈26×32×32 with stride = 16.
    let (ds_data, ds_dims) = downsample_stride(&raw_data, full_shape, 16);

    // Minmax-normalize to [0, 1] so NCC is interpretable against the [−1, 1] range.
    let fixed = normalize_minmax(&ds_data);

    // Construct moving: cyclic 2-voxel shift in x.
    // new[z, y, x] = old[z, y, (x + 2) % nx]
    let [nz, ny, nx] = ds_dims;
    let mut moving = Vec::with_capacity(fixed.len());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let src_x = (ix + 2) % nx;
                moving.push(fixed[iz * ny * nx + iy * nx + src_x]);
            }
        }
    }

    // NCC before registration: Pearson correlation of fixed vs 2-voxel-shifted moving.
    let ncc_before = ncc(&fixed, &moving);

    // Unit voxel spacing for the downsampled domain; control spacing is expressed
    // in voxels and the NCC metric is dimensionless.
    let unit_spacing = [1.0_f64, 1.0, 1.0];

    let config = BSplineFFDConfig {
        initial_control_spacing: [4, 4, 4],
        num_levels: 1,
        max_iterations_per_level: 50,
        learning_rate: 1.0,
        regularization_weight: 1e-3,
        convergence_threshold: 1e-5,
    };

    let result = BSplineFFDRegistration::register(&fixed, &moving, ds_dims, unit_spacing, &config)
        .expect("BSpline FFD registration must succeed on downsampled CT volume");

    let ncc_after = result.final_metric;

    assert!(
        ncc_after > ncc_before - 0.001,
        "BSpline FFD must not degrade NCC: ncc_before={:.6}, ncc_after={:.6}",
        ncc_before,
        ncc_after
    );
    assert!(
        ncc_after >= 0.80,
        "NCC after recovering 2-voxel x-shift must be ≥ 0.80, got {:.6} \
         (ncc_before={:.6})",
        ncc_after,
        ncc_before
    );
}

/// # Specification
///
/// CT (Hounsfield units) and T2-weighted MRI (signal intensity) from the same
/// porcine phantom must exhibit different intensity distributions:
///
/// 1. CT raw range `(max − min)` must exceed 100 HU (air→soft-tissue span is
///    > 1000 HU; even highly downsampled this bound holds).
/// 2. CT mean and MRI mean (in their native intensity scales) must differ by
///    more than 1.0 unit (different acquisition physics produce different
///    numeric ranges).
/// 3. The Pearson NCC between their minmax-normalized, downsampled, co-truncated
///    flat arrays must be `< 0.95` — confirming multi-modal contrast difference
///    even in gross spatial structure.
///
/// # Reference
/// Ger et al., Medical Physics 2018, DOI: 10.1002/mp.13090
#[test]
#[ignore = "requires test data"]
fn test_ct_mri_pair_intensity_statistics_differ() {
    let Some(root) = find_test_data_dir() else {
        eprintln!(
            "test_data directory not found; \
             skipping test_ct_mri_pair_intensity_statistics_differ"
        );
        return;
    };
    let ct_dir = root.join("3_head_ct_mridir").join("DICOM");
    let mri_dir = root.join("2_head_mri_t2").join("DICOM");
    if !ct_dir.exists() || !mri_dir.exists() {
        eprintln!(
            "CT path ({}) or MRI path ({}) not found; \
             skipping test_ct_mri_pair_intensity_statistics_differ",
            ct_dir.display(),
            mri_dir.display()
        );
        return;
    }

    let device = Default::default();
    let (ct_img, _ct_meta) = read_dicom_series_with_metadata::<B, _>(&ct_dir, &device)
        .expect("CT DICOM series must load");
    let (mri_img, _mri_meta) = read_dicom_series_with_metadata::<B, _>(&mri_dir, &device)
        .expect("MRI DICOM series must load");

    let ct_raw: Vec<f32> = ct_img
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("CT tensor must contain f32 data")
        .to_vec();
    let mri_raw: Vec<f32> = mri_img
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("MRI tensor must contain f32 data")
        .to_vec();

    let ct_shape = ct_img.shape();
    let mri_shape = mri_img.shape();

    // Downsample both volumes with stride = 16 to reduce memory and runtime.
    let (ct_down, _ct_ds_dims) = downsample_stride(&ct_raw, ct_shape, 16);
    let (mri_down, _mri_ds_dims) = downsample_stride(&mri_raw, mri_shape, 16);

    // ── Statistics on raw (unnormalized) downsampled volumes ──────────────
    let ct_min = ct_down.iter().cloned().fold(f32::INFINITY, f32::min);
    let ct_max = ct_down.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ct_mean: f64 = ct_down.iter().map(|&v| v as f64).sum::<f64>() / ct_down.len() as f64;

    let mri_min = mri_down.iter().cloned().fold(f32::INFINITY, f32::min);
    let mri_max = mri_down.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mri_mean: f64 = mri_down.iter().map(|&v| v as f64).sum::<f64>() / mri_down.len() as f64;

    // Assertion 1: CT HU range exceeds 100 (air to soft tissue > 1000 HU in-vivo).
    assert!(
        (ct_max - ct_min) as f64 > 100.0,
        "CT range (max − min) must exceed 100 HU, got {:.2} (min={:.2}, max={:.2})",
        ct_max - ct_min,
        ct_min,
        ct_max
    );

    // Assertion 2: CT and MRI means differ by > 1.0 (different modality intensity scales).
    assert!(
        (ct_mean - mri_mean).abs() > 1.0,
        "CT mean ({:.4}) and MRI mean ({:.4}) must differ by > 1.0 \
         (different modality intensity scales)",
        ct_mean,
        mri_mean
    );

    // Assertion 3: Normalized cross-NCC of the two modalities must be < 0.95.
    // Volumes have different sizes after downsampling (CT: ≈26×32×32, MRI: ≈6×ny×nx);
    // truncate to the common prefix length — sufficient to distinguish modalities.
    let ct_norm = normalize_minmax(&ct_down);
    let mri_norm = normalize_minmax(&mri_down);
    let common_len = ct_norm.len().min(mri_norm.len());

    // Sanity: both volumes must contribute samples to the comparison.
    assert!(
        common_len >= 256,
        "common_len must be ≥ 256 for a meaningful cross-NCC, got {}",
        common_len
    );

    let cross_ncc = ncc(&ct_norm[..common_len], &mri_norm[..common_len]);
    assert!(
        cross_ncc < 0.95,
        "normalized cross-NCC of CT and MRI must be < 0.95 \
         (different modalities, different intensity distributions), got {:.6}",
        cross_ncc
    );

    // Diagnostic output (visible only when running with --nocapture).
    eprintln!(
        "CT:  min={:.2}, max={:.2}, mean={:.4}, range={:.2}",
        ct_min,
        ct_max,
        ct_mean,
        ct_max - ct_min
    );
    eprintln!(
        "MRI: min={:.2}, max={:.2}, mean={:.4}",
        mri_min, mri_max, mri_mean
    );
    eprintln!("cross_ncc (common_len={}): {:.6}", common_len, cross_ncc);
}
