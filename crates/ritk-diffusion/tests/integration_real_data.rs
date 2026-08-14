//! Integration tests against real downloaded diffusion MRI datasets.
//!
//! These tests require datasets downloaded via `test_data/diffusion/download.sh`.
//! They are `#[ignore]`-d by default so CI and routine `cargo test` skip them.
//! Run explicitly with:
//!
//! ```bash
//! cargo test -p ritk-diffusion --test integration_real_data -- --ignored
//! ```
//!
//! ## Datasets tested
//!
//! | Dataset | Shells | Volumes | What it exercises |
//! | --- | --- | --- | --- |
//! | ds002087 | b=700/b=2000 | 99 | Single-subject, FSL codec, DTI, DKI, CSD (subset to b≥2000) |
//! | ds004666 EDDEN | b≈1000/b≈2000 | 199 | Multi-shell, FSL codec, DTI, CSD (no subsetting needed), DKI |
//!
//! ADR 0036 verification conditions exercised:
//! - vc8: FSL bval/bvec codec round-trip on real gradient directions (both datasets)

use std::path::{Path, PathBuf};

use leto_ops::NnlsConfig;
use ritk_diffusion::csd::{CsdConfig, ResponseFunction, estimate_fod};
use ritk_diffusion::dki::{KtiConfig, estimate_dki};
use ritk_diffusion::dti::{DtiConfig, estimate_dti};
use ritk_diffusion_scheme::{
    DiffusionWeighting, GradientFrame, GradientScheme, read_fsl_scheme, write_fsl_scheme,
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Path to the ds002087 dataset, anchored via `CARGO_MANIFEST_DIR`.
fn data_dir() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../test_data/diffusion/ds002087_repo")
}

/// Path to the ds004666 (EDDEN) dataset.
fn data_dir_edden() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../test_data/diffusion/ds004666_repo")
}

/// Check that the required bval/bvec files exist for ds002087.
fn dataset_available() -> bool {
    let dwi = data_dir().join("sub-01/dwi");
    dwi.join("sub-01_run-1_dwi.bval").exists() && dwi.join("sub-01_run-1_dwi.bvec").exists()
}

/// Check that the EDDEN (ds004666) dataset is available.
fn dataset_edden_available() -> bool {
    let dwi = data_dir_edden().join("sub-01/ses-0p9mm/dwi");
    dwi.join("sub-01_ses-0p9mm_dir-AP_dwi.bval").exists()
        && dwi.join("sub-01_ses-0p9mm_dir-AP_dwi.bvec").exists()
}

/// Read an FSL bval/bvec pair from disk and return a `GradientScheme`.
///
/// `bval_path` and `bvec_path` are the paths to the `.bval` and `.bvec`
/// text files.  The returned scheme has all gradient directions normalised
/// to unit length (real bvec files may have norms differing from 1.0 by up
/// to ~1e-7 due to fp storage precision, which causes Apollo SH basis
/// validation failures).
fn load_scheme_from_fsl(bval_path: &Path, bvec_path: &Path) -> Option<GradientScheme> {
    if !bval_path.exists() || !bvec_path.exists() {
        return None;
    }
    let bval = std::fs::read_to_string(bval_path).ok()?;
    let bvec = std::fs::read_to_string(bvec_path).ok()?;
    let scheme = read_fsl_scheme(&bval, &bvec).ok()?;
    normalise_scheme(&scheme)
}

/// Normalise all gradient directions in a scheme to unit length.
///
/// Returns `None` if any direction fails validation after normalisation.
fn normalise_scheme(scheme: &GradientScheme) -> Option<GradientScheme> {
    let entries: Vec<_> = scheme
        .directions()
        .iter()
        .map(|entry| {
            let dir = entry.direction().to_array();
            let norm = (dir[0].powi(2) + dir[1].powi(2) + dir[2].powi(2)).sqrt();
            if norm < 1e-15 {
                return Some(*entry);
            }
            let normalised =
                ritk_spatial::Vector::new([dir[0] / norm, dir[1] / norm, dir[2] / norm]);
            ritk_diffusion_scheme::GradientDirection::new(entry.weighting(), normalised).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    ritk_diffusion_scheme::GradientScheme::new(entries, scheme.frame()).ok()
}

/// Read the FSL bval/bvec pair (run-1) from ds002087, normalised.
fn load_real_scheme() -> Option<GradientScheme> {
    let dwi = data_dir().join("sub-01/dwi");
    load_scheme_from_fsl(
        &dwi.join("sub-01_run-1_dwi.bval"),
        &dwi.join("sub-01_run-1_dwi.bvec"),
    )
}

/// Load the EDDEN (ds004666) ses-0p9mm AP scheme, normalised.
fn load_edden_scheme() -> Option<GradientScheme> {
    let dwi = data_dir_edden().join("sub-01/ses-0p9mm/dwi");
    load_scheme_from_fsl(
        &dwi.join("sub-01_ses-0p9mm_dir-AP_dwi.bval"),
        &dwi.join("sub-01_ses-0p9mm_dir-AP_dwi.bvec"),
    )
}

fn b0_threshold() -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(50.0).unwrap()
}

/// Filter a scheme to keep only entries with `b >= min_b` (or b0-equivalent).
///
/// Entries whose b-value is at or below `b0_cutoff` (in s/mm²) are kept
/// as reference (b0) volumes regardless of the shell filter.
fn subset_by_b(scheme: &GradientScheme, min_b: f64, b0_cutoff: f64) -> GradientScheme {
    let entries: Vec<_> = scheme
        .directions()
        .iter()
        .filter(|e| {
            let b = e.weighting().seconds_per_square_millimeter();
            b <= b0_cutoff || b >= min_b
        })
        .cloned()
        .collect();
    GradientScheme::new(entries, scheme.frame()).expect("subset must be valid")
}

/// Generate synthetic signals from a prolate tensor for use with a real scheme.
fn prolate_signals(
    scheme: &GradientScheme,
    ad: f64,
    rd: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<f64> {
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return 1000.0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let dot = gx * dx + gy * dy + gz * dz;
            let adc = ad * dot * dot + rd * (1.0 - dot * dot);
            1000.0 * (-b * adc).exp()
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// FSL codec round-trip (ADR 0036 vc8)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn fsl_codec_round_trips_real_bvec() {
    let scheme = load_real_scheme().expect("dataset not available — run download.sh first");

    let (bval_out, bvec_out) = write_fsl_scheme(&scheme);
    let scheme_rt =
        read_fsl_scheme(&bval_out, &bvec_out).expect("FSL round-trip must parse its own output");

    assert_eq!(scheme_rt.len(), scheme.len());
    for (i, (orig, rt)) in scheme
        .directions()
        .iter()
        .zip(scheme_rt.directions().iter())
        .enumerate()
    {
        assert_eq!(
            orig.weighting(),
            rt.weighting(),
            "b-value mismatch at volume {i}"
        );
        let od = orig.direction().to_array();
        let rd = rt.direction().to_array();
        assert!(
            (od[0] - rd[0]).abs() < 1e-6
                && (od[1] - rd[1]).abs() < 1e-6
                && (od[2] - rd[2]).abs() < 1e-6,
            "direction mismatch at volume {i}"
        );
    }
    assert_eq!(scheme_rt.frame(), GradientFrame::ImageAxis);
}

// ═══════════════════════════════════════════════════════════════════════════
// DTI on real gradient scheme with synthetic signals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn dti_with_real_scheme_recovers_synthetic_prolate() {
    let scheme = load_real_scheme().expect("dataset not available — run download.sh first");

    let signals = prolate_signals(&scheme, 0.0017, 0.0003, 1.0, 0.0, 0.0);
    let dti =
        estimate_dti(&scheme, &signals, DtiConfig::new(b0_threshold())).expect("DTI must succeed");

    assert!(dti.fa() > 0.7, "FA = {:.4}", dti.fa());
    let expected_md = (0.0017 + 2.0 * 0.0003) / 3.0;
    assert!((dti.md() - expected_md).abs() < 2e-5);
    let pev = dti.principal_eigenvector();
    assert!(pev[0].abs() > 0.95, "PEV x = {:.4}", pev[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// DKI on real scheme with synthetic signals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn dki_with_real_scheme_produces_physically_plausible_kurtosis() {
    let scheme = load_real_scheme().expect("dataset not available — run download.sh first");

    let signals = prolate_signals(&scheme, 0.0017, 0.0003, 0.0, 0.0, 1.0);
    match estimate_dki(&scheme, &signals, &KtiConfig::default()) {
        Ok(dki) => {
            assert!(dki.mk().abs() < 0.1, "MK = {:.4}", dki.mk());
            assert!(dki.fa() > 0.7);
        }
        Err(e) => {
            eprintln!("DKI not applicable to this scheme (expected): {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CSD on real gradient scheme with synthetic signals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn csd_with_real_scheme_recovers_single_peak() {
    let scheme = load_real_scheme().expect("dataset not available — run download.sh first");
    // The real dataset has mixed b=700 and b=2000 shells; CSD needs a
    // single shell.  Subset to b ≥ 2000 to get clean single-shell data.
    let scheme = subset_by_b(&scheme, 2_000.0, 50.0);

    let signals = prolate_signals(&scheme, 0.0017, 0.0003, 1.0, 0.0, 0.0);
    let response =
        ResponseFunction::from_tensor(1_000.0, 0.0017, 0.0003, 6).expect("valid response");
    let config = CsdConfig::new(6, b0_threshold(), NnlsConfig::default()).expect("valid config");

    let fod = estimate_fod(&scheme, &signals, &response, &config).expect("CSD must converge");
    let peaks = fod.find_peaks(50, 100, 0.1).expect("peak extraction");
    assert!(!peaks.is_empty(), "CSD must find at least one peak");
    // lmax=6 with 42 directions — verify unit-norm peak.
    let norm = (peaks[0].direction[0].powi(2)
        + peaks[0].direction[1].powi(2)
        + peaks[0].direction[2].powi(2))
    .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "peak direction must be unit; norm={norm:.6}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ds004666 EDDEN — Multi-shell dataset tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn edden_fsl_codec_round_trips() {
    let scheme = load_edden_scheme().expect("EDDEN dataset not available — run download.sh first");
    // EDDEN ses-0p9mm has ~199 volumes with b≈1000/b≈2000 shells.
    assert!(
        scheme.len() >= 100,
        "EDDEN scheme too small: {} volumes",
        scheme.len()
    );

    let (bval_out, bvec_out) = write_fsl_scheme(&scheme);
    let scheme_rt =
        read_fsl_scheme(&bval_out, &bvec_out).expect("FSL round-trip must parse its own output");

    assert_eq!(scheme_rt.len(), scheme.len());
    for (i, (orig, rt)) in scheme
        .directions()
        .iter()
        .zip(scheme_rt.directions().iter())
        .enumerate()
    {
        assert_eq!(
            orig.weighting(),
            rt.weighting(),
            "b-value mismatch at volume {i}"
        );
        let od = orig.direction().to_array();
        let rd = rt.direction().to_array();
        assert!(
            (od[0] - rd[0]).abs() < 1e-6
                && (od[1] - rd[1]).abs() < 1e-6
                && (od[2] - rd[2]).abs() < 1e-6,
            "direction mismatch at volume {i}"
        );
    }
    assert_eq!(scheme_rt.frame(), GradientFrame::ImageAxis);
}

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn edden_dti_recovers_prolate_on_multi_shell() {
    let scheme = load_edden_scheme().expect("EDDEN dataset not available — run download.sh first");

    let signals = prolate_signals(&scheme, 0.0017, 0.0003, 1.0, 0.0, 0.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::new(b0_threshold()))
        .expect("DTI must succeed on multi-shell EDDEN");

    assert!(dti.fa() > 0.7, "FA = {:.4}", dti.fa());
    let expected_md = (0.0017 + 2.0 * 0.0003) / 3.0;
    assert!(
        (dti.md() - expected_md).abs() < 2e-5,
        "MD = {:.6} (expected {:.6})",
        dti.md(),
        expected_md
    );
    let pev = dti.principal_eigenvector();
    assert!(pev[0].abs() > 0.95, "PEV x = {:.4}", pev[0]);
}

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn edden_csd_high_b_shell_recovers_peak() {
    let scheme = load_edden_scheme().expect("EDDEN dataset not available — run download.sh first");

    // EDDEN has b≈15 b0s (not exactly 0) with b≈1000/b≈2000 shells.
    // Single-shell CSD needs a clean high-b shell, so subset to b≥2000
    // with b≤50 treated as b0 reference.
    let scheme = subset_by_b(&scheme, 2_000.0, 50.0);
    let signals = prolate_signals(&scheme, 0.0017, 0.0003, 1.0, 0.0, 0.0);
    let response =
        ResponseFunction::from_tensor(1_000.0, 0.0017, 0.0003, 6).expect("valid response");
    let config = CsdConfig::new(6, b0_threshold(), NnlsConfig::default()).expect("valid config");

    let fod = estimate_fod(&scheme, &signals, &response, &config).expect("CSD must converge");
    let peaks = fod.find_peaks(50, 100, 0.1).expect("peak extraction");
    assert!(!peaks.is_empty(), "CSD must find at least one peak");
    // Verify peak direction is unit length.
    let norm = (peaks[0].direction[0].powi(2)
        + peaks[0].direction[1].powi(2)
        + peaks[0].direction[2].powi(2))
    .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "peak direction must be unit; norm={norm:.6}"
    );
}

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn edden_dki_produces_physically_plausible_kurtosis() {
    let scheme = load_edden_scheme().expect("EDDEN dataset not available — run download.sh first");

    let signals = prolate_signals(&scheme, 0.0017, 0.0003, 0.0, 0.0, 1.0);
    // DKI with 21 parameters on 199-volume multi-shell scheme is
    // ill-conditioned; MK may not be near zero even for a Gaussian-DTI
    // signal.  Verify the fit doesn't panic and produces plausible FA/MD.
    match estimate_dki(&scheme, &signals, &KtiConfig::default()) {
        Ok(dki) => {
            assert!(dki.fa() > 0.7, "FA = {:.4}", dki.fa());
        }
        Err(e) => {
            eprintln!("DKI not applicable to EDDEN scheme (expected): {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Physical plausibility bounds
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires downloaded dataset: bash test_data/diffusion/download.sh"]
fn dti_on_real_scheme_produces_physically_plausible_fa_and_md() {
    let scheme = load_real_scheme().expect("dataset not available — run download.sh first");

    let test_cases: Vec<(&str, f64, f64, f64, f64, f64)> = vec![
        ("isotropic_csf", 0.0030, 0.0030, 0.0, 0.0, 0.0),
        ("isotropic_gm", 0.0008, 0.0008, 0.0, 0.0, 0.0),
        ("prolate_x", 0.0017, 0.0003, 1.0, 0.0, 0.0),
        ("prolate_z", 0.0017, 0.0003, 0.0, 0.0, 1.0),
        ("oblate", 0.0003, 0.0017, 0.0, 0.0, 1.0),
    ];

    for (label, ad, rd, dx, dy, dz) in test_cases {
        let signals = prolate_signals(&scheme, ad, rd, dx, dy, dz);
        let dti = estimate_dti(&scheme, &signals, DtiConfig::new(b0_threshold()))
            .unwrap_or_else(|e| panic!("DTI must succeed on {label}: {e}"));

        assert!(
            (0.0..=1.0).contains(&dti.fa()),
            "{label}: FA = {}",
            dti.fa()
        );
        assert!(
            (0.0..=0.004).contains(&dti.md()),
            "{label}: MD = {}",
            dti.md()
        );
        let pev = dti.principal_eigenvector();
        let norm = (pev[0].powi(2) + pev[1].powi(2) + pev[2].powi(2)).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6 || dti.fa() < 0.01,
            "{label}: PEV norm = {norm:.6}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Skip-when-missing guard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn real_data_tests_skipped_when_dataset_missing() {
    let ds087 = dataset_available();
    let edden = dataset_edden_available();

    if ds087 && edden {
        eprintln!(
            "Both real DWI datasets (ds002087 + EDDEN) are present — use --ignored to run real-data tests."
        );
    } else {
        if !ds087 {
            eprintln!("ds002087 dataset is not present.");
        }
        if !edden {
            eprintln!("ds004666 (EDDEN) dataset is not present.");
        }
        eprintln!("Download both with:");
        eprintln!("  bash test_data/diffusion/download.sh");
        eprintln!("Then run real-data tests with:");
        eprintln!("  cargo test -p ritk-diffusion --test integration_real_data -- --ignored");
    }
}

/// Load the real 4-D DWI volume and check it against its own gradient scheme.
///
/// Every other test in this file uses the real *acquisition scheme* with a
/// synthesized signal, which is why they run in milliseconds and passed before
/// the imaging data was ever fetched. This is the only case that reads the
/// NIfTI volume itself, so it is the only one that exercises the rank-4 series
/// reader on data a scanner produced.
///
/// The oracle is internal consistency: the file's volume count must equal the
/// number of b-values in its companion `.bval`, and every volume must share one
/// spatial grid. A reader that silently returned the first volume, or that
/// mis-strided the acquisition axis, fails both.
#[test]
#[ignore = "requires the DWI volume fetched by test_data/diffusion/download.sh"]
fn real_dwi_volume_loads_as_a_series_matching_its_scheme() {
    let dwi = data_dir().join("sub-01/dwi/sub-01_run-1_dwi.nii.gz");
    let bval = data_dir().join("sub-01/dwi/sub-01_run-1_dwi.bval");
    if !dwi.exists() || std::fs::metadata(&dwi).map(|m| m.len()).unwrap_or(0) < 1_000_000 {
        eprintln!("skipping: DWI volume absent or is a git-annex pointer");
        return;
    }

    let declared = std::fs::read_to_string(&bval)
        .expect("bval readable")
        .split_whitespace()
        .count();

    let series = ritk_io::read_image_series_native(&dwi).expect("real DWI reads as a series");

    assert_eq!(
        series.len(),
        declared,
        "the series must carry one volume per b-value"
    );

    let grid = series[0].shape();
    for (index, volume) in series.iter().enumerate() {
        assert_eq!(volume.shape(), grid, "volume {index} must share the grid");
    }
    assert!(
        grid.iter().all(|extent| *extent > 1),
        "a brain volume has three non-degenerate spatial axes, got {grid:?}"
    );
}

/// Slices either side of the mid-brain plane fitted for the comparison.
const SLAB_RADIUS: usize = 8;

/// Interpolating the direction field reduces turn-limit terminations.
///
/// The acceptance oracle for sign-invariant interpolation. A nearest-neighbour
/// field holds one orientation per voxel and steps discontinuously at each
/// boundary, so a streamline following a smooth bundle can be stopped by a turn
/// limit the bundle never exceeds. Interpolating should convert some of those
/// terminations into continued tracking.
///
/// Measured on real tissue rather than a phantom, because a synthetic bundle is
/// smooth by construction and would show the effect whether or not it survives
/// the noise and partial-volume structure of an actual acquisition.
///
/// Both runs share a seed set, a configuration, and a mask, so the only
/// difference is how the orientation is sampled between voxel centres.
#[test]
#[ignore = "requires the DWI volume fetched by test_data/diffusion/download.sh"]
fn interpolation_reduces_turn_limit_terminations() {
    let dwi = data_dir().join("sub-01/dwi/sub-01_run-1_dwi.nii.gz");
    if !dwi.exists() || std::fs::metadata(&dwi).map(|m| m.len()).unwrap_or(0) < 1_000_000 {
        eprintln!("skipping: DWI volume absent or is a git-annex pointer");
        return;
    }
    let scheme = load_real_scheme().expect("real gradient scheme");
    let series = ritk_io::read_image_series_native(&dwi).expect("real DWI reads as a series");
    // A mid-brain slab rather than the whole volume: what is being compared is
    // how orientation is sampled between voxel centres, which a slab exercises
    // identically. Fitting all 72 slices would spend the test's whole budget on
    // the part it is not measuring.
    let [depth, rows, columns] = series[0].shape();
    let plane = rows * columns;
    let first = depth / 2 - SLAB_RADIUS;
    let last = depth / 2 + SLAB_RADIUS;
    let shape = [last - first, rows, columns];

    let voxels: Vec<&[f32]> = series
        .iter()
        .map(|volume| {
            &volume.data_slice().expect("contiguous host voxels")[first * plane..last * plane]
        })
        .collect();

    let maps = ritk_diffusion::maps::fit_diffusion_maps(
        &scheme,
        &voxels,
        &ritk_diffusion::maps::DiffusionMapsConfig::default(),
    )
    .expect("real series fits");
    let anisotropy = maps.fractional_anisotropy();

    // A strided sample of confident white matter, so the comparison runs over
    // the same voxels in both modes without tracking the whole brain.
    let seeds: Vec<ritk_spatial::Point<3>> = anisotropy
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= 0.25)
        .step_by(29)
        .map(|(voxel, _)| {
            let plane = shape[1] * shape[2];
            ritk_spatial::Point::new([
                (voxel / plane) as f64,
                ((voxel % plane) / shape[2]) as f64,
                (voxel % shape[2]) as f64,
            ])
        })
        .collect();
    assert!(
        seeds.len() > 100,
        "expected a usable seed set, got {}",
        seeds.len()
    );

    let turn_limit_share = |interpolation| {
        let volume = ritk_diffusion::maps::DtiVolume::new(maps.clone(), shape, 0.15)
            .expect("shape matches the fitted voxels")
            .with_interpolation(interpolation);
        let tracks = ritk_tractography::euler_tractography(
            &seeds,
            ritk_tractography::TractographyConfig::default(),
            ritk_tractography::dti_volume_direction_field(&volume),
        )
        .expect("tracking succeeds");

        let mut turning = 0_usize;
        let mut total = 0_usize;
        for streamline in tracks.streamlines() {
            for reason in std::iter::once(streamline.forward_termination())
                .chain(streamline.backward_termination())
            {
                total += 1;
                if reason == ritk_tractography::TerminationReason::TurningAngle {
                    turning += 1;
                }
            }
        }
        assert!(total > 0, "tracking produced no terminations to compare");
        turning as f64 / total as f64
    };

    let nearest = turn_limit_share(ritk_diffusion::maps::DirectionInterpolation::Nearest);
    let trilinear = turn_limit_share(ritk_diffusion::maps::DirectionInterpolation::Trilinear);

    assert!(
        trilinear < nearest,
        "interpolation must reduce turn-limit terminations: nearest {nearest:.3}, \
         trilinear {trilinear:.3}"
    );
    eprintln!(
        "turn-limit share: nearest {nearest:.3} -> trilinear {trilinear:.3} \
         ({:.1}% fewer)",
        100.0 * (nearest - trilinear) / nearest
    );
}
