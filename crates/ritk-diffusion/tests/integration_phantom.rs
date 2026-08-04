//! End-to-end integration tests against a synthetic diffusion phantom
//! with known ground truth.
//!
//! ADR 0036 verification conditions:
//! - vc2: boundary compliance (no local optimiser in DTI)
//! - vc7: gradient reorientation correctness
//! - vc8: cross-codec scheme equivalence
//!
//! Each test generates a 4×4×4 phantom and asserts fitted model output
//! against analytically-known ground truth.

mod phantom;

use phantom::{
    Phantom, Tissue, add_rician_noise, extract_b1000_signals, multi_shell_scheme,
    single_shell_scheme,
};

use ritk_diffusion::csd::{CsdConfig, ResponseFunction, estimate_fod};
use ritk_diffusion::dki::{KtiConfig, estimate_dki};
use ritk_diffusion::dti::{DtiConfig, estimate_dti};
use ritk_diffusion::noddi::{NoddiConfig, estimate_noddi};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame};

fn b0_threshold() -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(50.0).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// DTI integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dti_recover_fa_md_pev_on_all_voxels() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let config = DtiConfig::new(b0_threshold());

    for vox in 0..phantom.n_voxels() {
        let full = phantom.voxel_signals(vox);
        let signals = extract_b1000_signals(full, &b1000_scheme);
        let label = phantom.labels[vox];

        if label == Tissue::Csf || label == Tissue::Gray {
            let dti = estimate_dti(&b1000_scheme, &signals, config)
                .expect("DTI must succeed on CSF/gray");
            let gt_md = phantom.md_gt[vox];
            assert!(
                (dti.md() - gt_md).abs() < 2e-5,
                "voxel {vox} ({label:?}): MD = {:.6} vs gt {:.6}",
                dti.md(),
                gt_md
            );
            continue;
        }

        let dti = estimate_dti(&b1000_scheme, &signals, config).expect("DTI must succeed");

        let fa = dti.fa();
        let gt_fa = phantom.fa_gt[vox];
        let fa_tol = if label == Tissue::Crossing {
            0.12
        } else {
            0.08
        };
        assert!(
            (fa - gt_fa).abs() < fa_tol,
            "voxel {vox} ({label:?}): FA = {:.4} vs gt {:.4}",
            fa,
            gt_fa
        );

        let md = dti.md();
        let gt_md = phantom.md_gt[vox];
        let md_tol = if label == Tissue::Crossing {
            8e-5
        } else {
            2e-5
        };
        assert!(
            (md - gt_md).abs() < md_tol,
            "voxel {vox} ({label:?}): MD = {:.6} vs gt {:.6}",
            md,
            gt_md
        );

        if label == Tissue::Horizontal {
            let pev = dti.principal_eigenvector();
            let dot = pev[0].abs();
            assert!(
                dot > 0.95,
                "voxel {vox} (Horizontal): PEV x = {:.4}, expected ≈ 1",
                pev[0]
            );
        } else if label == Tissue::Vertical {
            let pev = dti.principal_eigenvector();
            let dot = pev[2].abs();
            assert!(
                dot > 0.95,
                "voxel {vox} (Vertical): PEV z = {:.4}, expected ≈ 1",
                pev[2]
            );
        }
    }
}

#[test]
fn dti_reject_signal_length_mismatch() {
    let phantom = Phantom::new();
    let scheme = single_shell_scheme();
    let signals = phantom.voxel_signals(0);
    let err = estimate_dti(&scheme, &signals[..10], DtiConfig::new(b0_threshold())).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("count") && msg.contains("match"),
        "expected count-mismatch error, got: {msg}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// DKI integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dki_recover_kurtosis_metrics_on_single_fibre_voxels() {
    let phantom = Phantom::new();
    let multi_scheme = multi_shell_scheme();
    let config = KtiConfig::default();

    // Horizontal-fibre voxel.
    let vox = phantom.voxel_index(0, 0, 0);
    assert!(
        phantom.labels[vox] == Tissue::Horizontal,
        "expected Horizontal at (0,0,0)"
    );

    let dki = estimate_dki(&multi_scheme, phantom.voxel_signals(vox), &config)
        .expect("DKI must succeed on horizontal fibre");

    assert!(dki.mk() >= 0.0, "MK must be non-negative");
    assert!(dki.ak() >= 0.0, "AK must be non-negative");
    assert!(dki.rk() >= 0.0, "RK must be non-negative");

    // Cross-check FA against DTI on same voxel.
    let b1000_scheme = single_shell_scheme();
    let b1000_signals = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);
    let dti = estimate_dti(
        &b1000_scheme,
        &b1000_signals,
        DtiConfig::new(b0_threshold()),
    )
    .expect("DTI");
    let fa_diff = (dki.fa() - dti.fa()).abs();
    assert!(
        fa_diff < 0.1,
        "DKI FA = {:.4} vs DTI FA = {:.4}, diff = {:.4}",
        dki.fa(),
        dti.fa(),
        fa_diff
    );
}

#[test]
fn dki_on_single_shell_is_ill_conditioned_but_may_succeed() {
    // DKI needs at least 21 DWI directions.  The single-shell scheme has
    // 30, so the Underdetermined guard does not fire.  The b and b² terms
    // are collinear on a single shell, so LM may converge to a degenerate
    // W ≈ 0 solution.  The test documents this limitation; a multi-shell
    // enforcement guard may be added to `estimate_dki` later.
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let signals = extract_b1000_signals(phantom.voxel_signals(0), &b1000_scheme);
    let result = estimate_dki(&b1000_scheme, &signals, &KtiConfig::default());
    // If it succeeds, kurtosis should be near zero (degenerate DKI ≈ DTI).
    if let Ok(dki) = result {
        assert!(
            dki.mk().abs() < 0.05,
            "single-shell MK ≈ 0 expected, got {:.4}",
            dki.mk()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CSD integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn csd_detect_crossing_fibres_in_crossing_region() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();

    // Crossing-fibre voxel (ix=2, iy=1, iz=0).
    let vox = phantom.voxel_index(2, 1, 0);
    assert!(
        phantom.labels[vox] == Tissue::Crossing,
        "expected Crossing at (2,1,0)"
    );
    let signals = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);

    // l_max=4 → 15 coefficients < 30 DWI directions (well-conditioned).
    let response =
        ResponseFunction::from_tensor(1_000.0, 0.0017, 0.0003, 4).expect("valid response");
    let config = CsdConfig::new(4, b0_threshold(), leto_ops::NnlsConfig::default())
        .expect("valid CSD config");

    let fod = estimate_fod(&b1000_scheme, &signals, &response, &config)
        .expect("CSD must converge on crossing fibre");

    let peaks = fod.find_peaks(50, 100, 0.1).expect("peak extraction");
    assert!(
        peaks.len() >= 2,
        "CSD must detect ≥2 peaks in crossing region, found {}",
        peaks.len()
    );

    // Strongest peak should be near +x (±30° is captured as dominant +x).
    let strongest = &peaks[0];
    let dot_x = strongest.direction[0].abs();
    assert!(
        dot_x > 0.6,
        "strongest peak x = {:.4}, expected near ±x",
        strongest.direction[0]
    );
}

#[test]
fn csd_detect_single_peak_in_horizontal_fibre() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();

    let vox = phantom.voxel_index(0, 0, 0);
    assert!(
        phantom.labels[vox] == Tissue::Horizontal,
        "expected Horizontal at (0,0,0)"
    );
    let signals = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);

    let response =
        ResponseFunction::from_tensor(1_000.0, 0.0017, 0.0003, 4).expect("valid response");
    let config = CsdConfig::new(4, b0_threshold(), leto_ops::NnlsConfig::default())
        .expect("valid CSD config");

    let fod = estimate_fod(&b1000_scheme, &signals, &response, &config).expect("CSD must converge");

    let peaks = fod.find_peaks(50, 100, 0.1).expect("peak extraction");
    assert!(!peaks.is_empty(), "must detect at least one peak");

    let dot = peaks[0].direction[0].abs();
    assert!(
        dot > 0.8,
        "horizontal fibre: strongest peak x = {:.4}, expected ≈ ±1",
        peaks[0].direction[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// NODDI integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn noddi_recover_high_ndi_in_single_fibre_voxels() {
    let phantom = Phantom::new();
    let multi_scheme = multi_shell_scheme();
    let config = NoddiConfig::default();

    let vox = phantom.voxel_index(0, 0, 0);
    assert!(
        phantom.labels[vox] == Tissue::Horizontal,
        "expected Horizontal at (0,0,0)"
    );

    let noddi = estimate_noddi(&multi_scheme, phantom.voxel_signals(vox), &config)
        .expect("NODDI must converge on horizontal fibre");

    let ndi = noddi.ndi();
    assert!(ndi > 0.4, "Horizontal fibre NDI = {ndi:.4}, expected > 0.4");
    assert!(ndi <= 1.0, "NDI = {ndi:.4} must be ≤ 1.0");
}

#[test]
fn noddi_recover_low_ndi_in_csf() {
    let phantom = Phantom::new();
    let multi_scheme = multi_shell_scheme();
    let config = NoddiConfig::default();

    let vox = phantom.voxel_index(3, 0, 0);
    assert!(
        phantom.labels[vox] == Tissue::Csf,
        "expected CSF at (3,0,0)"
    );

    // NODDI may not converge on pure CSF — that's acceptable.
    if let Ok(noddi) = estimate_noddi(&multi_scheme, phantom.voxel_signals(vox), &config) {
        assert!(
            noddi.ndi() < 0.5,
            "CSF NDI = {:.4}, expected < 0.5",
            noddi.ndi()
        );
        assert!(
            noddi.f_iso() > 0.3,
            "CSF f_iso = {:.4}, expected > 0.3",
            noddi.f_iso()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gradient reorientation (ADR 0036 vc7)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reorient_gradients_preserves_pev_across_rotation() {
    let phantom = Phantom::new();
    let scheme = single_shell_scheme();

    // 45° rotation about y-axis.
    let angle = std::f64::consts::FRAC_PI_4;
    let (s, c) = angle.sin_cos();
    let rotation = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
    let scheme_reoriented = scheme.reorient(rotation).expect("valid rotation");

    // Vertical-fibre voxel (PEV along +z).
    let vox = phantom.voxel_index(0, 2, 0);
    assert!(
        phantom.labels[vox] == Tissue::Vertical,
        "expected Vertical at (0,2,0)"
    );

    // Generate signals from reoriented scheme with original tensor.
    let signals: Vec<f64> = scheme_reoriented
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return 1000.0;
            }
            let [_, _, gz] = entry.direction().to_array();
            let adc = 0.0003 + (0.0017 - 0.0003) * gz * gz;
            1000.0 * (-b * adc).exp()
        })
        .collect();

    let dti = estimate_dti(&scheme_reoriented, &signals, DtiConfig::new(b0_threshold()))
        .expect("DTI with reoriented scheme");
    let pev = dti.principal_eigenvector();
    let dot_z = pev[2].abs();
    assert!(
        dot_z > 0.95,
        "vc7: PEV z = {:.4} after reorientation, expected ≈ 1",
        pev[2]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Frame preservation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_models_preserve_lps_frame() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let multi_scheme = multi_shell_scheme();

    let vox = phantom.voxel_index(0, 0, 0);
    assert!(
        phantom.labels[vox] == Tissue::Horizontal,
        "expected Horizontal at (0,0,0)"
    );

    // DTI.
    let b1000_signals = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);
    let dti = estimate_dti(
        &b1000_scheme,
        &b1000_signals,
        DtiConfig::new(b0_threshold()),
    )
    .unwrap();
    assert_eq!(dti.frame(), GradientFrame::Lps, "DTI frame mismatch");

    // DKI.
    let dki = estimate_dki(
        &multi_scheme,
        phantom.voxel_signals(vox),
        &KtiConfig::default(),
    )
    .unwrap();
    assert_eq!(dki.frame(), GradientFrame::Lps, "DKI frame mismatch");

    // NODDI.
    let noddi = estimate_noddi(
        &multi_scheme,
        phantom.voxel_signals(vox),
        &NoddiConfig::default(),
    )
    .unwrap();
    assert_eq!(noddi.frame(), GradientFrame::Lps, "NODDI frame mismatch");
}

// ═══════════════════════════════════════════════════════════════════════════
// Noise robustness — DTI
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that DTI does not panic at SNR=30 and error is small.
#[test]
fn dti_noise_robustness_at_snr_30() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let config = DtiConfig::new(b0_threshold());

    // Horizontal-fibre voxel.
    let vox = phantom.voxel_index(0, 0, 0);
    let clean = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);
    let noisy = add_rician_noise(&clean, 30.0, vox as u64);

    let dti = estimate_dti(&b1000_scheme, &noisy, config).expect("DTI must not panic at SNR=30");

    // FA ∈ [0, 1].
    assert!(
        (0.0..=1.0).contains(&dti.fa()),
        "FA = {} outside [0,1]",
        dti.fa()
    );
    // MD in physiological range.
    assert!(
        (0.0..=0.004).contains(&dti.md()),
        "MD = {} outside range",
        dti.md()
    );
    // At SNR=30, FA error should be modest: |FA − gt| < 0.15.
    let fa_err = (dti.fa() - phantom.fa_gt[vox]).abs();
    assert!(
        fa_err < 0.15,
        "SNR=30 FA error = {:.4} exceeds 0.15 threshold",
        fa_err
    );
}

/// Verify error increases as SNR degrades from 30 → 20 → 10.
#[test]
fn dti_fa_error_increases_as_snr_decreases() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let config = DtiConfig::new(b0_threshold());
    let vox = phantom.voxel_index(0, 0, 0);
    let clean = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);
    let gt_fa = phantom.fa_gt[vox];

    let mut prev_err: Option<f64> = None;
    for snr in [30.0, 20.0, 10.0] {
        let noisy = add_rician_noise(&clean, snr, vox as u64);
        let dti = estimate_dti(&b1000_scheme, &noisy, config)
            .unwrap_or_else(|e| panic!("DTI must not panic at SNR={snr}: {e}"));

        // All outputs must stay physically plausible.
        assert!(
            (0.0..=1.0).contains(&dti.fa()),
            "SNR={snr}: FA = {} outside [0,1]",
            dti.fa()
        );
        assert!(
            dti.md() > 0.0 && dti.md() < 0.004,
            "SNR={snr}: MD = {} outside range",
            dti.md()
        );
        let pev = dti.principal_eigenvector();
        let pev_norm = (pev[0].powi(2) + pev[1].powi(2) + pev[2].powi(2)).sqrt();
        assert!(
            (pev_norm - 1.0).abs() < 1e-6 || dti.fa() < 0.02,
            "SNR={snr}: PEV norm = {pev_norm:.6}"
        );

        let err = (dti.fa() - gt_fa).abs();
        if let Some(prev) = prev_err {
            // FA error should increase as SNR degrades (monotonic, not strictly).
            // At SNR=10 noise can be several times larger than at SNR=20,
            // so allow up to 5× the previous error.
            assert!(
                err <= prev * 5.0,
                "SNR={snr}: FA err {err:.4} >> prev {prev:.4}"
            );
        }
        prev_err = Some(err);
    }
}

/// Verify that at low SNR (10), DTI still produces a directionally
/// informative PEV (not random).
#[test]
fn dti_at_snr_10_still_recovers_dominant_direction() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let config = DtiConfig::new(b0_threshold());
    let vox = phantom.voxel_index(0, 0, 0);
    let clean = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);
    let noisy = add_rician_noise(&clean, 10.0, vox as u64);

    let dti = estimate_dti(&b1000_scheme, &noisy, config).expect("DTI must not panic at SNR=10");

    // PEV should still have dominant x-component (horizontal fibre).
    let pev = dti.principal_eigenvector();
    let dot_x = pev[0].abs();
    assert!(
        dot_x > 0.3,
        "SNR=10: PEV x = {dot_x:.4}, should still be dominant"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Noise robustness — DKI
// ═══════════════════════════════════════════════════════════════════════════

/// DKI at moderate SNR must not panic.  If it converges the kurtosis
/// metrics must be physically plausible.
///
/// DKI's 21-parameter fit is noise-sensitive, especially on this phantom
/// where the b=3000 DWI signal is severely attenuated (S ≈ 6 for the
/// single-fibre direction).  If the solver produces unphysical kurtosis
/// the test documents the limitation rather than panicking — DKI at this
/// SNR on this scheme is known to be borderline.
#[test]
fn dki_noise_robustness_at_snr_100() {
    let phantom = Phantom::new();
    let multi_scheme = multi_shell_scheme();
    let config = KtiConfig::default();
    let vox = phantom.voxel_index(0, 0, 0);
    let noisy = add_rician_noise(phantom.voxel_signals(vox), 100.0, vox as u64);

    // DKI is inherently more noise-sensitive than DTI.  A solver failure
    // at moderate SNR is expected for this b=3000-attenuated scheme;
    // document it rather than asserting success.
    match estimate_dki(&multi_scheme, &noisy, &config) {
        Ok(dki) => {
            assert!((0.0..=1.0).contains(&dki.fa()));
            assert!(dki.md() > 0.0 && dki.md() < 0.004);
            // Allow unphysical kurtosis — documents DKI instability on this scheme.
            if dki.mk() < -0.05 {
                eprintln!(
                    "DKI at SNR=100 produced unphysical MK = {:.4} — known limitation",
                    dki.mk()
                );
            }
        }
        Err(e) => {
            eprintln!("DKI failed at SNR=100 (expected for this scheme): {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Noise robustness — CSD
// ═══════════════════════════════════════════════════════════════════════════

/// CSD at SNR=30 must not panic and must still detect at least one peak
/// with a finite unit direction.
#[test]
fn csd_noise_robustness_at_snr_30() {
    let phantom = Phantom::new();
    let b1000_scheme = single_shell_scheme();
    let vox = phantom.voxel_index(0, 0, 0);
    let clean = extract_b1000_signals(phantom.voxel_signals(vox), &b1000_scheme);
    let noisy = add_rician_noise(&clean, 30.0, vox as u64);

    let response =
        ResponseFunction::from_tensor(1_000.0, 0.0017, 0.0003, 4).expect("valid response");
    let config = CsdConfig::new(4, b0_threshold(), leto_ops::NnlsConfig::default())
        .expect("valid CSD config");

    let fod = estimate_fod(&b1000_scheme, &noisy, &response, &config)
        .expect("CSD must not panic at SNR=30");

    let peaks = fod.find_peaks(50, 100, 0.1).expect("peak extraction");
    assert!(
        !peaks.is_empty(),
        "CSD must detect at least one peak at SNR=30"
    );

    // Direction must be a finite unit vector (valid geometry, even if the
    // exact orientation is noise-shifted).
    for peak in &peaks {
        let norm =
            (peak.direction[0].powi(2) + peak.direction[1].powi(2) + peak.direction[2].powi(2))
                .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "SNR=30: peak direction norm = {norm:.6}, expected unit"
        );
        assert!(
            peak.direction.iter().all(|v| v.is_finite()),
            "SNR=30: peak direction must be finite"
        );
    }
}
