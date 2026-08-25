//! Synthetic diffusion phantom with known ground truth.
//!
//! Generates a small 3-D DWI volume with multiple tissue compartments,
//! a multi-shell gradient scheme, and analytically-computed ground-truth
//! maps (FA, MD, PEV, fODF peaks).  Every integration test runs against
//! known-oracle data so regressions are caught by value-level assertions.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;

// ═══════════════════════════════════════════════════════════════════════════
// Gradient scheme
// ═══════════════════════════════════════════════════════════════════════════

/// Build a multi-shell gradient scheme: 4 b0 + 30 dirs @ b=1000 + 60 dirs @ b=3000.
pub fn multi_shell_scheme() -> GradientScheme {
    let mut entries = Vec::with_capacity(94);
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let b1000 = DiffusionWeighting::from_seconds_per_square_millimeter(1_000.0).unwrap();
    let b3000 = DiffusionWeighting::from_seconds_per_square_millimeter(3_000.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);

    for _ in 0..4 {
        entries.push(GradientDirection::new(b0, zero).unwrap());
    }

    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..30 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / 30.0;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * i as f64;
        entries.push(
            GradientDirection::new(
                b1000,
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .unwrap(),
        );
    }
    for i in 0..60 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / 60.0;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * i as f64;
        entries.push(
            GradientDirection::new(
                b3000,
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .unwrap(),
        );
    }

    GradientScheme::new(entries, GradientFrame::Lps).unwrap()
}

/// Return a single-shell b=1000-only scheme: 4 b0 + 30 @ b=1000.
pub fn single_shell_scheme() -> GradientScheme {
    let mut entries = Vec::with_capacity(34);
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let b1000 = DiffusionWeighting::from_seconds_per_square_millimeter(1_000.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    for _ in 0..4 {
        entries.push(GradientDirection::new(b0, zero).unwrap());
    }
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..30 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / 30.0;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * i as f64;
        entries.push(
            GradientDirection::new(
                b1000,
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .unwrap(),
        );
    }
    GradientScheme::new(entries, GradientFrame::Lps).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// Phantom volume
// ═══════════════════════════════════════════════════════════════════════════

/// Tissue compartment labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tissue {
    /// Single fibre along +x.  Prolate: ad=0.0017, rd=0.0003.
    Horizontal = 1,
    /// Single fibre along +z.  Prolate: ad=0.0017, rd=0.0003.
    Vertical = 2,
    /// Two crossing fibres at ±30° from +x in the x-y plane.  Signal is
    /// (S₁ + S₂)/2 from two independent prolate tensors sharing the same
    /// ad/rd.
    Crossing = 3,
    /// CSF — isotropic, d=0.0030.
    Csf = 4,
    /// Gray matter — isotropic, d=0.0008.
    Gray = 5,
}

/// A 4×4×4 phantom volume with known scalar ground truth.
pub struct Phantom {
    /// Voxel data: flat [voxel * n_vol + vol].
    pub dwi: Vec<f64>,
    /// Shape `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// Tissue label per voxel (flat, z-major).
    pub labels: Vec<Tissue>,
    /// Ground-truth FA per voxel.
    pub fa_gt: Vec<f64>,
    /// Ground-truth MD per voxel.
    pub md_gt: Vec<f64>,
    /// Ground-truth PEV per voxel: flat `[x, y, z, …]`.
    // Computed but not yet asserted on. Kept rather than deleted so the
    // assertions can be written without recomputing the phantom; the gap
    // is tracked in atlas backlog.md#atlas-ritk-land-1.
    // allow, not expect: this file is compiled into two test binaries and the
    // fields are read in one of them, so an expectation would be unfulfilled
    // there.
    #[allow(dead_code, reason = "ratchet RITK-LINT-1")]
    pub pev_gt: Vec<f64>,
    /// Ground-truth fibre directions per voxel.
    // allow, not expect: this file is compiled into two test binaries and the
    // fields are read in one of them, so an expectation would be unfulfilled
    // there.
    #[allow(dead_code, reason = "ratchet RITK-LINT-1")]
    pub fibre_dirs_gt: Vec<Vec<[f64; 3]>>,
}

impl Default for Phantom {
    fn default() -> Self {
        Self::new()
    }
}

impl Phantom {
    /// Build the 4×4×4 multi-shell phantom.
    pub fn new() -> Self {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let n_vox = nx * ny * nz;
        let scheme = multi_shell_scheme();
        let n_vol = scheme.len();

        let mut labels = vec![Tissue::Gray; n_vox];
        let mut fa_gt = vec![0.0; n_vox];
        let mut md_gt = vec![0.0; n_vox];
        let mut pev_gt = vec![0.0_f64; n_vox];
        let mut fibre_dirs_gt: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_vox];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let (label, fa, md) = tissue_properties(ix, iy);
                    labels[idx] = label;
                    fa_gt[idx] = fa;
                    md_gt[idx] = md;
                    // The same per-tissue orientations the DWI generator
                    // applies, recorded so downstream assertions can compare
                    // estimated directions against them without recomputing.
                    match label {
                        Tissue::Horizontal => {
                            pev_gt[idx] = 1.0;
                            fibre_dirs_gt[idx] = vec![[1.0, 0.0, 0.0]];
                        }
                        Tissue::Vertical => {
                            pev_gt[idx] = 1.0;
                            fibre_dirs_gt[idx] = vec![[0.0, 0.0, 1.0]];
                        }
                        Tissue::Crossing => {
                            pev_gt[idx] = c30();
                            fibre_dirs_gt[idx] = vec![[c30(), s30(), 0.0], [c30(), -s30(), 0.0]];
                        }
                        Tissue::Csf | Tissue::Gray => {}
                    }
                }
            }
        }

        // Generate DWI signals.
        let mut dwi = vec![0.0; n_vox * n_vol];
        for (vox, &label) in labels.iter().enumerate() {
            let base = vox * n_vol;
            for (vol, entry) in scheme.directions().iter().enumerate() {
                let b = entry.weighting().seconds_per_square_millimeter();
                let [gx, gy, gz] = entry.direction().to_array();
                let signal = if b == 0.0 {
                    1000.0
                } else {
                    match label {
                        Tissue::Horizontal => {
                            let adc = adc_prolate(0.0017, 0.0003, [1.0, 0.0, 0.0], [gx, gy, gz]);
                            1000.0 * (-b * adc).exp()
                        }
                        Tissue::Vertical => {
                            let adc = adc_prolate(0.0017, 0.0003, [0.0, 0.0, 1.0], [gx, gy, gz]);
                            1000.0 * (-b * adc).exp()
                        }
                        Tissue::Crossing => {
                            let adc1 =
                                adc_prolate(0.0017, 0.0003, [c30(), s30(), 0.0], [gx, gy, gz]);
                            let adc2 =
                                adc_prolate(0.0017, 0.0003, [c30(), -s30(), 0.0], [gx, gy, gz]);
                            500.0 * (-b * adc1).exp() + 500.0 * (-b * adc2).exp()
                        }
                        Tissue::Csf => 1000.0 * (-b * 0.0030).exp(),
                        Tissue::Gray => 1000.0 * (-b * 0.0008).exp(),
                    }
                };
                dwi[base + vol] = signal;
            }
        }

        Self {
            dwi,
            shape: [nx, ny, nz],
            labels,
            fa_gt,
            md_gt,
            pev_gt,
            fibre_dirs_gt,
        }
    }

    /// Number of voxels.
    pub fn n_voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// Number of volumes.
    pub fn n_volumes(&self) -> usize {
        multi_shell_scheme().len()
    }

    /// Return the DWI signal vector for a single voxel.
    pub fn voxel_signals(&self, vox: usize) -> &[f64] {
        let nv = self.n_volumes();
        &self.dwi[vox * nv..(vox + 1) * nv]
    }

    /// Voxel index from (x, y, z).
    pub fn voxel_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.shape[1] * self.shape[0] + iy * self.shape[0] + ix
    }
}

// ── Tissue layout ─────────────────────────────────────────────────────────────

/// cos(30°), sin(30°) for the crossing-fibre directions.
fn c30() -> f64 {
    3.0_f64.sqrt() / 2.0
}
fn s30() -> f64 {
    0.5
}

fn tissue_properties(ix: usize, iy: usize) -> (Tissue, f64, f64) {
    // Layout (all z-slices identical):
    //   y=3  V | V | G | G
    //   y=2  V | V | G | G
    //   y=1  H | H | X | C
    //   y=0  H | H | C | C
    //        x=0  1   2   3

    let horizontal_fa = fa_prolate(0.0017, 0.0003);
    let horizontal_md = (0.0017 + 2.0 * 0.0003) / 3.0;
    let vertical_fa = fa_prolate(0.0017, 0.0003);
    let vertical_md = (0.0017 + 2.0 * 0.0003) / 3.0;
    // Crossing: averaged tensor has FA lower than single fibre.
    let crossing_avg = average_two_tensors(
        &prolate_tensor(0.0017, 0.0003, c30(), s30(), 0.0),
        &prolate_tensor(0.0017, 0.0003, c30(), -s30(), 0.0),
    );
    let crossing_fa = tensor_fa(&crossing_avg);
    let crossing_md = tensor_md(&crossing_avg);

    if ix < 2 && iy < 2 {
        (Tissue::Horizontal, horizontal_fa, horizontal_md)
    } else if ix < 2 {
        (Tissue::Vertical, vertical_fa, vertical_md)
    } else if iy < 2 && ix == 2 {
        (Tissue::Crossing, crossing_fa, crossing_md)
    } else if (ix == 3 && iy < 2) || (ix == 2 && iy >= 2) {
        (Tissue::Csf, 0.0, 0.0030)
    } else {
        (Tissue::Gray, 0.0, 0.0008)
    }
}

// ── Signal helpers ────────────────────────────────────────────────────────────

/// ADC `gᵀ D g` for a prolate tensor with principal direction `(dx, dy, dz)`.
fn adc_prolate(ad: f64, rd: f64, direction: [f64; 3], gradient: [f64; 3]) -> f64 {
    let [dx, dy, dz] = direction;
    let [gx, gy, gz] = gradient;
    let t = prolate_tensor(ad, rd, dx, dy, dz);
    tensor_adc(&t, gx, gy, gz)
}

fn prolate_tensor(ad: f64, rd: f64, dx: f64, dy: f64, dz: f64) -> [f64; 6] {
    [
        ad * dx * dx + rd * (1.0 - dx * dx),
        ad * dy * dy + rd * (1.0 - dy * dy),
        ad * dz * dz + rd * (1.0 - dz * dz),
        (ad - rd) * dx * dy,
        (ad - rd) * dx * dz,
        (ad - rd) * dy * dz,
    ]
}

/// `gᵀ D g` from Voigt elements [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
fn tensor_adc(elements: &[f64; 6], gx: f64, gy: f64, gz: f64) -> f64 {
    elements[0] * gx * gx
        + elements[1] * gy * gy
        + elements[2] * gz * gz
        + 2.0 * elements[3] * gx * gy
        + 2.0 * elements[4] * gx * gz
        + 2.0 * elements[5] * gy * gz
}

fn average_two_tensors(a: &[f64; 6], b: &[f64; 6]) -> [f64; 6] {
    [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
        (a[3] + b[3]) / 2.0,
        (a[4] + b[4]) / 2.0,
        (a[5] + b[5]) / 2.0,
    ]
}

/// FA of a prolate (or any) tensor: `√(3/2) · ‖D − MD·I‖_F / ‖D‖_F`.
fn tensor_fa(elements: &[f64; 6]) -> f64 {
    fa_prolate_with_trace(
        elements[0],
        elements[1],
        elements[2],
        elements[3],
        elements[4],
        elements[5],
    )
}

fn tensor_md(elements: &[f64; 6]) -> f64 {
    (elements[0] + elements[1] + elements[2]) / 3.0
}

fn fa_prolate(ad: f64, rd: f64) -> f64 {
    // Eigenvalues: λ₁ = ad, λ₂ = λ₃ = rd.
    let md = (ad + 2.0 * rd) / 3.0;
    let num = ((ad - md).powi(2) + 2.0 * (rd - md).powi(2)).sqrt();
    let den = (ad.powi(2) + 2.0 * rd.powi(2)).sqrt();
    if den < 1e-30 {
        return 0.0;
    }
    (3.0_f64 / 2.0).sqrt() * num / den
}

fn fa_prolate_with_trace(dxx: f64, dyy: f64, dzz: f64, dxy: f64, dxz: f64, dyz: f64) -> f64 {
    let md = (dxx + dyy + dzz) / 3.0;
    let dx = dxx - md;
    let dy = dyy - md;
    let dz = dzz - md;
    let num = (dx * dx + dy * dy + dz * dz + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz)).sqrt();
    let den =
        (dxx * dxx + dyy * dyy + dzz * dzz + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz)).sqrt();
    if den < 1e-30 {
        0.0
    } else {
        (3.0_f64 / 2.0).sqrt() * num / den
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Signal extraction helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Extract b ≤ 1500 signals from a full multi-shell signal vector by
/// matching against a b1000-only scheme's direction list.
///
/// Both schemes must use the same gradient-direction ordering for their
/// first N entries (the single-shell scheme is a prefix of the multi-shell
/// scheme).  This coupling is documented and verified in the integration
/// tests.
pub fn extract_b1000_signals(full_signals: &[f64], b1000_scheme: &GradientScheme) -> Vec<f64> {
    full_signals
        .iter()
        .zip(b1000_scheme.directions().iter())
        .map(|(&s, _)| s)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Rician noise
// ═══════════════════════════════════════════════════════════════════════════

/// Add Rician noise to a signal vector at a given signal-to-noise ratio.
///
/// MRI magnitude images follow a Rician distribution: the measured magnitude
/// `M = √((S + n_r)² + n_i²)` where `n_r, n_i ~ N(0, σ²)`.  SNR is defined
/// as `S₀ / σ` where `S₀` is the baseline (b0) signal.
///
/// Uses a linear congruential generator seeded by `voxel_seed` for
/// deterministic, reproducible noise.
///
/// # Panics
///
/// Panics if `snr` is not finite and strictly positive.
pub fn add_rician_noise(signals: &[f64], snr: f64, voxel_seed: u64) -> Vec<f64> {
    assert!(
        snr.is_finite() && snr > 0.0,
        "SNR must be finite and positive"
    );

    let baseline = signals[0]; // b0 volume is always first.
    let sigma = baseline / snr;
    let mut state = voxel_seed;

    signals
        .iter()
        .map(|&s| {
            // Box-Muller pair: two independent N(0,1) draws from two
            // consecutive LCG states.
            let ((nr_norm, ni_norm), next_state) = box_muller_pair(state);
            state = next_state;
            let nr = nr_norm * sigma;
            let ni = ni_norm * sigma;
            let magnitude = ((s + nr).powi(2) + ni.powi(2)).sqrt();
            // Clamp to a small positive floor so log-domain models don't
            // encounter -inf or NaN.
            magnitude.max(1e-6)
        })
        .collect()
}

/// Box-Muller transform producing a pair of independent N(0,1) samples.
///
/// Advances the 64-bit LCG twice to obtain two independent U(0,1) variates,
/// then transforms them via the Box-Muller formula.  Returns both samples
/// and the updated LCG state.
fn box_muller_pair(state: u64) -> ((f64, f64), u64) {
    let s1 = lcg_step(state);
    let u1 = (s1 as f64) / (u64::MAX as f64);
    let s2 = lcg_step(s1);
    let u2 = (s2 as f64) / (u64::MAX as f64);
    let r = (-2.0 * u1.max(1e-15).ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    ((r * theta.cos(), r * theta.sin()), s2)
}

/// Single step of a 64-bit linear congruential generator.
fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}
