//! IIR 1-D primitives for recursive Gaussian filtering.
//!
//! Contains the Deriche 4th-order coefficient computation (matching ITK
//! `RecursiveGaussianImageFilter`), 1-D line iteration helpers, the parallel
//! causal+anticausal smoothing pass, and the finite-difference derivative
//! kernels used by the recursive Gaussian filter.
//!
//! The smoothing recursion is accumulated in `f64` (ITK `RealType`) so the
//! interior is float-exact to SimpleITK `SmoothingRecursiveGaussian`; the
//! derivative (finite-difference) loops keep their boundary/interior split for
//! auto-vectorisation.

// ── Deriche coefficient set ────────────────────────────────────────────────────

// ITK Deriche pole constants (Farnebäck). Index [0]=zero-order, [2]=second-order.
const DERICHE_W1: f64 = 0.6681;
const DERICHE_L1: f64 = -1.3932;
const DERICHE_W2: f64 = 2.0787;
const DERICHE_L2: f64 = -1.3732;
const DERICHE_A1: [f64; 3] = [1.3530, -0.6724, -1.3563];
const DERICHE_B1: [f64; 3] = [1.8151, -3.4327, 5.2318];
const DERICHE_A2: [f64; 3] = [-0.3531, 0.6724, 0.3446];
const DERICHE_B2: [f64; 3] = [0.0902, 0.6100, -2.2355];

/// Denominator coefficients `[D1,D2,D3,D4]` plus the moment sums `(SD, DD, ED)`
/// used by the order normalisations (ITK `ComputeDCoefficients`).
fn deriche_d_coefficients(sigma: f64) -> ([f64; 4], f64, f64, f64) {
    let (cos1, exp1) = ((DERICHE_W1 / sigma).cos(), (DERICHE_L1 / sigma).exp());
    let (cos2, exp2) = ((DERICHE_W2 / sigma).cos(), (DERICHE_L2 / sigma).exp());
    let d4 = exp1 * exp1 * exp2 * exp2;
    let d3 = -2.0 * cos1 * exp1 * exp2 * exp2 - 2.0 * cos2 * exp2 * exp1 * exp1;
    let d2 = 4.0 * cos2 * cos1 * exp1 * exp2 + exp1 * exp1 + exp2 * exp2;
    let d1 = -2.0 * (exp2 * cos2 + exp1 * cos1);
    let sd = 1.0 + d1 + d2 + d3 + d4;
    let dd = d1 + 2.0 * d2 + 3.0 * d3 + 4.0 * d4;
    let ed = d1 + 4.0 * d2 + 9.0 * d3 + 16.0 * d4;
    ([d1, d2, d3, d4], sd, dd, ed)
}

/// Numerator coefficients `[N0,N1,N2,N3]` plus the moment sums `(SN, DN, EN)`
/// for one constant set (ITK `ComputeNCoefficients`).
fn deriche_n_coefficients(sigma: f64, idx: usize) -> ([f64; 4], f64, f64, f64) {
    let (a1, b1, a2, b2) = (
        DERICHE_A1[idx],
        DERICHE_B1[idx],
        DERICHE_A2[idx],
        DERICHE_B2[idx],
    );
    let (sin1, cos1, exp1) = (
        (DERICHE_W1 / sigma).sin(),
        (DERICHE_W1 / sigma).cos(),
        (DERICHE_L1 / sigma).exp(),
    );
    let (sin2, cos2, exp2) = (
        (DERICHE_W2 / sigma).sin(),
        (DERICHE_W2 / sigma).cos(),
        (DERICHE_L2 / sigma).exp(),
    );
    let n0 = a1 + a2;
    let n1 =
        exp2 * (b2 * sin2 - (a2 + 2.0 * a1) * cos2) + exp1 * (b1 * sin1 - (a1 + 2.0 * a2) * cos1);
    let n2 = 2.0 * exp1 * exp2 * ((a1 + a2) * cos2 * cos1 - b1 * cos2 * sin1 - b2 * cos1 * sin2)
        + a2 * exp1 * exp1
        + a1 * exp2 * exp2;
    let n3 =
        exp2 * exp1 * exp1 * (b2 * sin2 - a2 * cos2) + exp1 * exp2 * exp2 * (b1 * sin1 - a1 * cos1);
    let sn = n0 + n1 + n2 + n3;
    let dn = n1 + 2.0 * n2 + 3.0 * n3;
    let en = n1 + 4.0 * n2 + 9.0 * n3;
    ([n0, n1, n2, n3], sn, dn, en)
}

/// Precomputed Deriche 4th-order IIR coefficients for the recursive Gaussian (or
/// its derivatives), matching ITK `RecursiveGaussianImageFilter`.
///
/// The output is the SUM of a causal and an anticausal pass on the same input
/// (parallel, not cascaded):
/// causal:     `yc[n] = Σ_{k=0..3} n_k·x[n−k] − Σ_{k=1..4} d_k·yc[n−k]`
/// anticausal: `ya[n] = Σ_{k=1..4} m_k·x[n+k] − Σ_{k=1..4} d_k·ya[n+k]`
/// out\[n\] = yc\[n\] + ya\[n\].
pub(super) struct DericheCoefficients {
    /// Causal numerator coefficients n0..n3.
    pub(super) n: [f64; 4],
    /// Shared denominator (feedback) coefficients d1..d4.
    pub(super) d: [f64; 4],
    /// Anticausal numerator coefficients m1..m4.
    pub(super) m: [f64; 4],
}

impl DericheCoefficients {
    /// Symmetric anticausal coefficients (`M_k = N_k − D_k·N0`, `M4 = −D4·N0`),
    /// used by the even (zero/second) orders.
    fn symmetric(n: [f64; 4], d: [f64; 4]) -> Self {
        let m = [
            n[1] - d[0] * n[0],
            n[2] - d[1] * n[0],
            n[3] - d[2] * n[0],
            -d[3] * n[0],
        ];
        Self { n, d, m }
    }

    /// Antisymmetric anticausal coefficients (`M_k = −(N_k − D_k·N0)`,
    /// `M4 = +D4·N0`), used by the odd (first) order (ITK
    /// `ComputeRemainingCoefficients(symmetric=false)`).
    fn antisymmetric(n: [f64; 4], d: [f64; 4]) -> Self {
        let m = [
            -(n[1] - d[0] * n[0]),
            -(n[2] - d[1] * n[0]),
            -(n[3] - d[2] * n[0]),
            d[3] * n[0],
        ];
        Self { n, d, m }
    }

    /// Zero-order (smoothing) coefficients from a pixel-space sigma.
    ///
    /// DC-normalised by `alpha0 = 2·SN/SD − N0` (ITK `SetUp`, `ZeroOrder`).
    pub(super) fn from_sigma(sigma: f64) -> Self {
        let (d, sd, _dd, _ed) = deriche_d_coefficients(sigma);
        let (n, sn, _dn, _en) = deriche_n_coefficients(sigma, 0);
        let alpha0 = 2.0 * sn / sd - n[0];
        let n = n.map(|c| c / alpha0);
        Self::symmetric(n, d)
    }

    /// First-order (∂/∂x) coefficients from a pixel-space sigma (ITK `SetUp`,
    /// `FirstOrder`): normalised by `alpha1 = 2·(SN·DD − DN·SD)/SD²`, with the
    /// antisymmetric anticausal pass. Sign convention is ITK's `direction = +1`.
    pub(super) fn first_order(sigma: f64) -> Self {
        let (d, sd, dd, _ed) = deriche_d_coefficients(sigma);
        let (n, sn, dn, _en) = deriche_n_coefficients(sigma, 1);
        let alpha1 = 2.0 * (sn * dd - dn * sd) / (sd * sd);
        let n = n.map(|c| c / alpha1);
        Self::antisymmetric(n, d)
    }

    /// Second-order (∂²/∂x²) coefficients from a pixel-space sigma (ITK `SetUp`,
    /// `SecondOrder`): mixes the order-0 and order-2 constant sets via `beta`,
    /// then normalises by `alpha2 = (EN·SD² − ED·SN·SD − 2·DN·DD·SD + 2·DD²·SN)/SD³`.
    pub(super) fn second_order(sigma: f64) -> Self {
        let (d, sd, dd, ed) = deriche_d_coefficients(sigma);
        let (n0c, sn0, dn0, en0) = deriche_n_coefficients(sigma, 0);
        let (n2c, sn2, dn2, en2) = deriche_n_coefficients(sigma, 2);
        let beta = -(2.0 * sn2 - sd * n2c[0]) / (2.0 * sn0 - sd * n0c[0]);
        let n: [f64; 4] = std::array::from_fn(|i| n2c[i] + beta * n0c[i]);
        let sn = sn2 + beta * sn0;
        let dn = dn2 + beta * dn0;
        let en = en2 + beta * en0;
        let alpha2 = (en * sd * sd - ed * sn * sd - 2.0 * dn * dd * sd + 2.0 * dd * dd * sn)
            / (sd * sd * sd);
        let n = n.map(|c| c / alpha2);
        Self::symmetric(n, d)
    }
}

// ── 1-D line iteration helpers ────────────────────────────────────────────────

/// Information needed to iterate over a 1-D line within a 3-D volume along a
/// given dimension.
pub(super) struct LineParams {
    /// Length of each 1-D line along the target dimension.
    pub(super) len: usize,
    /// Number of independent lines (product of the other two dimensions).
    pub(super) num_lines: usize,
    /// Stride between consecutive elements along the target dimension.
    pub(super) stride: usize,
}

/// Compute the base offset (index of the first element) for the `line_idx`-th
/// line along dimension `dim` in a volume with shape `dims = [nz, ny, nx]`.
#[inline]
pub(super) fn line_base(dims: [usize; 3], dim: usize, line_idx: usize) -> usize {
    let [_nz, ny, nx] = dims;
    match dim {
        0 => {
            // Lines along Z. Each line is identified by (iy, ix).
            // line_idx = iy * nx + ix.  Base = iy*nx + ix.
            line_idx
        }
        1 => {
            // Lines along Y. Each line is identified by (iz, ix).
            // line_idx = iz * nx + ix.  Base = iz * ny * nx + ix.
            let iz = line_idx / nx;
            let ix = line_idx % nx;
            iz * ny * nx + ix
        }
        2 => {
            // Lines along X. Each line is identified by (iz, iy).
            // line_idx = iz * ny + iy.  Base = iz * ny * nx + iy * nx.
            let iz = line_idx / ny;
            let iy = line_idx % ny;
            iz * ny * nx + iy * nx
        }
        _ => unreachable!(),
    }
}

#[inline]
pub(super) fn line_params(dims: [usize; 3], dim: usize) -> LineParams {
    let [nz, ny, nx] = dims;
    match dim {
        0 => LineParams {
            len: nz,
            num_lines: ny * nx,
            stride: ny * nx,
        },
        1 => LineParams {
            len: ny,
            num_lines: nz * nx,
            stride: nx,
        },
        2 => LineParams {
            len: nx,
            num_lines: nz * ny,
            stride: 1,
        },
        _ => unreachable!(),
    }
}

// ── Deriche two-pass IIR (order-agnostic) ─────────────────────────────────────

/// Apply a Deriche 4th-order recursive pass along dimension `dim`. The `coeffs`
/// select the order (zero = smoothing, second = ∂²/∂x²); the recursion is
/// identical (ITK `RecursiveGaussianImageFilter`).
///
/// The output is the sum of a causal (forward) and an anticausal (backward)
/// pass over the SAME input:
///   `yc[n] = Σ n_k·x[n−k] − Σ d_k·yc[n−k]`  (k: n 0..3, d 1..4)
///   `ya[n] = Σ m_k·x[n+k] − Σ d_k·ya[n+k]`  (k: 1..4)
///   `out[n] = yc[n] + ya[n]`.
///
/// Boundary: constant (replicate) extension realised by padding each line with
/// `pad` edge-valued samples per side, where `pad` scales with `pixel_sigma`
/// so the IIR transient decays before reaching the real data. The recursion is
/// accumulated in `f64` to match ITK's `RealType` (float-exact to SimpleITK).
pub(super) fn apply_deriche_1d(
    data: &[f32],
    dims: [usize; 3],
    dim: usize,
    coeffs: &DericheCoefficients,
    pixel_sigma: f64,
) -> Vec<f32> {
    let lp = line_params(dims, dim);
    let mut output = vec![0.0_f32; data.len()];
    let len = lp.len;
    if len == 0 {
        return output;
    }

    // Pad enough that the dominant pole |exp(L/σ)| (L ≈ −1.37) decays below
    // ~1e-8 before the data starts: pad ≈ 13·σ (independent of the line length,
    // so short lines with large σ still settle).
    let pad = ((13.0 * pixel_sigma).ceil() as usize).max(8);
    let plen = len + 2 * pad;
    let [_nz, ny, nx] = dims;
    let nyx = ny * nx;

    // Each 1-D line is an independent recurrence writing disjoint output indices,
    // so any per-line schedule is bit-identical. For the X (dim 2) and Y (dim 1)
    // axes a whole Z-slice is a contiguous `nyx`-length output chunk, so the
    // work parallelises across Z with one scratch set per slice. The Z axis
    // (dim 0) has lines strided across the whole volume (no contiguous chunk),
    // so it stays serial.
    match dim {
        2 => {
            // X-lines: row `iy` within the slice is the contiguous span
            // `[iy*nx, iy*nx+nx)` (unit stride).
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut output,
                nyx,
                |slice_idx, out_slice| {
                    let in_slice = &data[slice_idx * nyx..slice_idx * nyx + nyx];
                    let mut scratch = LineScratch::new(plen);
                    for iy in 0..ny {
                        deriche_line(
                            in_slice, iy * nx, 1, out_slice, iy * nx, 1, len, coeffs, pad,
                            &mut scratch,
                        );
                    }
                },
            );
        }
        1 => {
            // Y-lines: column `ix` within the slice is `[ix, ix+nx, …]`
            // (stride `nx`, length `ny`).
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut output,
                nyx,
                |slice_idx, out_slice| {
                    let in_slice = &data[slice_idx * nyx..slice_idx * nyx + nyx];
                    let mut scratch = LineScratch::new(plen);
                    for ix in 0..nx {
                        deriche_line(
                            in_slice, ix, nx, out_slice, ix, nx, len, coeffs, pad, &mut scratch,
                        );
                    }
                },
            );
        }
        _ => {
            // Z-lines: strided by `nyx` across the whole buffer — serial.
            let mut scratch = LineScratch::new(plen);
            for li in 0..lp.num_lines {
                let base = line_base(dims, 0, li);
                deriche_line(
                    data, base, lp.stride, &mut output, base, lp.stride, len, coeffs, pad,
                    &mut scratch,
                );
            }
        }
    }
    output
}

/// Per-line `f64` scratch: padded input line plus the two recursion lines.
struct LineScratch {
    xp: Vec<f64>,
    yc: Vec<f64>,
    ya: Vec<f64>,
}

impl LineScratch {
    #[inline]
    fn new(plen: usize) -> Self {
        Self {
            xp: vec![0.0; plen],
            yc: vec![0.0; plen],
            ya: vec![0.0; plen],
        }
    }
}

/// Filter one 1-D line: read `input[in_off + i·in_stride]` for `i ∈ [0, len)`,
/// run the causal+anticausal Deriche recursion (edge-replicated padding), and
/// write the sum into `output[out_off + i·out_stride]`.
///
/// `scratch.{xp,yc,ya}` must be exactly `len + 2·pad` long. Arithmetic is
/// identical to the original single-loop form — only the line addressing is
/// parameterised — so results are bit-identical regardless of how lines are
/// scheduled across threads.
#[inline]
#[allow(clippy::too_many_arguments)]
fn deriche_line(
    input: &[f32],
    in_off: usize,
    in_stride: usize,
    output: &mut [f32],
    out_off: usize,
    out_stride: usize,
    len: usize,
    coeffs: &DericheCoefficients,
    pad: usize,
    scratch: &mut LineScratch,
) {
    let [n0, n1, n2, n3] = coeffs.n;
    let [d1, d2, d3, d4] = coeffs.d;
    let [m1, m2, m3, m4] = coeffs.m;
    let plen = len + 2 * pad;
    let (xp, yc, ya) = (&mut scratch.xp, &mut scratch.yc, &mut scratch.ya);

    // Edge-replicated padded input.
    let first = input[in_off] as f64;
    let last = input[in_off + (len - 1) * in_stride] as f64;
    for x in xp.iter_mut().take(pad) {
        *x = first;
    }
    for i in 0..len {
        xp[pad + i] = input[in_off + i * in_stride] as f64;
    }
    for x in xp.iter_mut().take(plen).skip(pad + len) {
        *x = last;
    }

    // Causal forward pass (taps before index 0 read the replicated first).
    for i in 0..plen {
        let x = xp[i];
        let xm1 = if i >= 1 { xp[i - 1] } else { first };
        let xm2 = if i >= 2 { xp[i - 2] } else { first };
        let xm3 = if i >= 3 { xp[i - 3] } else { first };
        let ym1 = if i >= 1 { yc[i - 1] } else { 0.0 };
        let ym2 = if i >= 2 { yc[i - 2] } else { 0.0 };
        let ym3 = if i >= 3 { yc[i - 3] } else { 0.0 };
        let ym4 = if i >= 4 { yc[i - 4] } else { 0.0 };
        yc[i] = n0 * x + n1 * xm1 + n2 * xm2 + n3 * xm3 - d1 * ym1 - d2 * ym2 - d3 * ym3 - d4 * ym4;
    }

    // Anticausal backward pass (taps after the end read the replicated last).
    for i in (0..plen).rev() {
        let xp1 = if i + 1 < plen { xp[i + 1] } else { last };
        let xp2 = if i + 2 < plen { xp[i + 2] } else { last };
        let xp3 = if i + 3 < plen { xp[i + 3] } else { last };
        let xp4 = if i + 4 < plen { xp[i + 4] } else { last };
        let yp1 = if i + 1 < plen { ya[i + 1] } else { 0.0 };
        let yp2 = if i + 2 < plen { ya[i + 2] } else { 0.0 };
        let yp3 = if i + 3 < plen { ya[i + 3] } else { 0.0 };
        let yp4 = if i + 4 < plen { ya[i + 4] } else { 0.0 };
        ya[i] = m1 * xp1 + m2 * xp2 + m3 * xp3 + m4 * xp4 - d1 * yp1 - d2 * yp2 - d3 * yp3 - d4 * yp4;
    }

    // Sum the two passes; write the unpadded interior back out.
    for i in 0..len {
        output[out_off + i * out_stride] = (yc[pad + i] + ya[pad + i]) as f32;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_iir.rs"]
mod tests;
