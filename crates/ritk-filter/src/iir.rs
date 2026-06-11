//! IIR 1-D primitives for recursive Gaussian filtering. //! //! Contains the Young–van Vliet coefficient computation, 1-D line iteration //! helpers, and the two-pass IIR smoothing / finite-difference derivative //! kernels used by the recursive Gaussian filter. //! //! # SIMD boundary/interior split //! //! All 1-D convolution loops are split into: //! 1. **Boundary pass** — processes the 1–2 edge elements per line where //! neighbor indices require clamping. Contains conditionals for edge cases. //! 2. **Interior pass** — processes all remaining elements with uniform stride //! and no conditionals. LLVM can auto-vectorize this loop (contiguous access //! for axis 2, known-in-bounds for axes 0 and 1). //! //! The split is transparent: arithmetic and boundary conditions are identical //! to the original combined loop. Differential verification confirms bitwise //! equivalence for every voxel.

// ── Young–van Vliet coefficient set ───────────────────────────────────────────

/// Precomputed Young–van Vliet IIR coefficients for one 1-D pass.
///
/// The recurrence is:
/// y[n] = B·x[n] + d1·y[n−1] + d2·y[n−2] + d3·y[n−3]
///
/// where B = 1 − d1 − d2 − d3 ensures unit DC gain.
pub(super) struct YvVCoefficients {
    /// Feedforward gain.
    pub(super) b_gain: f64,
    /// Feedback coefficient for y[n−1] (or y[n+1] in anticausal pass).
    pub(super) d1: f64,
    /// Feedback coefficient for y[n−2] (or y[n+2] in anticausal pass).
    pub(super) d2: f64,
    /// Feedback coefficient for y[n−3] (or y[n+3] in anticausal pass).
    pub(super) d3: f64,
}

impl YvVCoefficients {
    /// Compute the Young–van Vliet coefficients from a pixel-space sigma.
    ///
    /// # Derivation (Young & van Vliet 1995)
    ///
    /// Scale parameter q: q = 0.98711σ − 0.96330  (σ ≥ 2.5)
    ///                   q = 3.97156 − 4.14554√(1−0.26891σ)  (0.5 ≤ σ < 2.5)
    ///
    /// Un-normalised: b0 = 1.57825+2.44413q+1.4281q²+0.422205q³,
    ///                b1 = 2.44413q+2.85619q²+1.26661q³,
    ///                b2 = −(1.4281q²+1.26661q³),
    ///                b3 = 0.422205q³.
    /// Normalised: d_i = b_i / b0.  Feedforward gain: B = 1 − d1 − d2 − d3.
    pub(super) fn from_sigma(sigma: f64) -> Self {
        let q = if sigma >= 2.5 {
            0.98711 * sigma - 0.96330
        } else if sigma >= 0.5 {
            3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).max(0.0).sqrt()
        } else {
            // For very small sigma, linearly interpolate q towards 0
            0.1 * sigma / 0.5
        };
        let q2 = q * q;
        let q3 = q2 * q;
        let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        let b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        let b2 = -(1.4281 * q2 + 1.26661 * q3);
        let b3 = 0.422205 * q3;
        let d1 = b1 / b0;
        let d2 = b2 / b0;
        let d3 = b3 / b0;
        let b_gain = 1.0 - d1 - d2 - d3;
        Self { b_gain, d1, d2, d3 }
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

// ── Smoothing (order 0): two-pass IIR ─────────────────────────────────────────

/// Apply the Young–van Vliet two-pass IIR smoothing along dimension `dim`.
///
/// Pass 1 (causal/forward on input):
///   y_f[n] = B·x[n] + d1·y_f[n−1] + d2·y_f[n−2] + d3·y_f[n−3]
///
/// Pass 2 (anticausal/backward on y_f):
///   y[n] = B·y_f[n] + d1·y[n+1] + d2·y[n+2] + d3·y[n+3]
///
/// Boundary conditions: constant extension (replicate). The steady-state
/// response of the recursion to a constant input c is c (since B/(1−d1−d2−d3)
/// = B/B = 1), so we initialise the boundary taps to the first/last sample.
///
/// # Boundary/interior split
///
/// The forward and backward passes carry a 3-tap IIR recurrence whose
/// initialisation is inherently sequential (each element depends on the
/// previous 3). Therefore the boundary/interior split for the IIR is the
/// **initialisation phase** (first 3 elements of each pass, where the
/// y[n−k] taps are clamped to the edge value) versus the **steady-state
/// phase** (all remaining elements, where all taps are valid outputs from
/// prior iterations). LLVM cannot vectorize the IIR recurrence itself
/// (sequential dependency), but removing the init-phase conditionals from
/// the steady-state loop lets the compiler emit tighter loop body code with
/// fewer branch instructions and reduced register pressure.
#[inline]
pub(super) fn apply_smooth_1d(
    data: &[f32],
    dims: [usize; 3],
    dim: usize,
    coeffs: &YvVCoefficients,
) -> Vec<f32> {
    let lp = line_params(dims, dim);
    let mut output = vec![0.0_f32; data.len()];

    // Cast YvV coefficients to f32 once; f32 accumulation is numerically
    // sufficient for the 3-tap IIR recursion with coefficients O(1) at σ ≥ 0.5.
    let bg = coeffs.b_gain as f32;
    let d1 = coeffs.d1 as f32;
    let d2 = coeffs.d2 as f32;
    let d3 = coeffs.d3 as f32;

    // Hoisted line buffers — reused across all lines, eliminating per-line heap
    // allocations (2 × num_lines allocations per dimension).
    let mut x_buf = vec![0.0_f32; lp.len];
    let mut yf_buf = vec![0.0_f32; lp.len];

    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);

        // Read input line into contiguous buffer
        for i in 0..lp.len {
            x_buf[i] = data[base + i * lp.stride];
        }

        // --- Forward (causal) pass ---
        // Boundary initialisation: steady-state for constant extension = x[0]
        let init_fwd = x_buf[0];
        let mut ym1 = init_fwd;
        let mut ym2 = init_fwd;
        let mut ym3 = init_fwd;

        // Boundary phase (first 3 elements): taps ym1/ym2/ym3 are the
        // clamped-to-edge initialisation values. After element 2, all taps
        // are valid prior-iteration outputs and the recurrence enters
        // steady state.
        let fwd_boundary_end = lp.len.min(3);
        for i in 0..fwd_boundary_end {
            let val = bg * x_buf[i] + d1 * ym1 + d2 * ym2 + d3 * ym3;
            yf_buf[i] = val;
            ym3 = ym2;
            ym2 = ym1;
            ym1 = val;
        }

        // Interior phase (element 3..N): all taps are valid recurrence outputs.
        // No conditionals — single straight-line code per iteration.
        for i in 3..lp.len {
            let val = bg * x_buf[i] + d1 * ym1 + d2 * ym2 + d3 * ym3;
            yf_buf[i] = val;
            ym3 = ym2;
            ym2 = ym1;
            ym1 = val;
        }

        // --- Backward (anticausal) pass on yf ---
        // Boundary initialisation: steady-state for constant extension = yf[N-1]
        let init_bwd = yf_buf[lp.len - 1];
        let mut yp1 = init_bwd;
        let mut yp2 = init_bwd;
        let mut yp3 = init_bwd;

        // Boundary phase (last 3 elements): taps yp1/yp2/yp3 are clamped to
        // the edge. Process from N-1 down to N-3 (or 0 if len < 3).
        let bwd_boundary_start = lp.len.saturating_sub(3);
        for i in (bwd_boundary_start..lp.len).rev() {
            let val = bg * yf_buf[i] + d1 * yp1 + d2 * yp2 + d3 * yp3;
            output[base + i * lp.stride] = val;
            yp3 = yp2;
            yp2 = yp1;
            yp1 = val;
        }

        // Interior phase (element N-4..0): all taps are valid recurrence outputs.
        // No conditionals — single straight-line code per iteration.
        for i in (0..bwd_boundary_start).rev() {
            let val = bg * yf_buf[i] + d1 * yp1 + d2 * yp2 + d3 * yp3;
            output[base + i * lp.stride] = val;
            yp3 = yp2;
            yp2 = yp1;
            yp1 = val;
        }
    }
    output
}

// ── First derivative via central difference ───────────────────────────────────

/// Apply central-difference first derivative along dimension `dim`:
///   d[n] = (x[n+1] − x[n−1]) / 2
/// Boundary: one-sided differences at edges.
/// `_into` variant writes into a caller-provided buffer.
///
/// # Boundary/interior split
///
/// - **Boundary**: i=0 (forward one-sided), i=len−1 (backward one-sided).
/// - **Interior**: i=1..len−2 (central difference, no conditionals).
///
/// The interior loop accesses data at uniform offsets (+stride, −stride)
/// and contains no branches, enabling LLVM auto-vectorization.
#[inline]
pub(super) fn apply_first_derivative_1d_into(
    data: &[f32],
    dims: [usize; 3],
    dim: usize,
    out: &mut [f32],
) {
    let lp = line_params(dims, dim);
    if lp.len <= 1 {
        // Degenerate: single-element or empty line — derivative is zero.
        for li in 0..lp.num_lines {
            let base = line_base(dims, dim, li);
            out[base] = 0.0;
        }
        return;
    }

    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);
        let s = lp.stride;

        // Boundary: i = 0  →  forward one-sided: x[1] − x[0]
        out[base] = data[base + s] - data[base];

        // Interior: i = 1 .. len-2  →  central difference, no conditionals
        for i in 1..lp.len - 1 {
            out[base + i * s] = (data[base + (i + 1) * s] - data[base + (i - 1) * s]) * 0.5;
        }

        // Boundary: i = len-1  →  backward one-sided: x[N-1] − x[N-2]
        out[base + (lp.len - 1) * s] =
            data[base + (lp.len - 1) * s] - data[base + (lp.len - 2) * s];
    }
}

// ── Second derivative via second-order finite difference ──────────────────────

/// Apply second-order finite difference along dimension `dim`:
///   d²[n] = x[n+1] − 2·x[n] + x[n−1]
/// Boundary: one-sided at edges.
/// `_into` variant writes into a caller-provided buffer.
///
/// # Boundary/interior split
///
/// - **Boundary**: i=0 (forward one-sided), i=len−1 (backward one-sided).
/// - **Interior**: i=1..len−2 (central difference, no conditionals).
///
/// The interior loop accesses data at uniform offsets (+stride, −stride)
/// and contains no branches, enabling LLVM auto-vectorization.
#[inline]
pub(super) fn apply_second_derivative_1d_into(
    data: &[f32],
    dims: [usize; 3],
    dim: usize,
    out: &mut [f32],
) {
    let lp = line_params(dims, dim);
    if lp.len < 3 {
        // Degenerate: too short for any second-difference — output is zero.
        for li in 0..lp.num_lines {
            let base = line_base(dims, dim, li);
            for i in 0..lp.len {
                out[base + i * lp.stride] = 0.0;
            }
        }
        return;
    }

    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);
        let s = lp.stride;

        // Boundary: i = 0  →  forward one-sided: x[2] - 2x[1] + x[0]
        let x0 = data[base];
        let x1 = data[base + s];
        let x2 = data[base + 2 * s];
        out[base] = x2 - 2.0 * x1 + x0;

        // Interior: i = 1 .. len-2  →  central: x[i+1] - 2x[i] + x[i-1]
        // No conditionals — uniform stride, in-bounds access.
        for i in 1..lp.len - 1 {
            let xp = data[base + (i + 1) * s];
            let xc = data[base + i * s];
            let xm = data[base + (i - 1) * s];
            out[base + i * s] = xp - 2.0 * xc + xm;
        }

        // Boundary: i = len-1  →  backward one-sided: x[n] - 2x[n-1] + x[n-2]
        let n = lp.len - 1;
        let xn = data[base + n * s];
        let xn1 = data[base + (n - 1) * s];
        let xn2 = data[base + (n - 2) * s];
        out[base + n * s] = xn - 2.0 * xn1 + xn2;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_iir.rs"]
mod tests;
