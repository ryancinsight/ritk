//! The Marchenko-Pastur noise-signal boundary (Veraart et al. 2016, Eq. 10–12).

use leto_ops::RealScalar;

/// The Marchenko-Pastur aspect ratio `γ_p` assigned to the trailing `m − p`
/// eigenvalues when testing `p` signal components.
///
/// The two published estimators differ only here; MRtrix `dwidenoise` exposes
/// them as `Exp1` and `Exp2`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum MpEstimator {
    /// `γ_p = (m − p)/n` — Veraart et al. (2016), *NeuroImage* 142, Eq. 11.
    Veraart2016,
    /// `γ_p = (m − p)/(n − p)` — Cordero-Grande, Christiaens, Hutter, Price &
    /// Hajnal (2019), "Complex diffusion-weighted image estimation via matrix
    /// recovery under general noise models", *NeuroImage* 200, 391–404.
    ///
    /// Removing `p` signal components also removes `p` degrees of freedom from
    /// the larger dimension, so the residual noise matrix is
    /// `(m − p) × (n − p)`; its bulk is correspondingly wider, and Eq. 11 with
    /// `n` understates that width and over-counts signal components.
    #[default]
    CorderoGrande2019,
}

impl MpEstimator {
    /// Larger dimension of the residual noise matrix after `p` components.
    fn residual_columns(self, n: usize, p: usize) -> usize {
        match self {
            Self::Veraart2016 => n,
            Self::CorderoGrande2019 => n - p,
        }
    }
}

/// Signal rank and noise variance of one window's spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NoiseBoundary<T> {
    /// `P̂`: the number of leading eigenvalues outside the noise bulk.
    pub(crate) signal_components: usize,
    /// `σ̂²`: the mean of the trailing `m − P̂` eigenvalues (Eq. 12).
    pub(crate) variance: T,
}

/// Locate the Marchenko-Pastur boundary of a spectrum.
///
/// `descending` holds `λ₁ ≥ … ≥ λ_m`, the eigenvalues of the Gram matrix
/// divided by `n` (the larger Casorati dimension). `p` increases from zero
/// until the trailing sum reaches `(m − p)·σ̂²(p)` with
/// `σ̂²(p) = (λ_{p+1} − λ_m) / (4√γ_p)` (Eq. 10–11), `γ_p` per `estimator`.
///
/// At `p = m − 1` the single trailing eigenvalue has zero support width, so
/// the criterion reduces to `λ_m ≥ 0`, which rounding can break for a
/// rank-deficient window; that last candidate is therefore accepted
/// unconditionally. For the same reason the returned variance is floored at
/// zero: a negative mean of trailing eigenvalues is rounding residue of an
/// exactly zero spectrum, never a measurement.
pub(crate) fn marchenko_pastur_boundary<T: RealScalar>(
    descending: &[T],
    n: usize,
    estimator: MpEstimator,
) -> NoiseBoundary<T> {
    let m = descending.len();
    debug_assert!(m >= 1 && n >= m, "invariant: 1 ≤ m ≤ n");
    let four = T::from_usize(4);
    let smallest = descending[m - 1];
    // Suffix sums let every candidate read its trailing sum in O(1).
    let mut trailing = vec![T::ZERO; m + 1];
    for i in (0..m).rev() {
        trailing[i] = trailing[i + 1] + descending[i];
    }
    for p in 0..m {
        let count = m - p;
        let sum = trailing[p];
        let count_t = T::from_usize(count);
        let accept = p + 1 == m || {
            let gamma = count_t / T::from_usize(estimator.residual_columns(n, p));
            let bulk_variance = (descending[p] - smallest) / (four * gamma.sqrt());
            sum >= count_t * bulk_variance
        };
        if accept {
            let variance = sum / count_t;
            return NoiseBoundary {
                signal_components: p,
                variance: if variance > T::ZERO {
                    variance
                } else {
                    T::ZERO
                },
            };
        }
    }
    unreachable!("invariant: the p = m − 1 candidate is always accepted")
}
