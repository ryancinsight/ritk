//! Rotationally invariant scalars derived from tensor eigenvalues.
//!
//! Every standard DTI scalar map is a function of the three eigenvalues alone —
//! that is what makes them rotationally invariant, and why a map can be built
//! from a stored eigenvalue triple without keeping the tensor. Defining them
//! here rather than on [`crate::dti::DiffusionTensor`] keeps one definition per
//! measure: the per-voxel tensor and the whole-volume
//! [`crate::maps::DiffusionMaps`] both call into this module instead of each
//! carrying its own copy.
//!
//! Eigenvalues arrive sorted descending, `λ₁ ≥ λ₂ ≥ λ₃`, and every function
//! here is total: a degenerate or all-zero triple returns the limiting value
//! rather than a NaN, because an unfitted voxel must read as a number.
//!
//! # References
//!
//! * Basser, P. J. & Pierpaoli, C. (1996). Microstructural and physiological
//!   features of tissues elucidated by quantitative-diffusion-tensor MRI.
//!   *Journal of Magnetic Resonance B* 111(3):209–219. — FA, RA, MD.
//! * Westin, C.-F., Peled, S., Gudbjartsson, H., Kikinis, R. & Jolesz, F. A.
//!   (1997). Geometrical diffusion measures for MRI from tensor basis analysis.
//!   *Proceedings of ISMRM* 5:1742. — linear / planar / spherical measures.
//! * Ennis, D. B. & Kindlmann, G. (2006). Orthogonal tensor invariants and the
//!   analysis of diffusion tensor magnetic resonance images. *Magnetic
//!   Resonance in Medicine* 55(1):136–146. — mode of anisotropy.

/// Below this the eigenvalue triple carries no magnitude and every normalised
/// measure is defined by its limit rather than by division.
///
/// Squared diffusivities in tissue are of order `10⁻⁶ mm⁴/s²`, twelve orders
/// above this, so the guard only ever fires on the exact zeros an unfitted
/// voxel stores.
const DEGENERATE_MAGNITUDE: f64 = 1.0e-30;

/// Mean diffusivity `(λ₁ + λ₂ + λ₃) / 3`, in mm²/s.
///
/// One third of the trace, hence invariant under rotation and equal to the
/// average apparent diffusivity over all directions.
#[must_use]
pub fn mean_diffusivity([l1, l2, l3]: [f64; 3]) -> f64 {
    (l1 + l2 + l3) / 3.0
}

/// Axial diffusivity `λ₁`, in mm²/s — diffusivity along the principal axis.
#[must_use]
pub fn axial_diffusivity([l1, _, _]: [f64; 3]) -> f64 {
    l1
}

/// Radial diffusivity `(λ₂ + λ₃) / 2`, in mm²/s — diffusivity across the
/// principal axis.
#[must_use]
pub fn radial_diffusivity([_, l2, l3]: [f64; 3]) -> f64 {
    (l2 + l3) / 2.0
}

/// Fractional anisotropy, in `[0, 1]`.
///
/// ```text
/// FA = √(3/2) · ‖λ − λ̄‖ / ‖λ‖
/// ```
///
/// The ratio of the deviatoric part's magnitude to the whole tensor's, scaled
/// so that a perfectly prolate tensor `(λ, 0, 0)` reaches one and an isotropic
/// tensor reaches zero. Returns zero when the tensor has no magnitude.
#[must_use]
pub fn fractional_anisotropy(eigenvalues: [f64; 3]) -> f64 {
    let magnitude = squared_magnitude(eigenvalues);
    if magnitude <= DEGENERATE_MAGNITUDE {
        return 0.0;
    }
    (1.5 * deviatoric_squared_magnitude(eigenvalues) / magnitude).sqrt()
}

/// Relative anisotropy, in `[0, √2]`.
///
/// ```text
/// RA = ‖λ − λ̄‖ / (√3 · λ̄)
/// ```
///
/// The deviatoric magnitude measured against the isotropic part rather than
/// against the whole tensor, which is what separates it from FA: RA is
/// unbounded above as the tensor approaches a line, and is clamped here only by
/// the eigenvalues themselves being nonnegative. Returns zero for a tensor with
/// no isotropic part.
#[must_use]
pub fn relative_anisotropy(eigenvalues: [f64; 3]) -> f64 {
    let mean = mean_diffusivity(eigenvalues);
    if mean.abs() <= DEGENERATE_MAGNITUDE {
        return 0.0;
    }
    deviatoric_squared_magnitude(eigenvalues).sqrt() / (3.0_f64.sqrt() * mean)
}

/// Frobenius norm `‖D‖ = √(Σ λᵢ²)`, in mm²/s.
#[must_use]
pub fn tensor_norm(eigenvalues: [f64; 3]) -> f64 {
    squared_magnitude(eigenvalues).sqrt()
}

/// Westin geometric measures `(cₗ, cₚ, cₛ)` — the linear, planar, and spherical
/// fractions of the tensor's shape.
///
/// ```text
/// cₗ = (λ₁ − λ₂) / (λ₁ + λ₂ + λ₃)
/// cₚ = 2(λ₂ − λ₃) / (λ₁ + λ₂ + λ₃)
/// cₛ = 3λ₃ / (λ₁ + λ₂ + λ₃)
/// ```
///
/// The three are a partition — they sum to one for any nonnegative triple — so
/// they answer "what shape is this tensor" where FA answers only "how far from
/// a sphere". A single coherent fibre bundle is near `cₗ = 1`; crossing fibres
/// within a voxel present as a disc, near `cₚ = 1`; free water is `cₛ = 1`.
/// FA cannot tell the first two apart, and that is exactly the ambiguity these
/// resolve.
///
/// Returns `(0, 0, 1)` — the spherical limit — for a tensor with no trace.
#[must_use]
pub fn westin_measures([l1, l2, l3]: [f64; 3]) -> (f64, f64, f64) {
    let trace = l1 + l2 + l3;
    if trace <= DEGENERATE_MAGNITUDE {
        return (0.0, 0.0, 1.0);
    }
    ((l1 - l2) / trace, 2.0 * (l2 - l3) / trace, 3.0 * l3 / trace)
}

/// Mode of anisotropy, in `[−1, 1]`.
///
/// ```text
/// mode = 3√6 · det(D̃) / ‖D̃‖³,   D̃ = D − λ̄ I
/// ```
///
/// The third invariant of the deviatoric tensor, normalised by its own
/// magnitude, so it describes shape independently of how anisotropic the tensor
/// is. `+1` is a prolate (cigar) tensor, `−1` an oblate (pancake) one, and `0`
/// the orthotropic case midway between. Mode and FA are orthogonal invariants:
/// FA gives the magnitude of the deviation from isotropy, mode gives its type,
/// and neither is recoverable from the other.
///
/// Returns zero for an isotropic tensor, whose deviatoric part has no magnitude
/// and therefore no shape.
#[must_use]
pub fn mode(eigenvalues: [f64; 3]) -> f64 {
    let mean = mean_diffusivity(eigenvalues);
    let [a1, a2, a3] = [
        eigenvalues[0] - mean,
        eigenvalues[1] - mean,
        eigenvalues[2] - mean,
    ];
    let magnitude_squared = a1 * a1 + a2 * a2 + a3 * a3;
    if magnitude_squared <= DEGENERATE_MAGNITUDE {
        return 0.0;
    }
    let normalisation = magnitude_squared.sqrt().powi(3);
    // The deviatoric part is diagonal in the eigenbasis, so its determinant is
    // the product of the shifted eigenvalues.
    (3.0 * 6.0_f64.sqrt() * a1 * a2 * a3 / normalisation).clamp(-1.0, 1.0)
}

/// Direction-encoded colour: `FA · |v₁|` as an RGB triple in `[0, 1]³`.
///
/// The standard clinical rendering of a tensor field. The principal
/// eigenvector's absolute components map to red, green, and blue — left-right,
/// anterior-posterior, and superior-inferior in a conventional frame — scaled
/// by FA so that isotropic tissue fades to black rather than showing the
/// arbitrary orientation of a sphere. Absolute values are taken because an
/// eigenvector has no sign: `v` and `−v` describe the same fibre.
///
/// The caller is responsible for the frame: the components are coloured in
/// whatever frame the eigenvector was expressed in, which
/// [`crate::dti::DiffusionTensor::frame`] records.
#[must_use]
pub fn colour_by_orientation(eigenvalues: [f64; 3], principal: [f64; 3]) -> [f64; 3] {
    let scale = fractional_anisotropy(eigenvalues);
    principal.map(|component| (component.abs() * scale).clamp(0.0, 1.0))
}

/// `Σ λᵢ²`.
fn squared_magnitude([l1, l2, l3]: [f64; 3]) -> f64 {
    l1 * l1 + l2 * l2 + l3 * l3
}

/// `Σ (λᵢ − λ̄)²` — the squared magnitude of the deviatoric part.
fn deviatoric_squared_magnitude(eigenvalues: [f64; 3]) -> f64 {
    let mean = mean_diffusivity(eigenvalues);
    eigenvalues
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum()
}

#[cfg(test)]
mod tests;
