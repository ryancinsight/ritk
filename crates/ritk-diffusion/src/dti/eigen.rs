//! Eigendecomposition of a 3×3 real symmetric matrix.
//!
//! A diffusion tensor is only ever read through its eigensystem: every scalar
//! invariant is a function of the eigenvalues, and every orientation claim is a
//! function of the eigenvectors. The decomposition is therefore the hot path of
//! the whole model, run once per voxel over a volume, so it is solved in closed
//! form rather than by an iterative dense routine.
//!
//! # Method
//!
//! Eigenvalues come from the analytic roots of the characteristic polynomial.
//! The polynomial `λ³ − I₁λ² + I₂λ − I₃ = 0` is depressed by `λ = μ + I₁/3`
//! into `μ³ + pμ + q = 0`, and a real symmetric matrix has three real roots, so
//! the discriminant is nonpositive and `p ≤ 0`. The trigonometric form
//!
//! ```text
//! μₖ = 2√(−p/3) · cos( (φ + 2πk)/3 ),  φ = acos( −q / (2(−p/3)^{3/2}) )
//! ```
//!
//! returns all three without a complex intermediate. Rounding can nudge `p`
//! just above zero or the arccosine argument just outside `[−1, 1]`; both are
//! clamped, because in both cases the true value is the boundary.
//!
//! # Why the trace is removed first
//!
//! The depression is applied to the *matrix*, not to the polynomial: the
//! decomposition runs on the deviatoric part `D̃ = D − (tr D/3) I` and adds the
//! shift back to each root. Algebraically the two routes agree, but numerically
//! they do not, and the difference decides whether a near-isotropic voxel
//! decomposes at all.
//!
//! Taking the polynomial route, `p = I₂ − I₁²/3` subtracts two quantities of
//! order `‖D‖²` to produce one of order `(λ₁ − λ₃)²`. For a voxel whose
//! eigenvalues are separated by `10⁻⁶` of their magnitude — grey matter, CSF,
//! anywhere the tissue is near isotropic — that is a cancellation of twelve
//! digits, and the result is dominated by the rounding of the operands rather
//! than by the spread it is meant to measure. Removing the trace first makes
//! `I₁ = 0` by construction, so `p = I₂(D̃)` is formed directly from entries
//! that are already the size of the spread, and nothing large is ever
//! subtracted. The recovered separation is then accurate to the spread's own
//! precision instead of to the trace's.
//!
//! Eigenvectors are recovered from the nullspace of `D − λI`. Any two
//! independent rows of that singular matrix span the orthogonal complement of
//! the eigenvector, so their cross product is the eigenvector; all three row
//! pairs are tried and the longest product kept, since a single pair can be
//! degenerate when its two rows happen to be parallel.
//!
//! The basis is closed by construction rather than by decomposing three times:
//! `v₁` comes from `λ₁` and `v₃` from `λ₃` — the two extremal eigenvalues, whose
//! eigenspaces are the best separated — and `v₂ = v₃ × v₁` completes a
//! right-handed orthonormal triad. `v₃` is then re-formed as `v₁ × v₂` so the
//! triad is exactly orthonormal even when the two independently recovered
//! vectors were not quite perpendicular.
//!
//! # Degeneracy
//!
//! Repeated eigenvalues make individual eigenvectors non-unique: for
//! `λ₁ = λ₂ > λ₃` any vector of the plane spanned by `v₁` and `v₂` is a valid
//! principal eigenvector. The routine still returns an orthonormal basis that
//! satisfies `D vᵢ = λᵢ vᵢ`, which is the contract callers rely on; it does not
//! promise a particular representative from a degenerate eigenspace, because
//! none is distinguished. A numerically isotropic matrix returns the canonical
//! axes.
//!
//! # References
//!
//! * Kopp, J. (2008). Efficient numerical diagonalization of Hermitian 3×3
//!   matrices. *International Journal of Modern Physics C* 19(3):523–548.
//! * Smith, O. K. (1961). Eigenvalues of a symmetric 3×3 matrix.
//!   *Communications of the ACM* 4(4):168.

/// Below this squared length a candidate eigenvector is numerical noise.
const DEGENERATE_NORM_SQUARED: f64 = 1.0e-30;

/// The eigensystem of a symmetric 3×3 matrix.
///
/// Eigenvalues are sorted descending and each `eigenvectors[i]` is the unit
/// eigenvector of `eigenvalues[i]`. The three vectors are orthonormal and
/// right-handed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SymmetricEigen {
    /// `λ₁ ≥ λ₂ ≥ λ₃`.
    pub(crate) values: [f64; 3],
    /// Unit eigenvectors, in the same order as `values`.
    pub(crate) vectors: [[f64; 3]; 3],
}

/// Eigenvalues and an orthonormal eigenbasis of the symmetric matrix whose six
/// unique elements are `[Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]` (Voigt order).
///
/// The result is raw: no positivity check is applied, because positivity is a
/// property of a *diffusion tensor*, not of a symmetric matrix. Callers whose
/// matrix is legitimately singular or indefinite — the dyadic
/// `Σ wᵢ vᵢvᵢᵀ` that [`crate::maps::DtiVolume`] interpolates is positive
/// *semi*-definite by construction — use this directly.
pub(crate) fn symmetric_eigen(elements: [f64; 6]) -> SymmetricEigen {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = elements;

    // Deviatoric part: the trace is removed from the matrix before the
    // invariants are formed, so no large quantity is ever subtracted from
    // another. See the module note on why this is not merely equivalent.
    let shift = (dxx + dyy + dzz) / 3.0;
    let [axx, ayy, azz] = [dxx - shift, dyy - shift, dzz - shift];

    // With a traceless matrix the depressed cubic μ³ + pμ + q is the
    // characteristic polynomial itself: p = I₂(D̃), q = −I₃(D̃).
    // p ≤ 0 for a real symmetric matrix; rounding can lift it, and clamping
    // restores the true boundary.
    let p = (axx * ayy + axx * azz + ayy * azz - dxy * dxy - dxz * dxz - dyz * dyz).min(0.0);
    let q = -(axx * ayy * azz + 2.0 * dxy * dxz * dyz
        - axx * dyz * dyz
        - ayy * dxz * dxz
        - azz * dxy * dxy);

    let radius = (-p / 3.0).sqrt();
    // The eigenvalues are `shift ± O(radius)`, so a spread below the rounding of
    // the shift itself is not a spread the entries could have carried: the three
    // roots coincide to the last representable bit. The test is relative rather
    // than absolute because the same matrix scaled by a constant must classify
    // the same way, which a fixed threshold in mm²/s would not do.
    let magnitude = shift.abs().max(radius);
    if radius <= magnitude * f64::EPSILON {
        // Every direction is an eigenvector, so the canonical axes are as good a
        // basis as any other.
        return SymmetricEigen {
            values: [shift; 3],
            vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
    }

    let argument = (-q / (2.0 * radius.powi(3))).clamp(-1.0, 1.0);
    let phi = argument.acos();
    let amplitude = 2.0 * radius;
    let trigonometric = [
        amplitude * (phi / 3.0).cos(),
        amplitude * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos(),
        amplitude * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos(),
    ];

    // Only one of the three trigonometric roots is taken; the other two are
    // deflated from it. The reason is that `acos` has an infinite derivative at
    // ±1, which is exactly where the argument lands when the cubic has a double
    // root — a prolate or oblate tensor, the two commonest shapes in white
    // matter. A rounding error ε in the argument becomes an error of order √ε in
    // φ, so the roots taken from a cosine with nonvanishing slope lose half
    // their significant digits: eigenvalues that should be equal come back
    // separated by ~10⁻⁸ of their magnitude, which is a visible spurious
    // anisotropy rather than a rounding detail.
    //
    // The root of largest magnitude is immune to this. At a double root the
    // triple is `(2a, −a, −a)`, so the extremal root is the simple one, and it
    // is read off a cosine at 0 or π where the slope vanishes and the √ε error
    // cancels to second order. Anchoring there and recovering the other two from
    // the exact relations `μ₁ + μ₂ = −μ₀` and `μ₁μ₂ = p + μ₀²` — the elementary
    // symmetric functions of a depressed cubic — turns the degenerate case into
    // a discriminant of exactly zero instead of a near-cancelling square root.
    let anchor = trigonometric
        .into_iter()
        .max_by(|left, right| left.abs().total_cmp(&right.abs()))
        .unwrap_or(0.0);
    // The remaining pair solves t² + μ₀t + (p + μ₀²) = 0. Its discriminant is
    // nonnegative for a real symmetric matrix; rounding can dip it below zero
    // at the double root, where the true value is zero.
    let discriminant = (-3.0 * anchor * anchor - 4.0 * p).max(0.0);
    let separation = discriminant.sqrt();
    let mut values = [
        anchor + shift,
        0.5 * (-anchor + separation) + shift,
        0.5 * (-anchor - separation) + shift,
    ];
    values.sort_by(|a, b| b.total_cmp(a));

    // The extremal eigenvalues have the best-separated eigenspaces, so their
    // nullspaces are the two the cross-product extraction resolves most
    // reliably. The middle vector is then fixed by orthogonality.
    let first = nullspace_direction(elements, values[0]);
    let last = nullspace_direction(elements, values[2]);
    let vectors = match (first, last) {
        (Some(v1), Some(v3)) => close_basis(v1, v3),
        (Some(v1), None) => complete_from_one(v1),
        (None, Some(v3)) => {
            let [a, b, c] = complete_from_one(v3);
            // `complete_from_one` puts its argument first; the recovered vector
            // belongs to λ₃, so rotate it into the last slot while keeping the
            // triad right-handed.
            [b, c, a]
        }
        (None, None) => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    SymmetricEigen { values, vectors }
}

/// A unit vector spanning the nullspace of `D − λI`, or `None` when every row
/// pair is degenerate.
fn nullspace_direction(elements: [f64; 6], lambda: f64) -> Option<[f64; 3]> {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = elements;
    let rows = [
        [dxx - lambda, dxy, dxz],
        [dxy, dyy - lambda, dyz],
        [dxz, dyz, dzz - lambda],
    ];
    // Two independent rows of a singular matrix span the complement of its
    // nullspace, so their cross product lies in it. Trying all three pairs
    // covers the case where one pair is itself dependent.
    let candidates = [
        cross(rows[0], rows[1]),
        cross(rows[0], rows[2]),
        cross(rows[1], rows[2]),
    ];
    let best = candidates
        .into_iter()
        .max_by(|a, b| norm_squared(*a).total_cmp(&norm_squared(*b)))?;
    normalize(best)
}

/// Complete an orthonormal right-handed triad whose first vector is `v1`.
fn complete_from_one(v1: [f64; 3]) -> [[f64; 3]; 3] {
    let v2 = normalize(cross(v1, any_perpendicular(v1))).unwrap_or([0.0, 1.0, 0.0]);
    let v3 = cross(v1, v2);
    [v1, v2, v3]
}

/// Close a right-handed orthonormal triad from the two extremal eigenvectors.
///
/// `v3` is re-formed from `v1 × v2` rather than kept as supplied, so the triad
/// is exactly orthonormal even where the two independent extractions disagree
/// slightly.
fn close_basis(v1: [f64; 3], v3: [f64; 3]) -> [[f64; 3]; 3] {
    let Some(v2) = normalize(cross(v3, v1)) else {
        // v1 and v3 are parallel, which means the two extremal eigenvalues share
        // an eigenspace: the matrix is degenerate and any completion is valid.
        return complete_from_one(v1);
    };
    [v1, v2, cross(v1, v2)]
}

/// Any unit-length vector not parallel to `v`.
///
/// Choosing the axis of smallest component guarantees the cross product with
/// `v` is well conditioned.
fn any_perpendicular(v: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = v.map(f64::abs);
    if x <= y && x <= z {
        [1.0, 0.0, 0.0]
    } else if y <= z {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm_squared(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let squared = norm_squared(v);
    if squared < DEGENERATE_NORM_SQUARED {
        return None;
    }
    let norm = squared.sqrt();
    Some([v[0] / norm, v[1] / norm, v[2] / norm])
}

/// Multiply the symmetric matrix in Voigt order by a vector.
///
/// The eigen relation `D vᵢ = λᵢ vᵢ` is what the decomposition claims, and
/// checking it needs the left-hand side; nothing in the library reads a tensor
/// that way, so this exists for the tests that verify the claim.
#[cfg(test)]
pub(crate) fn apply(elements: [f64; 6], v: [f64; 3]) -> [f64; 3] {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = elements;
    [
        dxx * v[0] + dxy * v[1] + dxz * v[2],
        dxy * v[0] + dyy * v[1] + dyz * v[2],
        dxz * v[0] + dyz * v[1] + dzz * v[2],
    ]
}

#[cfg(test)]
mod tests;
