use super::*;

/// Tolerance on a *simple* eigenvalue or on an orthonormality check.
///
/// The decomposition is closed form over `f64`, so its error is a handful of
/// rounding steps on quantities of order one after normalisation; the invariant
/// assembly is the widest of them, involving products of three elements. Ten
/// decimal digits leaves headroom over that and still fails on any real defect,
/// which would move a simple root by far more.
const TOLERANCE: f64 = 1.0e-10;

/// Tolerance on the separation of a *repeated* eigenvalue.
///
/// A multiple root of a polynomial is ill-conditioned in a way no amount of care
/// in the solver removes: a double root satisfies `(λ − r)² = 0` locally, so a
/// perturbation δ of the coefficients displaces it by `√δ`, not by `δ`. With
/// coefficients carrying relative rounding `ε ≈ 2.2·10⁻¹⁶`, a double root
/// resolves only to `√ε ≈ 1.5·10⁻⁸` of the matrix magnitude — and the two
/// commonest white-matter shapes, prolate and oblate, are exactly the double-root
/// cases.
///
/// The consequence is bounded and harmless at physiological magnitudes: at
/// `‖D‖ ≈ 10⁻³ mm²/s` the spurious separation is `10⁻¹¹ mm²/s`, eight orders
/// below the smallest diffusivity any tissue shows, so no derived invariant
/// moves in a reportable digit. This tolerance is that intrinsic bound with an
/// order of slack, not a threshold tuned until the test passed.
const REPEATED_ROOT_TOLERANCE: f64 = 1.0e-7;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn difference(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Assert the basis is orthonormal, right-handed, and the eigenvalues ordered.
///
/// These hold unconditionally — they are properties of how the triad is
/// assembled, not of how well separated the eigenvalues are — so every case
/// checks them with the same tolerance.
fn assert_orthonormal_basis(eigen: &SymmetricEigen) {
    for (index, vector) in eigen.vectors.iter().enumerate() {
        assert!(
            (norm(*vector) - 1.0).abs() < TOLERANCE,
            "eigenvector {index} must be a unit vector, has norm {}",
            norm(*vector)
        );
    }
    for (first, second) in [(0, 1), (0, 2), (1, 2)] {
        let projection = dot(eigen.vectors[first], eigen.vectors[second]);
        assert!(
            projection.abs() < TOLERANCE,
            "eigenvectors {first} and {second} must be orthogonal, dot product {projection}"
        );
    }
    let [v1, v2, v3] = eigen.vectors;
    let determinant = dot(v1, cross(v2, v3));
    assert!(
        (determinant - 1.0).abs() < TOLERANCE,
        "the basis must be right-handed, determinant {determinant}"
    );
    assert!(
        eigen.values[0] >= eigen.values[1] && eigen.values[1] >= eigen.values[2],
        "eigenvalues must be sorted descending, got {:?}",
        eigen.values
    );
}

/// Assert each vector satisfies `D vᵢ = λᵢ vᵢ` within `bound`.
///
/// Checking the eigen relation rather than comparing against expected vectors is
/// what keeps the assertion meaningful under degeneracy, where the vectors are
/// not unique but the relation still is.
///
/// The bound is supplied per case rather than fixed, because how tightly the
/// relation *can* hold depends on the eigenvalue gaps. A vector belonging to an
/// eigenvalue separated from its neighbour by `g` is determined only up to a
/// rotation within that near-degenerate subspace, and such a rotation leaves a
/// residual of order `g`. Demanding better than the gap would be demanding the
/// solver invent a distinction the matrix does not carry.
fn assert_eigen_relation(elements: [f64; 6], eigen: &SymmetricEigen, bound: f64) {
    for (index, vector) in eigen.vectors.iter().enumerate() {
        let applied = apply(elements, *vector);
        let expected = vector.map(|component| component * eigen.values[index]);
        let residual = norm(difference(applied, expected));
        assert!(
            residual < bound,
            "eigenvector {index} must satisfy D v = λ v within {bound:.3e}; residual {residual:.3e}"
        );
    }
}

/// Both assertions at the tolerance appropriate for well-separated eigenvalues.
fn assert_is_eigensystem(elements: [f64; 6], eigen: &SymmetricEigen, scale: f64) {
    assert_orthonormal_basis(eigen);
    assert_eigen_relation(elements, eigen, TOLERANCE * scale.max(1.0));
}

// ── Diagonal matrices, where the answer is read off directly ─────────────

#[test]
fn diagonal_matrix_returns_its_diagonal_sorted() {
    let eigen = symmetric_eigen([2.0, 5.0, 1.0, 0.0, 0.0, 0.0]);

    assert!((eigen.values[0] - 5.0).abs() < TOLERANCE);
    assert!((eigen.values[1] - 2.0).abs() < TOLERANCE);
    assert!((eigen.values[2] - 1.0).abs() < TOLERANCE);
    assert_is_eigensystem([2.0, 5.0, 1.0, 0.0, 0.0, 0.0], &eigen, 5.0);
    // The largest entry is on y, so the principal eigenvector is ±ŷ.
    assert!((eigen.vectors[0][1].abs() - 1.0).abs() < TOLERANCE);
}

#[test]
fn isotropic_matrix_returns_repeated_eigenvalues_and_an_orthonormal_basis() {
    let elements = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0];
    let eigen = symmetric_eigen(elements);

    for value in eigen.values {
        assert!((value - 3.0).abs() < TOLERANCE, "got {value}");
    }
    assert_is_eigensystem(elements, &eigen, 3.0);
}

// ── Analytically known non-diagonal cases ────────────────────────────────

/// A symmetric matrix with off-diagonal structure whose eigenvalues are known
/// in closed form.
///
/// `[[2,1,0],[1,2,0],[0,0,1]]` has eigenvalues `3, 1, 1` — the `2×2` block
/// contributes `2 ± 1` — with the `3` belonging to `(1,1,0)/√2`. The repeated
/// `1` is a genuine two-dimensional eigenspace, so this exercises degeneracy
/// alongside a known answer.
#[test]
fn block_matrix_matches_its_closed_form_eigenvalues() {
    let elements = [2.0, 2.0, 1.0, 1.0, 0.0, 0.0];
    let eigen = symmetric_eigen(elements);

    assert!(
        (eigen.values[0] - 3.0).abs() < TOLERANCE,
        "{:?}",
        eigen.values
    );
    assert!(
        (eigen.values[1] - 1.0).abs() < REPEATED_ROOT_TOLERANCE,
        "{:?}",
        eigen.values
    );
    assert!(
        (eigen.values[2] - 1.0).abs() < REPEATED_ROOT_TOLERANCE,
        "{:?}",
        eigen.values
    );
    assert_orthonormal_basis(&eigen);
    assert_eigen_relation(elements, &eigen, REPEATED_ROOT_TOLERANCE);

    // The leading eigenvalue is simple, so its eigenvector *is* determined.
    let expected = 1.0 / std::f64::consts::SQRT_2;
    let principal = eigen.vectors[0];
    assert!(
        (principal[0].abs() - expected).abs() < TOLERANCE
            && (principal[1].abs() - expected).abs() < TOLERANCE
            && principal[2].abs() < TOLERANCE,
        "principal eigenvector must be ±(1,1,0)/√2, got {principal:?}"
    );
}

/// A rotated tensor must return the rotated eigenvectors and the unchanged
/// eigenvalues — the statement that the decomposition is equivariant, which is
/// the property every orientation claim downstream depends on.
#[test]
fn rotating_the_tensor_rotates_the_eigenbasis_and_fixes_the_eigenvalues() {
    // D = R diag(3, 2, 1) Rᵀ for a rotation of π/6 about z.
    let angle = std::f64::consts::FRAC_PI_6;
    let (sin, cos) = angle.sin_cos();
    let diagonal = [3.0, 2.0, 1.0];
    // Only the xy block is affected by a z rotation.
    let dxx = diagonal[0] * cos * cos + diagonal[1] * sin * sin;
    let dyy = diagonal[0] * sin * sin + diagonal[1] * cos * cos;
    let dxy = (diagonal[0] - diagonal[1]) * sin * cos;
    let elements = [dxx, dyy, diagonal[2], dxy, 0.0, 0.0];

    let eigen = symmetric_eigen(elements);

    for (index, expected) in diagonal.iter().enumerate() {
        assert!(
            (eigen.values[index] - expected).abs() < TOLERANCE,
            "eigenvalue {index}: expected {expected}, got {}",
            eigen.values[index]
        );
    }
    assert_is_eigensystem(elements, &eigen, 3.0);

    // The principal axis is the rotated x̂.
    let principal = eigen.vectors[0];
    assert!(
        (principal[0].abs() - cos).abs() < TOLERANCE
            && (principal[1].abs() - sin).abs() < TOLERANCE
            && principal[2].abs() < TOLERANCE,
        "principal eigenvector must be the rotated x̂, got {principal:?}"
    );
}

// ── Degeneracy and near-degeneracy ───────────────────────────────────────

/// Two coincident leading eigenvalues leave the principal eigenvector
/// undetermined, but the basis must still be a valid eigensystem.
#[test]
fn repeated_leading_eigenvalue_still_yields_a_valid_eigensystem() {
    let elements = [4.0, 4.0, 1.0, 0.0, 0.0, 0.0];
    let eigen = symmetric_eigen(elements);

    assert!((eigen.values[0] - 4.0).abs() < REPEATED_ROOT_TOLERANCE);
    assert!((eigen.values[1] - 4.0).abs() < REPEATED_ROOT_TOLERANCE);
    assert!((eigen.values[2] - 1.0).abs() < REPEATED_ROOT_TOLERANCE);
    assert_orthonormal_basis(&eigen);
    assert_eigen_relation(elements, &eigen, REPEATED_ROOT_TOLERANCE);
}

/// Eigenvalues separated by a billionth of their magnitude — a near-isotropic
/// voxel — must still come back separated by that amount.
///
/// This is the case the trace removal exists for. Forming `p = I₂ − I₁²/3` from
/// the unshifted matrix would subtract two quantities near 3 to produce one near
/// `10⁻¹⁸`, losing the spread entirely into rounding; taking the invariants of
/// the deviatoric part instead keeps it exact.
///
/// The eigen relation is checked against the *gap* rather than machine epsilon:
/// three eigenvalues within `2·10⁻⁹` of one another leave their eigenvectors
/// free to rotate among themselves, and no solver can pin a direction the matrix
/// does not distinguish. What the assertion still catches is a vector mixing
/// outside that near-degenerate set.
#[test]
fn near_degenerate_eigenvalues_survive_the_decomposition() {
    let spread = 1.0e-9;
    let elements = [1.0, 1.0 + spread, 1.0 - spread, 0.0, 0.0, 0.0];
    let eigen = symmetric_eigen(elements);

    assert_orthonormal_basis(&eigen);
    // Representing `1 ± 10⁻⁹` already rounds the offset at `ε/spread ≈ 2·10⁻⁷`
    // relative, so a thousandth is comfortably inside what the input carries and
    // still four orders tighter than the total loss the unshifted formulation
    // would produce.
    let recovered = eigen.values[0] - eigen.values[2];
    let expected = 2.0 * spread;
    assert!(
        (recovered - expected).abs() < 1.0e-3 * expected,
        "the {expected:.0e} spread must survive the decomposition, got {recovered:.6e}"
    );
    // Every pair is within 2·spread, so 2·spread bounds any residual a rotation
    // inside the degenerate set can leave.
    assert_eigen_relation(elements, &eigen, 4.0 * spread);
}

/// A tensor at physiological scale — `10⁻³ mm²/s` — rather than order one, so
/// the solver is exercised where the invariants `I₂` and `I₃` are `10⁻⁶` and
/// `10⁻⁹` and cancellation is worst.
#[test]
fn physiological_magnitudes_decompose_without_cancellation_loss() {
    let elements = [0.3e-3, 0.4e-3, 1.7e-3, 0.05e-3, 0.02e-3, 0.01e-3];
    let eigen = symmetric_eigen(elements);

    assert_is_eigensystem(elements, &eigen, 1.0);
    // The trace is invariant, and is the cheapest independent check that the
    // three roots are the right ones.
    let trace: f64 = eigen.values.iter().sum();
    assert!(
        (trace - (0.3e-3 + 0.4e-3 + 1.7e-3)).abs() < 1.0e-16,
        "sum of eigenvalues must equal the trace, got {trace:.12e}"
    );
}

// ── Semi-definite and indefinite input ───────────────────────────────────

/// The dyadic `v vᵀ` that direction interpolation builds is rank one, so two
/// eigenvalues are exactly zero. The solver must handle it without a positivity
/// assumption, since this is the caller the unchecked entry point exists for.
#[test]
fn rank_one_dyadic_returns_one_nonzero_eigenvalue_along_its_generator() {
    let generator = [0.6, 0.0, 0.8];
    let elements = [
        generator[0] * generator[0],
        generator[1] * generator[1],
        generator[2] * generator[2],
        generator[0] * generator[1],
        generator[0] * generator[2],
        generator[1] * generator[2],
    ];

    let eigen = symmetric_eigen(elements);

    // The simple root is exact to full precision; the double root at zero is
    // resolvable only to √ε of the matrix magnitude, per REPEATED_ROOT_TOLERANCE.
    assert!(
        (eigen.values[0] - 1.0).abs() < TOLERANCE,
        "{:?}",
        eigen.values
    );
    assert!(
        eigen.values[1].abs() < REPEATED_ROOT_TOLERANCE,
        "{:?}",
        eigen.values
    );
    assert!(
        eigen.values[2].abs() < REPEATED_ROOT_TOLERANCE,
        "{:?}",
        eigen.values
    );
    assert_orthonormal_basis(&eigen);
    assert_eigen_relation(elements, &eigen, REPEATED_ROOT_TOLERANCE);
    // The generator spans the simple eigenvalue's eigenspace, which is
    // one-dimensional and therefore fully determined.
    assert!(
        (eigen.vectors[0][0].abs() - 0.6).abs() < TOLERANCE
            && (eigen.vectors[0][2].abs() - 0.8).abs() < TOLERANCE,
        "the nonzero eigenvalue belongs to the generator, got {:?}",
        eigen.vectors[0]
    );
}

/// An indefinite symmetric matrix is not a diffusion tensor, but the generic
/// solver must not assume otherwise — the positivity contract lives one level
/// up, and silently mishandling a negative root here would hide it.
#[test]
fn indefinite_matrix_decomposes_with_its_negative_eigenvalue_last() {
    let elements = [2.0, -1.0, 0.5, 0.0, 0.0, 0.0];
    let eigen = symmetric_eigen(elements);

    assert!((eigen.values[0] - 2.0).abs() < TOLERANCE);
    assert!((eigen.values[1] - 0.5).abs() < TOLERANCE);
    assert!((eigen.values[2] + 1.0).abs() < TOLERANCE);
    assert_is_eigensystem(elements, &eigen, 2.0);
}
