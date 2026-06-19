use std::f64::consts::PI;

// ── Eigendecomposition (3×3 symmetric) ────────────────────────────────────────

/// Eigenvalues and eigenvectors of a 3×3 symmetric matrix.
///
/// Eigenvalues are sorted ascending: λ₁ ≤ λ₂ ≤ λ₃.
/// Columns of `eigenvecs` are the corresponding eigenvectors.
pub struct EigenDecomp {
    pub eigenvalues: [f64; 3],
    pub eigenvecs: [[f64; 3]; 3], // eigenvecs[k] = k-th eigenvector
}

/// Analytical eigenvalue decomposition of a 3×3 symmetric matrix.
///
/// Uses the trigonometric method (Smith 1961; Kopp 2008, arXiv:physics/0610206).
/// For degenerate eigenvalues, eigenvectors are selected by Gram-Schmidt
/// orthogonalisation in the invariant subspace.
///
/// Input layout: `h = [J_11, J_12, J_13, J_22, J_23, J_33]`
/// (upper triangle, row-major).
///
/// ```text
/// M = | h[0] h[1] h[2] |
///     | h[1] h[3] h[4] |
///     | h[2] h[4] h[5] |
/// ```
pub fn eigen_3x3_symmetric(h: [f64; 6]) -> EigenDecomp {
    let m00 = h[0];
    let m01 = h[1];
    let m02 = h[2];
    let m11 = h[3];
    let m12 = h[4];
    let m22 = h[5];

    // Off-diagonal Frobenius contribution.
    let p1 = m01 * m01 + m02 * m02 + m12 * m12;
    if p1 == 0.0 {
        // Diagonal matrix: eigenvalues are the diagonal entries.
        // Eigenvectors are the standard basis vectors.
        let eigs = [m00, m11, m22];
        let mut indices = [0usize, 1, 2];
        indices.sort_unstable_by(|&a, &b| {
            eigs[a]
                .partial_cmp(&eigs[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted = [eigs[indices[0]], eigs[indices[1]], eigs[indices[2]]];
        let mut vecs = [[0.0f64; 3]; 3];
        for (k, &idx) in indices.iter().enumerate() {
            vecs[k][idx] = 1.0;
        }
        return EigenDecomp {
            eigenvalues: sorted,
            eigenvecs: vecs,
        };
    }

    let q = (m00 + m11 + m22) / 3.0;
    let p2 = (m00 - q) * (m00 - q) + (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p < f64::EPSILON {
        // Numerically a scalar multiple of identity.
        return EigenDecomp {
            eigenvalues: [q, q, q],
            eigenvecs: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
    }

    // B = (M − q·I) / p
    let b00 = (m00 - q) / p;
    let b01 = m01 / p;
    let b02 = m02 / p;
    let b11 = (m11 - q) / p;
    let b12 = m12 / p;
    let b22 = (m22 - q) / p;

    // det(B) via cofactor expansion along the first row.
    let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);

    // r = det(B)/2, clamped to [−1, 1] for numerical safety.
    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    // Three eigenvalues before sorting.
    let eig_a = q + 2.0 * p * phi.cos();
    let eig_c = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
    // Trace identity: eig_b = 3q − eig_a − eig_c.
    let eig_b = 3.0 * q - eig_a - eig_c;

    let eigs_unsorted = [eig_a, eig_b, eig_c];
    let mut idx = [0usize, 1, 2];
    idx.sort_unstable_by(|&a, &b| {
        eigs_unsorted[a]
            .partial_cmp(&eigs_unsorted[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eigenvalues = [
        eigs_unsorted[idx[0]],
        eigs_unsorted[idx[1]],
        eigs_unsorted[idx[2]],
    ];

    // ── Eigenvectors ──────────────────────────────────────────────────
    // (M − λI) v = 0 → solve via cross products of rows.
    let eigenvecs = eigenvectors_3x3_symmetric(h, eigenvalues);

    EigenDecomp {
        eigenvalues,
        eigenvecs,
    }
}

/// Compute eigenvectors of a 3×3 symmetric matrix given its eigenvalues.
///
/// For each eigenvalue λ, the eigenvector is the cross product of two
/// rows of (M − λI). For degenerate eigenvalues, Gram-Schmidt is applied
/// within the invariant subspace.
fn eigenvectors_3x3_symmetric(h: [f64; 6], eigenvalues: [f64; 3]) -> [[f64; 3]; 3] {
    let m00 = h[0];
    let m01 = h[1];
    let m02 = h[2];
    let m11 = h[3];
    let m12 = h[4];
    let m22 = h[5];

    let mut vecs = [[0.0f64; 3]; 3];

    for (k, &lam) in eigenvalues.iter().enumerate() {
        // (M − λI) rows:
        let r0 = [m00 - lam, m01, m02];
        let r1 = [m01, m11 - lam, m12];
        let r2 = [m02, m12, m22 - lam];

        // Cross products of row pairs.
        let c01 = cross3(r0, r1);
        let c02 = cross3(r0, r2);
        let c12 = cross3(r1, r2);

        let n01 = norm3(c01);
        let n02 = norm3(c02);
        let n12 = norm3(c12);

        // Select the cross product with the largest norm (best conditioning).
        let v = if n01 >= n02 && n01 >= n12 {
            if n01 > 0.0 {
                scale3(c01, 1.0 / n01)
            } else {
                // Fallback: try cross product with canonical axis.
                let c = cross3(r0, [1.0, 0.0, 0.0]);
                let nc = norm3(c);
                if nc > 0.0 {
                    scale3(c, 1.0 / nc)
                } else {
                    [0.0, 1.0, 0.0]
                }
            }
        } else if n02 >= n12 {
            scale3(c02, 1.0 / n02)
        } else {
            scale3(c12, 1.0 / n12)
        };
        vecs[k] = v;
    }

    // Handle degenerate eigenvalues: orthogonalise within each subspace.
    // Since eigenvalues are sorted ascending, we process in order.
    let degen_tol = f64::max(1e-12, f64::abs(eigenvalues[2]) * 1e-10);

    // Check λ₁ ≈ λ₂.
    if (eigenvalues[1] - eigenvalues[0]).abs() < degen_tol {
        // Orthogonalise v₂ against v₁.
        vecs[1] = orthogonalise_against(vecs[1], vecs[0]);
    }
    // Check λ₂ ≈ λ₃.
    if (eigenvalues[2] - eigenvalues[1]).abs() < degen_tol {
        // Orthogonalise v₃ against v₂ (and transitively v₁).
        vecs[2] = orthogonalise_against(vecs[2], vecs[1]);
        vecs[2] = orthogonalise_against(vecs[2], vecs[0]);
    }
    // If λ₁ ≈ λ₃ (implies all equal), orthogonalise v₃ against v₁ too.
    if (eigenvalues[2] - eigenvalues[0]).abs() < degen_tol
        && (eigenvalues[2] - eigenvalues[1]).abs() >= degen_tol
    {
        vecs[2] = orthogonalise_against(vecs[2], vecs[0]);
    }

    vecs
}

// ── Small vector helpers ──────────────────────────────────────────────────────

#[inline(always)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline(always)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Orthogonalise `v` against `u` (assumed unit). Returns a unit vector.
#[inline(always)]
fn orthogonalise_against(v: [f64; 3], u: [f64; 3]) -> [f64; 3] {
    let proj = dot3(v, u);
    let mut w = [v[0] - proj * u[0], v[1] - proj * u[1], v[2] - proj * u[2]];
    let n = norm3(w);
    if n > 1e-15 {
        w = scale3(w, 1.0 / n);
    } else {
        // v was parallel to u; pick an arbitrary perpendicular.
        w = if u[0].abs() < 0.9 {
            let c = cross3(u, [1.0, 0.0, 0.0]);
            scale3(c, 1.0 / norm3(c))
        } else {
            let c = cross3(u, [0.0, 1.0, 0.0]);
            scale3(c, 1.0 / norm3(c))
        };
    }
    w
}

// ── Diffusion tensor ──────────────────────────────────────────────────────────

/// Construct the diffusion tensor D at a single voxel from the structure tensor.
///
/// Returns the 6 independent components of D in the same layout as the
/// structure tensor: [D_11, D_12, D_13, D_22, D_23, D_33].
///
/// Eigenvalue assignment (Weickert 1999):
/// α₁ = α + (1 − α) · (1 − exp(−C · (λ₃ − λ₁)² / (λ₃² + ε))) `[coherence dir]`
/// α₂ = α + (1 − α) · (1 − exp(−C · (λ₂ − λ₁)² / (λ₃² + ε))) `[intermediate]`
/// α₃ = α `[edge dir]`
pub(crate) fn diffusion_tensor(st: [f64; 6], alpha: f64, contrast: f64) -> [f64; 6] {
    let decomp = eigen_3x3_symmetric(st);
    let [lam1, lam2, lam3] = decomp.eigenvalues;
    let [e1, e2, e3] = decomp.eigenvecs;

    let lam3_sq = lam3 * lam3;

    // Coherence measure for the primary coherence direction.
    let diff_31 = lam3 - lam1;
    let exponent1 = -contrast * diff_31 * diff_31 / (lam3_sq + EPS);
    let alpha1 = alpha + (1.0 - alpha) * (1.0 - exponent1.exp());

    // Intermediate direction.
    let diff_21 = lam2 - lam1;
    let exponent2 = -contrast * diff_21 * diff_21 / (lam3_sq + EPS);
    let alpha2 = alpha + (1.0 - alpha) * (1.0 - exponent2.exp());

    // Edge direction: minimal diffusion.
    let alpha3 = alpha;

    // Reconstruct D = α₁·e₁·e₁ᵀ + α₂·e₂·e₂ᵀ + α₃·e₃·e₃ᵀ
    let mut d = [0.0f64; 6];
    for (alpha_i, e) in [(alpha1, e1), (alpha2, e2), (alpha3, e3)] {
        // Outer product: e · eᵀ, upper triangle.
        d[0] += alpha_i * e[0] * e[0]; // D_11
        d[1] += alpha_i * e[0] * e[1]; // D_12
        d[2] += alpha_i * e[0] * e[2]; // D_13
        d[3] += alpha_i * e[1] * e[1]; // D_22
        d[4] += alpha_i * e[1] * e[2]; // D_23
        d[5] += alpha_i * e[2] * e[2]; // D_33
    }
    d
}

/// Numerical floor for the denominator in diffusion eigenvalue construction.
pub(crate) const EPS: f64 = 1e-20;
