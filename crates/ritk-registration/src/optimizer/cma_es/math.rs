//! Internal linear algebra derivations for CMA-ES covariance mappings.
//! Exposes exact flat-array routines skipping ND runtime allocations.

/// Returns n×n identity matrix natively as a flat row-major array.
pub(crate) fn identity(n: usize) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Euclidean norm of an unrolled generic float vector natively bounded.
pub(crate) fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Lower-triangular Cholesky decomposition of a symmetric positive-definite matrix natively.
/// Input `a` is a flat row-major n×n matrix. Output is a flat lower-triangular packed array.
pub(crate) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * (n + 1) / 2];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i * n + j];
            for k in 0..j {
                let idx_ik = i * (i + 1) / 2 + k;
                let idx_jk = j * (j + 1) / 2 + k;
                s -= l[idx_ik] * l[idx_jk];
            }
            let idx_ij = i * (i + 1) / 2 + j;
            if i == j {
                if s <= 0.0 {
                    return None; // Invalid or non-positive-definite invariant
                }
                l[idx_ij] = s.sqrt();
            } else {
                let idx_jj = j * (j + 1) / 2 + j;
                l[idx_ij] = s / l[idx_jj];
            }
        }
    }
    Some(l)
}

/// Exactly multiplies the lower-triangular Cholesky factor A (packed style) directly crossing z.
pub(crate) fn chol_mul(chol: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..=i {
            let idx_ij = i * (i + 1) / 2 + j;
            y[i] += chol[idx_ij] * z[j];
        }
    }
    y
}

/// Solves exactly bounded A·x = b substituting `x`, where `A` reflects lower-triangular Cholesky style natively.
pub(crate) fn chol_solve_lower(chol: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            let idx_ij = i * (i + 1) / 2 + j;
            x[i] -= chol[idx_ij] * x[j];
        }
        let idx_ii = i * (i + 1) / 2 + i;
        x[i] /= chol[idx_ii];
    }
    x
}
