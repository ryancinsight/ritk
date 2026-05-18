//! Displacement field composition φ₁ ∘ φ₂.

use super::{flat, trilinear_interpolate};

/// Compute the composition `φ_composed = φ₁ ∘ φ₂` into caller-provided buffers.
///
/// `φ_composed(x) = φ₁(x + φ₂(x))` — the combined displacement at each voxel
/// `x` is obtained by displacing `x` by `φ₂(x)` and then sampling `φ₁` at the
/// resulting position via trilinear interpolation.
///
/// Output buffers must have length `dims[0] * dims[1] * dims[2]`.
pub(crate) fn compose_fields_into(
    phi1_z: &[f32],
    phi1_y: &[f32],
    phi1_x: &[f32],
    phi2_z: &[f32],
    phi2_y: &[f32],
    phi2_x: &[f32],
    dims: [usize; 3],
    out_z: &mut [f32],
    out_y: &mut [f32],
    out_x: &mut [f32],
) {
    let [nz, ny, nx] = dims;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);

                // Displaced position x + φ₂(x).
                let wz = iz as f32 + phi2_z[fi];
                let wy = iy as f32 + phi2_y[fi];
                let wx = ix as f32 + phi2_x[fi];

                // Sample φ₁ at the displaced position.
                out_z[fi] = phi2_z[fi] + trilinear_interpolate(phi1_z, dims, wz, wy, wx);
                out_y[fi] = phi2_y[fi] + trilinear_interpolate(phi1_y, dims, wz, wy, wx);
                out_x[fi] = phi2_x[fi] + trilinear_interpolate(phi1_x, dims, wz, wy, wx);
            }
        }
    }
}

/// Compute the composition `φ_composed = φ₁ ∘ φ₂`.
#[cfg(test)]
pub(crate) fn compose_fields(
    phi1_z: &[f32],
    phi1_y: &[f32],
    phi1_x: &[f32],
    phi2_z: &[f32],
    phi2_y: &[f32],
    phi2_x: &[f32],
    dims: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut cz = vec![0.0_f32; n];
    let mut cy = vec![0.0_f32; n];
    let mut cx = vec![0.0_f32; n];

    compose_fields_into(
        phi1_z, phi1_y, phi1_x, phi2_z, phi2_y, phi2_x, dims, &mut cz, &mut cy, &mut cx,
    );

    (cz, cy, cx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity composition: φ ∘ 0 = φ.
    #[test]
    fn compose_with_zero_is_identity() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let phiz: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let phiy: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.01).collect();
        let phix = vec![0.5_f32; n];
        let zero = vec![0.0_f32; n];

        let (cz, cy, cx) = compose_fields(&phiz, &phiy, &phix, &zero, &zero, &zero, dims);

        for i in 0..n {
            assert!(
                (cz[i] - phiz[i]).abs() < 1e-4,
                "cz[{i}]: expected {}, got {}",
                phiz[i],
                cz[i]
            );
            assert!(
                (cy[i] - phiy[i]).abs() < 1e-4,
                "cy[{i}]: expected {}, got {}",
                phiy[i],
                cy[i]
            );
            assert!(
                (cx[i] - phix[i]).abs() < 1e-4,
                "cx[{i}]: expected {}, got {}",
                phix[i],
                cx[i]
            );
        }
    }

    /// 0 ∘ φ = φ.
    #[test]
    fn compose_zero_with_field_is_field() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let phi: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let zero = vec![0.0_f32; n];

        let (cz, cy, cx) = compose_fields(&zero, &zero, &zero, &phi, &phi, &phi, dims);

        for i in 0..n {
            assert!(
                (cz[i] - phi[i]).abs() < 1e-4,
                "cz[{i}]: expected {}",
                phi[i]
            );
            assert!(
                (cy[i] - phi[i]).abs() < 1e-4,
                "cy[{i}]: expected {}",
                phi[i]
            );
            assert!(
                (cx[i] - phi[i]).abs() < 1e-4,
                "cx[{i}]: expected {}",
                phi[i]
            );
        }
    }
}
