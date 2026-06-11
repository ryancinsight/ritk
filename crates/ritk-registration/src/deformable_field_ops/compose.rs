//! Displacement field composition φ₁ ∘ φ₂.

#[cfg(test)]
use super::VelocityField;
use super::{trilinear_interpolate, VectorField3D, VectorFieldMut3D};
use crate::parallel::CellSlice;
use ritk_spatial::VolumeDims;

/// Compute the composition `φ_composed = φ₁ ∘ φ₂` into caller-provided buffers.
///
/// `φ_composed(x) = φ₁(x + φ₂(x))` — the combined displacement at each voxel
/// `x` is obtained by displacing `x` by `φ₂(x)` and then sampling `φ₁` at the
/// resulting position via trilinear interpolation.
///
/// Output buffers must have length `dims.total_voxels()`.
pub(crate) fn compose_fields_into(
    phi1: VectorField3D<'_>,
    phi2: VectorField3D<'_>,
    dims: VolumeDims,
    out: VectorFieldMut3D<'_>,
) {
    let [nz, ny, nx] = dims.0;
    let VectorField3D {
        z: phi1_z,
        y: phi1_y,
        x: phi1_x,
    } = phi1;
    let VectorField3D {
        z: phi2_z,
        y: phi2_y,
        x: phi2_x,
    } = phi2;
    let VectorFieldMut3D {
        z: out_z,
        y: out_y,
        x: out_x,
    } = out;

    // Parallelize over z-slices: each slice writes to a disjoint contiguous
    // range in the output buffers; all reads are from immutable inputs.
    let slice_len = ny * nx;
    let out_z = CellSlice::from_mut(out_z);
    let out_y = CellSlice::from_mut(out_y);
    let out_x = CellSlice::from_mut(out_x);
    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        // SAFETY: out_z/out_y/out_x each have length nz*ny*nx and are split
        // at identical disjoint z-slice boundaries; each thread writes only
        // to its own [base, base + slice_len) range.
        let out_z_s = unsafe { out_z.slice_mut(base, slice_len) };
        let out_y_s = unsafe { out_y.slice_mut(base, slice_len) };
        let out_x_s = unsafe { out_x.slice_mut(base, slice_len) };
        for iy in 0..ny {
            for ix in 0..nx {
                let local = iy * nx + ix;
                let fi = base + local;

                // Displaced position x + φ₂(x).
                let wz = iz as f32 + phi2_z[fi];
                let wy = iy as f32 + phi2_y[fi];
                let wx = ix as f32 + phi2_x[fi];

                // Sample φ₁ at the displaced position.
                out_z_s[local] = phi2_z[fi] + trilinear_interpolate(phi1_z, dims, wz, wy, wx);
                out_y_s[local] = phi2_y[fi] + trilinear_interpolate(phi1_y, dims, wz, wy, wx);
                out_x_s[local] = phi2_x[fi] + trilinear_interpolate(phi1_x, dims, wz, wy, wx);
            }
        }
    });
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
    dims: VolumeDims,
) -> VelocityField {
    let [nz, ny, nx] = dims.0;
    let n = nz * ny * nx;

    let mut cz = vec![0.0_f32; n];
    let mut cy = vec![0.0_f32; n];
    let mut cx = vec![0.0_f32; n];

    compose_fields_into(
        VectorField3D {
            z: phi1_z,
            y: phi1_y,
            x: phi1_x,
        },
        VectorField3D {
            z: phi2_z,
            y: phi2_y,
            x: phi2_x,
        },
        dims,
        VectorFieldMut3D {
            z: &mut cz,
            y: &mut cy,
            x: &mut cx,
        },
    );

    VelocityField {
        z: cz,
        y: cy,
        x: cx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity composition: φ ∘ 0 = φ.
    #[test]
    fn compose_with_zero_is_identity() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let phiz: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let phiy: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.01).collect();
        let phix = vec![0.5_f32; n];
        let zero = vec![0.0_f32; n];

        let comp = compose_fields(&phiz, &phiy, &phix, &zero, &zero, &zero, dims);

        for i in 0..n {
            assert!(
                (comp.z[i] - phiz[i]).abs() < 1e-4,
                "cz[{i}]: expected {}, got {}",
                phiz[i],
                comp.z[i]
            );
            assert!(
                (comp.y[i] - phiy[i]).abs() < 1e-4,
                "cy[{i}]: expected {}, got {}",
                phiy[i],
                comp.y[i]
            );
            assert!(
                (comp.x[i] - phix[i]).abs() < 1e-4,
                "cx[{i}]: expected {}, got {}",
                phix[i],
                comp.x[i]
            );
        }
    }

    /// 0 ∘ φ = φ.
    #[test]
    fn compose_zero_with_field_is_field() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let phi: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let zero = vec![0.0_f32; n];

        let comp = compose_fields(&zero, &zero, &zero, &phi, &phi, &phi, dims);

        for (i, p) in phi.iter().copied().enumerate() {
            assert!((comp.z[i] - p).abs() < 1e-4, "cz[{i}]: expected {p}");
            assert!((comp.y[i] - p).abs() < 1e-4, "cy[{i}]: expected {p}");
            assert!((comp.x[i] - p).abs() < 1e-4, "cx[{i}]: expected {p}");
        }
    }
}
