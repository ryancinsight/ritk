//! Displacement field composition Ï†â‚ âˆ˜ Ï†â‚‚.

#[cfg(test)]
use super::VelocityField;
use super::{trilinear_interpolate_field, VectorField, VectorFieldMut};
use ritk_spatial::VolumeDims;

/// Compute the composition `Ï†_composed = Ï†â‚ âˆ˜ Ï†â‚‚` into caller-provided buffers.
///
/// `Ï†_composed(x) = Ï†â‚(x + Ï†â‚‚(x))` â€” the combined displacement at each voxel
/// `x` is obtained by displacing `x` by `Ï†â‚‚(x)` and then sampling `Ï†â‚` at the
/// resulting position via trilinear interpolation.
///
/// Output buffers must have length `dims.total_voxels()`.
pub(crate) fn compose_fields_into(
    phi1: VectorField<'_>,
    phi2: VectorField<'_>,
    dims: VolumeDims,
    out: VectorFieldMut<'_>,
) {
    let [_nz, ny, nx] = dims.0;
    let VectorField {
        z: phi2_z,
        y: phi2_y,
        x: phi2_x } = phi2;
    let VectorFieldMut {
        z: out_z,
        y: out_y,
        x: out_x } = out;

    // Parallelize over z-slices: each slice writes to a disjoint contiguous
    // range in the output buffers; all reads are from immutable inputs.
    let slice_len = ny * nx;
    moirai::for_each_chunk_triple_mut_enumerated_with::<moirai::Adaptive, _, _, _, _>(
        out_z,
        out_y,
        out_x,
        slice_len,
        |iz, out_z_s, out_y_s, out_x_s| {
            let base = iz * slice_len;
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let fi = base + local;

                    // Displaced position x + Ï†â‚‚(x).
                    let wz = iz as f32 + phi2_z[fi];
                    let wy = iy as f32 + phi2_y[fi];
                    let wx = ix as f32 + phi2_x[fi];

                    // Sample Ï†â‚ at the displaced position with one shared stencil.
                    let [sample_z, sample_y, sample_x] =
                        trilinear_interpolate_field(phi1, dims, wz, wy, wx);
                    out_z_s[local] = phi2_z[fi] + sample_z;
                    out_y_s[local] = phi2_y[fi] + sample_y;
                    out_x_s[local] = phi2_x[fi] + sample_x;
                }
            }
        },
    );
}

/// Compute the composition `Ï†_composed = Ï†â‚ âˆ˜ Ï†â‚‚`.
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
        VectorField {
            z: phi1_z,
            y: phi1_y,
            x: phi1_x },
        VectorField {
            z: phi2_z,
            y: phi2_y,
            x: phi2_x },
        dims,
        VectorFieldMut {
            z: &mut cz,
            y: &mut cy,
            x: &mut cx },
    );

    VelocityField {
        z: cz,
        y: cy,
        x: cx }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity composition: Ï† âˆ˜ 0 = Ï†.
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

    /// 0 âˆ˜ Ï† = Ï†.
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
