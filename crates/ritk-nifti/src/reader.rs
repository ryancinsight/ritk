use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject, ReaderOptions};
use ritk_core::image::Image;
use std::io::Cursor;
use std::path::Path;

use crate::spatial::metadata_from_nifti_ras_affine;

pub fn read_nifti<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let obj = ReaderOptions::new().read_file(path).map_err(|e| {
        tracing::error!("Failed to read NIfTI file {:?}: {}", path, e);
        anyhow!("Failed to read NIfTI file")
    })?;

    image_from_nifti_object::<B>(obj, device)
}

/// Read a NIfTI payload from in-memory bytes.
///
/// Accepts either `.nii` or `.nii.gz` encoded bytes and produces a 3-D image
/// in RITK ZYX tensor order.
pub fn read_nifti_from_bytes<B: Backend>(bytes: &[u8], device: &B::Device) -> Result<Image<B, 3>> {
    let cursor = Cursor::new(bytes);
    let obj = InMemNiftiObject::from_reader(cursor).map_err(|e| {
        tracing::error!("Failed to read NIfTI bytes: {}", e);
        anyhow!("Failed to read NIfTI bytes")
    })?;

    image_from_nifti_object::<B>(obj, device)
}

fn image_from_nifti_object<B: Backend>(
    obj: InMemNiftiObject,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let header = obj.header();

    // Sform
    let affine = if header.sform_code > 0 {
        let r0 = header.srow_x;
        let r1 = header.srow_y;
        let r2 = header.srow_z;
        // Rows to [[f32; 4]; 4]
        [r0, r1, r2, [0.0, 0.0, 0.0, 1.0]]
    } else if header.qform_code > 0 {
        // Qform implementation
        // See NIfTI standard
        let b = header.quatern_b;
        let c = header.quatern_c;
        let d = header.quatern_d;
        let a = qform_quaternion_scalar(b, c, d)?;

        let qfac = qfac_from_pixdim(header.pixdim[0])?;

        let r11 = a * a + b * b - c * c - d * d;
        let r12 = 2.0 * b * c - 2.0 * a * d;
        let r13 = 2.0 * b * d + 2.0 * a * c;

        let r21 = 2.0 * b * c + 2.0 * a * d;
        let r22 = a * a + c * c - b * b - d * d;
        let r23 = 2.0 * c * d - 2.0 * a * b;

        let r31 = 2.0 * b * d - 2.0 * a * c;
        let r32 = 2.0 * c * d + 2.0 * a * b;
        let r33 = a * a + d * d - c * c - b * b;

        let [dx, dy, dz_abs] = checked_spatial_pixdim(header.pixdim)?;
        let dz = dz_abs * qfac;

        let qx = header.quatern_x;
        let qy = header.quatern_y;
        let qz = header.quatern_z;

        [
            [r11 * dx, r12 * dy, r13 * dz, qx],
            [r21 * dx, r22 * dy, r23 * dz, qy],
            [r31 * dx, r32 * dy, r33 * dz, qz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else {
        // Fallback: use pixdim scaling only
        let [dx, dy, dz] = checked_spatial_pixdim(header.pixdim)?;
        [
            [dx, 0.0, 0.0, 0.0],
            [0.0, dy, 0.0, 0.0],
            [0.0, 0.0, dz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    let spatial =
        metadata_from_nifti_ras_affine(affine).context("Invalid NIfTI spatial metadata")?;

    // Load voxel data
    let volume = obj.into_volume();
    // We load as f32. Burn tensors are typically float.
    // Note: this loads entire volume into memory (CPU) first.
    let ndarray_volume = volume
        .into_ndarray::<f32>()
        .context("Failed to convert volume to ndarray")?;

    // ndarray shape
    let shape = ndarray_volume.shape();
    // Assuming 3D for now based on Image<B, 3> return type.
    if shape.len() != 3 {
        anyhow::bail!("Expected 3D NIfTI file, found {} dimensions", shape.len());
    }
    let dim0 = shape[0];
    let dim1 = shape[1];
    let dim2 = shape[2];

    // Rebuild the internal ZYX buffer by logical coordinates. The nifti crate
    // exposes the volume as [x, y, z]; RITK stores tensors as [z, y, x].
    use ndarray::Ix3;
    let arr = ndarray_volume
        .into_dimensionality::<Ix3>()
        .context("Failed to convert NIfTI ndarray to Ix3")?;

    let voxel_count = checked_voxel_count(dim0, dim1, dim2)?;
    let mut data_vec = vec![0.0_f32; voxel_count];
    for z in 0..dim2 {
        for y in 0..dim1 {
            for x in 0..dim0 {
                data_vec[z * dim1 * dim0 + y * dim0 + x] = arr[[x, y, z]];
            }
        }
    }

    let shape_burn = Shape::new([dim2, dim1, dim0]);
    let data = TensorData::new(data_vec, shape_burn);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    Ok(Image::new(
        tensor,
        spatial.origin,
        spatial.spacing,
        spatial.direction,
    ))
}

/// Read a NIfTI file as an integer label map in ZYX order.
///
/// # Label extraction
///
/// The file is read as `f32` (consistent with [`read_nifti`]) and each value
/// is converted to `u32` via `max(0.0).round() as u32`.  This is exact for
/// integer label values ≤ 2²⁴ (the f32 mantissa width), covering all
/// practical segmentation label counts.
///
/// # Returns
///
/// `(labels, [nz, ny, nx])` where `labels` is flat ZYX-order:
/// `labels[z * ny * nx + y * nx + x]`.
///
/// # Errors
///
/// Returns `Err` when the file cannot be read or does not contain a 3-D volume.
pub fn read_nifti_labels<P: AsRef<Path>>(path: P) -> Result<(Vec<u32>, [usize; 3])> {
    let path = path.as_ref();
    let obj = ReaderOptions::new().read_file(path).map_err(|e| {
        tracing::error!("Failed to read NIfTI label file: {}", e);
        anyhow!("Failed to read NIfTI label file")
    })?;

    let volume = obj.into_volume();
    let ndarray_volume = volume
        .into_ndarray::<f32>()
        .context("Failed to convert label volume to ndarray")?;

    let shape = ndarray_volume.shape();
    if shape.len() != 3 {
        anyhow::bail!(
            "Expected 3-D NIfTI label file, found {} dimensions",
            shape.len()
        );
    }
    // NIfTI stores (nx, ny, nz); ZYX shape = [nz, ny, nx].
    let nx = shape[0];
    let ny = shape[1];
    let nz = shape[2];

    // Use logical array indexing so the extraction is independent of
    // the nifti crate's in-memory layout (which may be F-order / x-fastest).
    // NIfTI dim convention: dim[1]=nx, dim[2]=ny, dim[3]=nz → shape (nx,ny,nz),
    // array[[x,y,z]] = voxel at coordinates (x,y,z).
    use ndarray::Ix3;
    let arr = ndarray_volume
        .into_dimensionality::<Ix3>()
        .map_err(|e| anyhow::anyhow!("read_nifti_labels: dimensionality error: {e}"))?;

    let voxel_count = checked_voxel_count(nx, ny, nz)?;
    let mut labels = vec![0u32; voxel_count];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let src = arr[[x, y, z]];
                labels[z * ny * nx + y * nx + x] = src.max(0.0).round() as u32;
            }
        }
    }

    Ok((labels, [nz, ny, nx]))
}

fn checked_voxel_count(nx: usize, ny: usize, nz: usize) -> Result<usize> {
    nx.checked_mul(ny)
        .and_then(|xy| xy.checked_mul(nz))
        .ok_or_else(|| anyhow!("NIfTI voxel count overflows usize: nx={nx}, ny={ny}, nz={nz}"))
}

fn checked_spatial_pixdim(pixdim: [f32; 8]) -> Result<[f32; 3]> {
    let spatial = [pixdim[1], pixdim[2], pixdim[3]];
    for (offset, value) in spatial.iter().enumerate() {
        let index = offset + 1;
        if !value.is_finite() || *value <= 0.0 {
            anyhow::bail!("NIfTI pixdim[{index}] must be positive and finite, got {value}");
        }
    }

    Ok(spatial)
}

fn qfac_from_pixdim(value: f32) -> Result<f32> {
    if !value.is_finite() {
        anyhow::bail!("NIfTI pixdim[0] qfac must be finite, got {value}");
    }

    if value == 0.0 || value == 1.0 {
        Ok(1.0)
    } else if value == -1.0 {
        Ok(-1.0)
    } else {
        anyhow::bail!("NIfTI pixdim[0] qfac must be -1, 0, or 1, got {value}");
    }
}

fn qform_quaternion_scalar(b: f32, c: f32, d: f32) -> Result<f32> {
    for (name, value) in [("b", b), ("c", c), ("d", d)] {
        if !value.is_finite() {
            anyhow::bail!("NIfTI qform quaternion {name} must be finite, got {value}");
        }
    }

    let squared_vector_norm = b.mul_add(b, c.mul_add(c, d * d));
    if squared_vector_norm > 1.0 + 1.0e-5 {
        anyhow::bail!(
            "NIfTI qform quaternion vector norm squared must be <= 1, got {squared_vector_norm}"
        );
    }

    Ok((1.0 - squared_vector_norm).max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::{
        checked_spatial_pixdim, checked_voxel_count, qfac_from_pixdim, qform_quaternion_scalar,
    };

    #[test]
    fn checked_voxel_count_multiplies_dimensions() {
        assert_eq!(
            checked_voxel_count(4, 3, 2).expect("small dimensions must multiply"),
            24
        );
    }

    #[test]
    fn checked_voxel_count_rejects_overflow() {
        let err = checked_voxel_count(usize::MAX, 2, 1)
            .expect_err("overflowing NIfTI dimensions must be rejected");

        assert!(
            err.to_string().contains("overflows usize"),
            "error must name overflow invariant: {err}"
        );
    }

    #[test]
    fn checked_spatial_pixdim_rejects_zero() {
        let err = checked_spatial_pixdim([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect_err("zero NIfTI spacing must be rejected");

        assert!(
            err.to_string().contains("pixdim[2]"),
            "error must name offending pixdim index: {err}"
        );
    }

    #[test]
    fn qfac_accepts_standard_values() {
        assert_eq!(qfac_from_pixdim(0.0).expect("0 qfac maps to +1"), 1.0);
        assert_eq!(qfac_from_pixdim(1.0).expect("+1 qfac is valid"), 1.0);
        assert_eq!(qfac_from_pixdim(-1.0).expect("-1 qfac is valid"), -1.0);
    }

    #[test]
    fn qfac_rejects_non_standard_value() {
        let err = qfac_from_pixdim(2.0).expect_err("non-standard qfac must be rejected");

        assert!(
            err.to_string().contains("qfac"),
            "error must name qfac invariant: {err}"
        );
    }

    #[test]
    fn qform_quaternion_rejects_impossible_norm() {
        let err = qform_quaternion_scalar(1.0, 1.0, 0.0)
            .expect_err("impossible qform quaternion must be rejected");

        assert!(
            err.to_string().contains("norm squared"),
            "error must name quaternion norm invariant: {err}"
        );
    }
}
