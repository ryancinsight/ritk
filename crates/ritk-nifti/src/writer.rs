use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use nifti::NiftiHeader;
use ritk_core::image::Image;
use std::path::Path;

use crate::shape::checked_voxel_count;
use crate::spatial::sform_from_internal_lps_metadata;

/// Write a label map to a NIfTI-1 file with `DT_UINT32` data type.
///
/// # Spatial convention
///
/// `shape` is `[nz, ny, nx]` (ZYX). `spacing` is `[dz, dy, dx]`.
/// `direction` is the 3x3 direction-cosine matrix in **row-major flat layout**,
/// matching `LoadedVolume::direction`.  The sform affine follows the same
/// convention as [`write_nifti`]: it maps NIfTI file axes `[x,y,z]` to RAS by
/// reordering internal `[depth,row,col]` columns to `[col,row,depth]` and
/// flipping the first two physical rows from LPS to RAS.
///
/// ```text
/// A_lps_internal = [D[:,0]*dz, D[:,1]*dy, D[:,2]*dx, origin]
/// A_ras_nifti[:,0] = lps_to_ras(A_lps_internal[:,2])
/// A_ras_nifti[:,1] = lps_to_ras(A_lps_internal[:,1])
/// A_ras_nifti[:,2] = lps_to_ras(A_lps_internal[:,0])
/// ```
///
/// # Errors
///
/// Returns `Err` when `labels.len() != nz * ny * nx` or when the nifti
/// writer reports a failure.
pub fn write_nifti_labels<P: AsRef<Path>>(
    path: P,
    labels: &[u32],
    shape: [usize; 3], // [nz, ny, nx]
    origin: [f32; 3],
    spacing: [f32; 3],   // [dz, dy, dx]
    direction: [f32; 9], // row-major 3×3
) -> Result<()> {
    use ndarray::Array3;
    use nifti::writer::WriterOptions;

    let [nz, ny, nx] = shape;
    let expected = checked_voxel_count(nx, ny, nz)?;
    if labels.len() != expected {
        anyhow::bail!(
            "write_nifti_labels: labels.len()={} != shape product {}",
            labels.len(),
            expected
        );
    }

    // Fill the ndarray using logical indexing so the writer is independent of
    // the in-memory layout chosen by ndarray.  The NIfTI crate treats dim[1]=x,
    // dim[2]=y, dim[3]=z, so the array shape must be (nx, ny, nz) with
    // array[[x, y, z]] = label at RITK ZYX position (z, y, x).
    let mut array = Array3::<u32>::zeros((nx, ny, nz));
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                array[[x, y, z]] = labels[z * ny * nx + y * nx + x];
            }
        }
    }

    let origin64 = [origin[0] as f64, origin[1] as f64, origin[2] as f64];
    let spacing64 = [spacing[0] as f64, spacing[1] as f64, spacing[2] as f64];
    let direction64 = [
        direction[0] as f64,
        direction[1] as f64,
        direction[2] as f64,
        direction[3] as f64,
        direction[4] as f64,
        direction[5] as f64,
        direction[6] as f64,
        direction[7] as f64,
        direction[8] as f64,
    ];
    let sform = sform_from_internal_lps_metadata(origin64, spacing64, direction64);

    let header = NiftiHeader {
        sform_code: 1,
        qform_code: 0,
        srow_x: sform.x,
        srow_y: sform.y,
        srow_z: sform.z,
        pixdim: [1.0, spacing[2], spacing[1], spacing[0], 1.0, 1.0, 1.0, 1.0],
        xyzt_units: 2, // NIFTI_UNITS_MM
        ..NiftiHeader::default()
    };

    WriterOptions::new(path.as_ref())
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|e| anyhow::anyhow!("Failed to write NIfTI labels: {e}"))?;

    Ok(())
}

/// Write an image to a NIfTI file with full sform spatial metadata.
///
/// # Spatial convention
/// RITK tensors are ordered `[Z, Y, X]`; NIfTI file axes are `[X, Y, Z]`.
/// The writer emits file columns `[internal X, internal Y, internal Z]`, i.e.
/// `[direction.col(2)*spacing[2], direction.col(1)*spacing[1],
/// direction.col(0)*spacing[0]]`, then converts LPS rows to RAS rows.
pub fn write_nifti<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    use ndarray::Array3;
    use nifti::writer::WriterOptions;

    let shape = image.shape();
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];
    let expected = checked_voxel_count(nx, ny, nz)?;

    let slice = &image
        .try_data_vec()
        .context("NIfTI writer requires f32 image data")?;
    if slice.len() != expected {
        anyhow::bail!(
            "write_nifti: image data len {} != shape product {}",
            slice.len(),
            expected
        );
    }

    let mut array = Array3::<f32>::zeros((nx, ny, nz));
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                array[[x, y, z]] = slice[z * ny * nx + y * nx + x];
            }
        }
    }

    let path_ref = path.as_ref();

    let direction = image.direction().0;
    let origin = image.origin();
    let spacing = image.spacing();

    let direction_row_major = [
        direction[(0, 0)],
        direction[(0, 1)],
        direction[(0, 2)],
        direction[(1, 0)],
        direction[(1, 1)],
        direction[(1, 2)],
        direction[(2, 0)],
        direction[(2, 1)],
        direction[(2, 2)],
    ];
    let sform = sform_from_internal_lps_metadata(
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major,
    );

    let header = NiftiHeader {
        sform_code: 1,
        qform_code: 0,
        srow_x: sform.x,
        srow_y: sform.y,
        srow_z: sform.z,
        pixdim: [
            1.0,
            spacing[2] as f32,
            spacing[1] as f32,
            spacing[0] as f32,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        xyzt_units: 2, // NIFTI_UNITS_MM
        ..NiftiHeader::default()
    };

    WriterOptions::new(path_ref)
        .reference_header(&header)
        .write_nifti(&array)
        .map_err(|e| anyhow::anyhow!("Failed to write NIfTI file: {}", e))?;

    Ok(())
}
