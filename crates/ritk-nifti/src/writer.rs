use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_core::image::Image;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::header::{HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader};
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
/// # Errors
///
/// Returns `Err` when `labels.len() != nz * ny * nx`, when the shape cannot be
/// represented by a NIfTI-1 header, or when writing fails.
pub fn write_nifti_labels<P: AsRef<Path>>(
    path: P,
    labels: &[u32],
    shape: [usize; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
    direction: [f32; 9],
) -> Result<()> {
    write_nifti_labels_with_version(
        HeaderVersion::One,
        path,
        labels,
        shape,
        origin,
        spacing,
        direction,
    )
}

/// Write a label map to a NIfTI-2 file with `DT_UINT32` data type.
///
/// This emits the native single-file `.nii`/`.nii.gz` NIfTI-2 header (`n+2`)
/// and the same ZYX-to-XYZ voxel ordering and LPS-to-RAS sform convention as
/// [`write_nifti_labels`].
pub fn write_nifti2_labels<P: AsRef<Path>>(
    path: P,
    labels: &[u32],
    shape: [usize; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
    direction: [f32; 9],
) -> Result<()> {
    write_nifti_labels_with_version(
        HeaderVersion::Two,
        path,
        labels,
        shape,
        origin,
        spacing,
        direction,
    )
}

fn write_nifti_labels_with_version<P: AsRef<Path>>(
    version: HeaderVersion,
    path: P,
    labels: &[u32],
    shape: [usize; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
    direction: [f32; 9],
) -> Result<()> {
    let [nz, ny, nx] = shape;
    let expected = checked_voxel_count(nx, ny, nz)?;
    if labels.len() != expected {
        anyhow::bail!(
            "write_nifti_labels: labels.len()={} != shape product {}",
            labels.len(),
            expected
        );
    }

    let header = header_from_spatial(
        version,
        HeaderDims { nx, ny, nz },
        NiftiDatatype::Uint32,
        origin.map(f64::from),
        spacing.map(f64::from),
        [
            direction[0] as f64,
            direction[1] as f64,
            direction[2] as f64,
            direction[3] as f64,
            direction[4] as f64,
            direction[5] as f64,
            direction[6] as f64,
            direction[7] as f64,
            direction[8] as f64,
        ],
    )?;

    write_single_file_with(path, &header, |writer| {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    writer.write_all(&labels[z * ny * nx + y * nx + x].to_le_bytes())?;
                }
            }
        }
        Ok(())
    })
}

/// Write an image to a NIfTI-1 single-file stream with full sform metadata.
///
/// # Spatial convention
///
/// RITK tensors are ordered `[Z, Y, X]`; NIfTI file axes are `[X, Y, Z]`.
/// The writer emits file columns `[internal X, internal Y, internal Z]`, i.e.
/// `[direction.col(2)*spacing[2], direction.col(1)*spacing[1],
/// direction.col(0)*spacing[0]]`, then converts LPS rows to RAS rows.
pub fn write_nifti<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    write_nifti_with_version(HeaderVersion::One, path, image)
}

/// Write an image to a NIfTI-2 single-file stream with full sform metadata.
///
/// The reader auto-detects NIfTI-1 and NIfTI-2. This writer is explicit so
/// callers do not silently change on-disk format when NIfTI-1 dimensions still
/// suffice.
pub fn write_nifti2<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    write_nifti_with_version(HeaderVersion::Two, path, image)
}

fn write_nifti_with_version<B: Backend, P: AsRef<Path>>(
    version: HeaderVersion,
    path: P,
    image: &Image<B, 3>,
) -> Result<()> {
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
    let header = header_from_spatial(
        version,
        HeaderDims { nx, ny, nz },
        NiftiDatatype::Float32,
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major,
    )?;

    write_single_file_with(path, &header, |writer| {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    writer.write_all(&slice[z * ny * nx + y * nx + x].to_le_bytes())?;
                }
            }
        }
        Ok(())
    })
}

fn header_from_spatial(
    version: HeaderVersion,
    dims: HeaderDims,
    datatype: NiftiDatatype,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
) -> Result<NiftiHeader> {
    let sform = sform_from_internal_lps_metadata(origin, spacing, direction);
    let pixdim = [1.0, spacing[2], spacing[1], spacing[0], 1.0, 1.0, 1.0, 1.0];
    NiftiHeader::new_3d_with_version(
        version,
        dims,
        datatype,
        HeaderSpatial {
            pixdim,
            srow_x: sform.x.map(f64::from),
            srow_y: sform.y.map(f64::from),
            srow_z: sform.z.map(f64::from),
        },
    )
}

fn write_single_file_with<P, F>(path: P, header: &NiftiHeader, write_payload: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut dyn Write) -> Result<()>,
{
    let path = path.as_ref();
    if is_gzip_path(path) {
        let file = File::create(path)?;
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::fast());
        write_header(&mut encoder, header)?;
        write_payload(&mut encoder)?;
        encoder.finish()?;
    } else {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, header)?;
        write_payload(&mut writer)?;
        writer.flush()?;
    }
    Ok(())
}

fn write_header(mut writer: impl Write, header: &NiftiHeader) -> Result<()> {
    writer.write_all(&header.encode())?;
    writer.write_all(&[0, 0, 0, 0])?;
    Ok(())
}

fn is_gzip_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
}
