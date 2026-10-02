use anyhow::Result;
use coeus_core::ComputeBackend;
use consus_core::ByteOrder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_codecs::sample::{write_samples, Sample, SampleType};
use ritk_image::Image;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::header::{HeaderDims, HeaderSpatial, HeaderVersion, NiftiHeader};
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
        SampleType::U32,
        origin.map(f64::from),
        spacing.map(f64::from),
        direction.map(f64::from),
    )?;

    write_single_file_with(path, &header, |writer| write_volume(writer, labels))
}

/// Write an image to a NIfTI-1 single-file stream with full sform metadata.
///
/// The samples are written in `T`, under the `datatype` code for `T`, with no
/// rescale.
///
/// # Spatial convention
///
/// RITK tensors are ordered `[Z, Y, X]`; NIfTI file axes are `[X, Y, Z]`.
/// The writer emits file columns `[internal X, internal Y, internal Z]`, i.e.
/// `[direction.col(2)*spacing[2], direction.col(1)*spacing[1],
/// direction.col(0)*spacing[0]]`, then converts LPS rows to RAS rows.
///
/// # Errors
///
/// Returns an error when the shape cannot be represented by a NIfTI-1 header
/// or when writing fails.
pub fn write_nifti<T, B, P>(path: P, image: &Image<T, B, 3>, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    write_nifti_with_version(HeaderVersion::One, path, image, backend)
}

/// Write an image to a NIfTI-2 single-file stream with full sform metadata.
///
/// The reader auto-detects NIfTI-1 and NIfTI-2. This writer is explicit so
/// callers do not silently change on-disk format when NIfTI-1 dimensions still
/// suffice. The samples are written in `T`, as by [`write_nifti`].
///
/// # Errors
///
/// Returns an error when writing fails.
pub fn write_nifti2<T, B, P>(path: P, image: &Image<T, B, 3>, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    write_nifti_with_version(HeaderVersion::Two, path, image, backend)
}

/// Write an acquisition series to a NIfTI-1 single-file stream.
///
/// Every volume must share one spatial grid — shape, origin, spacing, and
/// direction — because a NIfTI series has exactly one sform. The grid is taken
/// from the first volume and the rest are validated against it, so a caller that
/// assembled volumes from different images fails here rather than writing a file
/// whose geometry silently applies to only some of its content.
///
/// A one-volume series writes as an ordinary rank-3 file, byte-identical to
/// [`write_nifti`]; more volumes raise the header to rank 4 with the count in
/// `dim[4]`.
///
/// # Errors
///
/// Returns an error when `volumes` is empty, when any volume's grid differs from
/// the first, when the shape or volume count exceeds the header's capacity, or
/// when writing fails.
pub fn write_nifti_series<T, B, P>(path: P, volumes: &[Image<T, B, 3>], backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    write_series_with_version(HeaderVersion::One, path, volumes, backend)
}

/// Write an acquisition series to a NIfTI-2 single-file stream.
///
/// The NIfTI-2 counterpart of [`write_nifti_series`], with the same grid
/// agreement requirement and rank selection.
///
/// # Errors
///
/// Returns an error under the same conditions as [`write_nifti_series`].
pub fn write_nifti2_series<T, B, P>(path: P, volumes: &[Image<T, B, 3>], backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    write_series_with_version(HeaderVersion::Two, path, volumes, backend)
}

fn write_series_with_version<T, B, P>(
    version: HeaderVersion,
    path: P,
    volumes: &[Image<T, B, 3>],
    backend: &B,
) -> Result<()>
where
    T: Sample,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let Some((first, rest)) = volumes.split_first() else {
        anyhow::bail!("write_nifti_series: a series requires at least one volume");
    };

    let shape = first.shape();
    let origin = first.origin();
    let spacing = first.spacing();
    let direction = direction_row_major(first.direction());

    for (index, volume) in rest.iter().enumerate() {
        let position = index + 1;
        if volume.shape() != shape {
            anyhow::bail!(
                "write_nifti_series: volume {position} shape {:?} differs from volume 0 {shape:?}; \
                 a NIfTI series has one spatial grid",
                volume.shape()
            );
        }
        if volume.origin() != origin || volume.spacing() != spacing {
            anyhow::bail!(
                "write_nifti_series: volume {position} origin or spacing differs from volume 0; \
                 a NIfTI series has one spatial grid"
            );
        }
        if direction_row_major(volume.direction()) != direction {
            anyhow::bail!(
                "write_nifti_series: volume {position} direction differs from volume 0; \
                 a NIfTI series has one spatial grid"
            );
        }
    }

    let [nz, ny, nx] = shape;
    let expected = checked_voxel_count(nx, ny, nz)?;
    let payloads = volumes
        .iter()
        .map(|volume| volume.data_cow_on(backend))
        .collect::<Vec<_>>();
    for (position, payload) in payloads.iter().enumerate() {
        if payload.len() != expected {
            anyhow::bail!(
                "write_nifti_series: volume {position} data len {} != shape product {expected}",
                payload.len()
            );
        }
    }

    let header = header_from_spatial_with_volumes(
        version,
        HeaderDims { nx, ny, nz },
        payloads.len(),
        T::TYPE,
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction,
    )?;

    write_single_file_with(path, &header, |writer| {
        // The acquisition axis is slowest, so volumes serialize back to back in
        // acquisition order.
        for payload in &payloads {
            write_volume(writer, payload)?;
        }
        Ok(())
    })
}

fn write_nifti_with_version<T, B, P>(
    version: HeaderVersion,
    path: P,
    image: &Image<T, B, 3>,
    backend: &B,
) -> Result<()>
where
    T: Sample,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let shape = image.shape();
    let voxels = image.data_cow_on(backend);
    let origin = image.origin();
    let spacing = image.spacing();

    write_flat_with_version(
        version,
        path.as_ref(),
        &voxels,
        shape,
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major(image.direction()),
    )
}

/// Flatten a 3×3 direction-cosine matrix to the row-major layout the header
/// builder consumes.
fn direction_row_major(direction: &ritk_spatial::Direction<3>) -> [f64; 9] {
    let d = direction.0;
    [
        d[(0, 0)],
        d[(0, 1)],
        d[(0, 2)],
        d[(1, 0)],
        d[(1, 1)],
        d[(1, 2)],
        d[(2, 0)],
        d[(2, 1)],
        d[(2, 2)],
    ]
}

/// NIfTI serialization core: header plus the `[Z, Y, X]` voxel stream.
fn write_flat_with_version<T: Sample>(
    version: HeaderVersion,
    path: &Path,
    voxels: &[T],
    shape: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction_row_major: [f64; 9],
) -> Result<()> {
    let [nz, ny, nx] = shape;
    let expected = checked_voxel_count(nx, ny, nz)?;
    if voxels.len() != expected {
        anyhow::bail!(
            "write_nifti: image data len {} != shape product {}",
            voxels.len(),
            expected
        );
    }

    let header = header_from_spatial(
        version,
        HeaderDims { nx, ny, nz },
        T::TYPE,
        origin,
        spacing,
        direction_row_major,
    )?;

    write_single_file_with(path, &header, |writer| write_volume(writer, voxels))
}

/// Serialize one volume little-endian.
///
/// RITK's flat `[Z, Y, X]` order has x varying fastest, then y, then z, which
/// is NIfTI's file order, so the samples stream out as stored.
fn write_volume<T: Sample>(writer: &mut dyn Write, voxels: &[T]) -> Result<()> {
    write_samples(voxels, ByteOrder::LittleEndian, writer)?;
    Ok(())
}

fn header_from_spatial(
    version: HeaderVersion,
    dims: HeaderDims,
    sample_type: SampleType,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
) -> Result<NiftiHeader> {
    header_from_spatial_with_volumes(version, dims, 1, sample_type, origin, spacing, direction)
}

fn header_from_spatial_with_volumes(
    version: HeaderVersion,
    dims: HeaderDims,
    volumes: usize,
    sample_type: SampleType,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
) -> Result<NiftiHeader> {
    let sform = sform_from_internal_lps_metadata(origin, spacing, direction);
    // pixdim[4] is the acquisition-axis step. A diffusion series has no
    // meaningful step along it, so it stays at unity rather than carrying an
    // invented repetition time.
    let pixdim = [1.0, spacing[2], spacing[1], spacing[0], 1.0, 1.0, 1.0, 1.0];
    NiftiHeader::new_with_version(
        version,
        dims,
        volumes,
        sample_type,
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
