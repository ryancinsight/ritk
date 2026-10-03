//! VTK legacy structured points format writer.
//!
//! Writes `DATASET STRUCTURED_POINTS` files in **BINARY** encoding with
//! scalar data stored as big-endian `float` (IEEE 754 single precision).
//!
//! ## Coordinate Convention
//!
//! RITK tensor shape is **[nz, ny, nx]** (Z varies slowest, X varies fastest).
//! VTK `DIMENSIONS` expects **[nx, ny, nz]** order, so the first and last
//! tensor dimensions are swapped when emitting the header.
//!
//! RITK spacing and direction columns follow tensor axes **[Z, Y, X]**. The
//! writer maps spacing to VTK **[X, Y, Z]** and accepts only the corresponding
//! VTK-aligned direction. Spacing components must be finite and strictly
//! positive; image geometry is checked before opening the destination. Physical
//! origin remains XYZ.
//!
//! VTK stores scalar data with X varying fastest, matching RITK's memory
//! layout. No data permutation is required.

use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Encode flat voxel data plus geometry as a VTK legacy structured-points
/// stream (BINARY) into an arbitrary writer.
///
/// This is the shared, substrate-free core underlying both the coeus-backed
/// [`write_vtk`] and the Coeus-backed `ritk_io` native writer: identical byte
/// output given identical inputs, since neither carrier participates in the
/// encode.
///
/// ## Argument convention
///
/// - `slice` is row-major scalar data with X varying fastest, Y next, Z slowest
///   (matching RITK's `[nz, ny, nx]` tensor memory layout); it is emitted
///   verbatim as big-endian IEEE 754 single-precision (`f32`) with no
///   permutation.
/// - `dims` is `[nz, ny, nx]` — RITK tensor order (Z slowest, X fastest); the
///   emitted `DIMENSIONS` header field is permuted to VTK **[X, Y, Z]** order.
/// - `origin` / `spacing` are `[ox, oy, oz]` / `[sx, sy, sz]` in VTK **[X, Y, Z]**
///   order, matching the `ORIGIN` / `SPACING` fields directly.
///
/// The header is always ASCII (VTK's `BINARY` declaration governs only the data
/// section). The writer is flushed before return.
///
/// This flat-data encoder has no direction or coordinate-map inputs. Callers
/// writing an [`Image`] should use [`write_vtk`], which checks that its
/// geometry is representable before touching the destination.
///
/// # Errors
///
/// Returns an error when the writer fails, or when `slice.len()` does not equal
/// the positive, overflow-checked product of `dims`, or when spacing is not
/// finite and strictly positive.
pub fn encode_vtk_flat<W: Write>(
    writer: &mut W,
    slice: &[f32],
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Result<()> {
    let [nz, ny, nx] = dims;
    let total_voxels = crate::io::structured_points::voxel_count([nx, ny, nz])?;
    crate::io::structured_points::validate_spacing(spacing)?;
    anyhow::ensure!(
        slice.len() == total_voxels,
        "data contains {} elements but expected {} ({}×{}×{})",
        slice.len(),
        total_voxels,
        nx,
        ny,
        nz
    );

    let [ox, oy, oz] = origin;
    let [sx, sy, sz] = spacing;

    tracing::debug!(
        nx,
        ny,
        nz,
        ox,
        oy,
        oz,
        sx,
        sy,
        sz,
        "VTK writer: emitting {} voxels",
        total_voxels
    );

    // --- Write header ---
    //
    // VTK legacy format requires lines terminated by '\n'. The header is
    // always ASCII regardless of the BINARY/ASCII declaration (which only
    // governs the data section).

    writeln!(writer, "# vtk DataFile Version 3.0")
        .with_context(|| "failed to write VTK version line")?;
    writeln!(writer, "RITK exported image")
        .with_context(|| "failed to write VTK description line")?;
    writeln!(writer, "BINARY").with_context(|| "failed to write VTK encoding line")?;
    writeln!(writer, "DATASET STRUCTURED_POINTS")
        .with_context(|| "failed to write VTK dataset line")?;
    writeln!(writer, "DIMENSIONS {} {} {}", nx, ny, nz)
        .with_context(|| "failed to write VTK DIMENSIONS")?;
    writeln!(writer, "ORIGIN {} {} {}", ox, oy, oz)
        .with_context(|| "failed to write VTK ORIGIN")?;
    writeln!(writer, "SPACING {} {} {}", sx, sy, sz)
        .with_context(|| "failed to write VTK SPACING")?;
    writeln!(writer, "POINT_DATA {}", total_voxels)
        .with_context(|| "failed to write VTK POINT_DATA")?;
    writeln!(writer, "SCALARS scalars float 1").with_context(|| "failed to write VTK SCALARS")?;
    writeln!(writer, "LOOKUP_TABLE default").with_context(|| "failed to write VTK LOOKUP_TABLE")?;

    // --- Write binary scalar data (big-endian f32) ---
    let mut binary_buf = Vec::with_capacity(total_voxels * 4);
    for &value in slice {
        binary_buf.extend_from_slice(&value.to_be_bytes());
    }
    writer
        .write_all(&binary_buf)
        .with_context(|| "failed to write VTK binary scalar data")?;

    writer
        .flush()
        .with_context(|| "failed to flush VTK output")?;

    tracing::debug!(
        "VTK data written: {} voxels, {} bytes payload",
        total_voxels,
        total_voxels * 4
    );

    Ok(())
}

/// Write a native Coeus image to a VTK legacy structured-points file (BINARY).
///
/// The output file conforms to VTK legacy format version 3.0 with:
/// - `DATASET STRUCTURED_POINTS`
/// - `BINARY` encoding
/// - `SCALARS scalars float 1` point data
/// - Big-endian IEEE 754 single-precision scalar values
///
/// Extracts flat data and geometry from the native tensor carrier, then
/// delegates the byte-level encode to [`encode_vtk_flat`].
///
/// # Errors
///
/// Returns an error when:
/// - The image has a non-Cartesian coordinate map or a direction outside the
///   VTK-aligned ZYX-to-XYZ axis order. These checks occur before the destination
///   is created or truncated.
/// - A spacing component is non-finite or not strictly positive. This check
///   occurs before the destination is created or truncated.
/// - The file cannot be created or written.
/// - The tensor data cannot be extracted as `f32`.
pub fn write_vtk<B, P>(path: P, image: &Image<f32, B, 3>, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    anyhow::ensure!(
        image.coordinate_map().is_cartesian(),
        "legacy VTK structured points cannot preserve a non-Cartesian coordinate map"
    );
    anyhow::ensure!(
        image.direction() == &crate::domain::axis_order::vtk_image_direction(),
        "legacy VTK structured points cannot preserve a direction matrix outside the VTK-aligned ZYX-to-XYZ axis order"
    );

    let dims = image.shape(); // [nz, ny, nx]
    let [nz, ny, nx] = dims;
    let dims_xyz = [nx, ny, nz];
    let expected_voxels = crate::io::structured_points::voxel_count(dims_xyz)?;

    let image_spacing = image.spacing();
    let spacing_arr = crate::domain::axis_order::reverse_axes([
        image_spacing[0],
        image_spacing[1],
        image_spacing[2],
    ]);
    crate::io::structured_points::validate_spacing(spacing_arr)?;

    let origin = image.origin(); // [X, Y, Z] order
    let origin_arr = [origin[0], origin[1], origin[2]];
    let f32_vec = image.data_cow_on(backend);
    anyhow::ensure!(
        f32_vec.len() == expected_voxels,
        "image data length does not match VTK DIMENSIONS"
    );

    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create VTK file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    encode_vtk_flat(&mut writer, &f32_vec, dims, origin_arr, spacing_arr)?;

    tracing::debug!(path = %path.display(), "VTK file written");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::encode_vtk_flat;

    #[test]
    fn flat_encoder_rejects_nonfinite_or_nonpositive_spacing_before_writing() {
        for spacing in [
            [0.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [f64::NAN, 1.0, 1.0],
            [f64::INFINITY, 1.0, 1.0],
        ] {
            let mut output = Vec::new();
            let error = encode_vtk_flat(&mut output, &[1.0], [1, 1, 1], [0.0; 3], spacing)
                .expect_err("invalid spacing must be rejected");

            assert!(
                error
                    .to_string()
                    .contains("legacy VTK structured points requires finite, positive SPACING"),
                "unexpected validation error: {error}"
            );
            assert!(output.is_empty(), "invalid geometry must write no bytes");
        }
    }
}
