use super::utils::{
    emit_pixel_format_tags, ensure_series_directory, format_pair, format_six, format_triplet,
    generate_instance_uid, generate_series_uid, normalize_to_u16,
    DICOM_SOP_CLASS_SECONDARY_CAPTURE, MONOCHROME2,
};
use anyhow::{bail, Context, Result};
use coeus_core::MoiraiBackend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;

/// Spatial geometry inputs for the substrate-free series encode core.
///
/// Field conventions mirror [`crate::format::dicom::series`]' `decode_series`
/// exactly so a written series round-trips through the native/Burn series
/// readers to the same voxels and geometry:
/// - `spacing` is image-axis spacing `[Δx(col), Δy(row), Δz(slice)]`, matching
///   the reader's `Spacing::new([dx, dy, dz])`.
/// - `direction_columns` are the direction-cosine columns `[dir_x, dir_y,
///   dir_z]` (image row-axis, column-axis, slice-axis), matching the reader's
///   `Direction::from_columns([dir_x, dir_y, dir_z])`.
struct SeriesGeometry {
    origin: [f64; 3],
    spacing: [f64; 3],
    direction_columns: [[f64; 3]; 3],
}

fn series_geometry(
    origin: &Point<3>,
    spacing: &Spacing<3>,
    direction: &Direction<3>,
) -> SeriesGeometry {
    let columns = direction.axis_directions_array();
    SeriesGeometry {
        origin: origin.to_array(),
        spacing: spacing.to_array(),
        direction_columns: [
            columns[0].to_array(),
            columns[1].to_array(),
            columns[2].to_array(),
        ],
    }
}

/// Write a 3-D `Image<B, 3>` with shape `[depth, rows, cols]` as a series of
/// per-slice single-frame DICOM Part 10 files.
///
/// Delegates to the substrate-free `write_series_flat` encode core; the Burn
/// carrier only supplies the host pixel buffer and spatial geometry. Retained
/// for consumers not yet migrated off the Burn `Image`; new native code uses
/// [`write_dicom_series_native`].
pub fn write_dicom_series<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let all_data = image
        .try_data_vec()
        .context("DICOM series writer requires f32 image data")?;
    let geom = series_geometry(image.origin(), image.spacing(), image.direction());
    write_series_flat(path.as_ref(), &all_data, image.shape(), &geom)
}

/// Write a native `Image<f32, MoiraiBackend, 3>` with shape `[depth, rows,
/// cols]` as a series of per-slice single-frame DICOM Part 10 files.
///
/// Native counterpart of [`write_dicom_series`]: both route through the shared
/// substrate-free `write_series_flat` encode core, so they emit
/// pixel-and-geometry-identical output for identical voxels, differing only in
/// the per-call random UIDs.
///
/// ## Geometry conventions (round-trip contract with the series reader)
/// - Slice ordering: slice `z` is written to `slice_{z:04}.dcm` with
///   InstanceNumber (0020,0013) = `z + 1`.
/// - ImagePositionPatient (0020,0032) of slice `z` = `origin + z · Δz · dir_z`,
///   where `Δz = spacing[2]` and `dir_z` is the slice-axis direction column.
///   Because `Δz > 0` and `dir_z` is a unit vector, the per-slice position
///   projected onto `dir_z` increases monotonically with `z`, so the reader's
///   projection sort recovers the original slice order.
/// - ImageOrientationPatient (0020,0037) = `[dir_x, dir_y]` (row-axis then
///   column-axis direction cosines).
/// - PixelSpacing (0028,0030) = `[Δy(row), Δx(col)]` = `[spacing[1],
///   spacing[0]]`.
/// - SliceThickness (0018,0050) = `Δz = spacing[2]` (also the single-slice
///   spacing fallback the reader uses when `depth == 1`).
/// - Pixel representation: unsigned 16-bit MONOCHROME2; a single per-slice
///   linear rescale (slope/intercept) maps the slice's f32 range onto
///   `[0, 65535]` (see `normalize_to_u16`).
pub fn write_dicom_series_native<P: AsRef<Path>>(
    path: P,
    image: &NativeImage<f32, MoiraiBackend, 3>,
) -> Result<()> {
    let all_data = image
        .data_slice()
        .context("DICOM series writer requires contiguous f32 image data")?;
    let geom = series_geometry(image.origin(), image.spacing(), image.direction());
    write_series_flat(path.as_ref(), all_data, image.shape(), &geom)
}

/// Serialize a flat `[depth, rows, cols]` row-major `f32` buffer as a series of
/// per-slice single-frame DICOM Part 10 files.
///
/// Substrate-free encode core shared by the Burn and native series writers. The
/// per-slice pixel rescale, tag emission, geometry derivation, and file layout
/// are defined here so the two carriers produce pixel-and-geometry-identical
/// output for identical voxels (SSOT).
fn write_series_flat(
    path: &Path,
    all_data: &[f32],
    shape: [usize; 3],
    geom: &SeriesGeometry,
) -> Result<()> {
    let [depth, rows, cols] = shape;
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM: depth={depth} rows={rows} cols={cols} must be >0");
    }
    let series_dir = ensure_series_directory(path)?;
    let series_uid = generate_series_uid();
    let study_uid = series_uid.clone();
    let series_instance_uid = format!("{}.1", series_uid);

    let [dir_x, dir_y, dir_z] = geom.direction_columns;
    // Reader convention: PixelSpacing = [row spacing, col spacing] = [Δy, Δx].
    let pixel_spacing = [geom.spacing[1], geom.spacing[0]];
    let slice_spacing = geom.spacing[2];
    let orientation = [dir_x[0], dir_x[1], dir_x[2], dir_y[0], dir_y[1], dir_y[2]];

    let slice_len = rows * cols;
    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (pixel_u16, rescale_slope, rescale_intercept) = normalize_to_u16(slice_f32);
        let sop_instance_uid = generate_instance_uid(&series_uid, z);
        let zf = z as f64;
        let image_position = [
            geom.origin[0] + zf * slice_spacing * dir_z[0],
            geom.origin[1] + zf * slice_spacing * dir_z[1],
            geom.origin[2] + zf * slice_spacing * dir_z[2],
        ];
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(DICOM_SOP_CLASS_SECONDARY_CAPTURE),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("OT"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0064),
            VR::CS,
            PrimitiveValue::from("WSD"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from(study_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", z + 1)),
        ));
        // PS3.3 C.7.1.1 Patient Module (Type 2 => present with empty value when unknown).
        obj.put(DataElement::new(
            Tag(0x0008, 0x0090),
            VR::PN,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0010, 0x0010),
            VR::PN,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0010, 0x0020),
            VR::LO,
            PrimitiveValue::from(""),
        ));
        // PS3.3 C.7.2.1 General Study Module (Type 2).
        obj.put(DataElement::new(
            Tag(0x0008, 0x0020),
            VR::DA,
            PrimitiveValue::from(""),
        ));
        // PS3.3 C.7.3.1 General Series Module: SeriesNumber (Type 2).
        obj.put(DataElement::new(
            Tag(0x0020, 0x0011),
            VR::IS,
            PrimitiveValue::from("0"),
        ));
        // PS3.3 C.7.6.2 Image Plane Module: spatial geometry (round-trips
        // through the series reader; see `write_dicom_series_native` docs).
        obj.put(DataElement::new(
            Tag(0x0018, 0x0050),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", slice_spacing)),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(format_triplet(image_position)),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from(format_six(orientation)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from(format_pair(pixel_spacing)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols as u16),
        ));
        emit_pixel_format_tags(&mut obj);
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_slope)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_intercept)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from(MONOCHROME2),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(DICOM_SOP_CLASS_SECONDARY_CAPTURE)
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax(EXPLICIT_VR_LE),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj
            .write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}
