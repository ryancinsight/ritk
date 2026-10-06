use super::decimal_string::{format_dicom_decimal, format_pair, format_six, format_triplet};
use super::error::DicomWriteError;
use super::output::{serialize_file, write_series_files};
use super::pixel_encoding::{
    emit_pixel_format_tags, generate_instance_uid, generate_series_uid, prepare_frame_encodings,
    validate_image_shape, validate_spatial_metadata, DICOM_SOP_CLASS_SECONDARY_CAPTURE,
    MONOCHROME2,
};
use crate::format::dicom::writer::elements::PutValue;
use anyhow::{Context, Result};
use coeus_core::MoiraiBackend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use eunomia::FloatElement;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;

/// Spatial geometry inputs for the substrate-free series encode core.
///
/// Field conventions mirror [`crate::format::dicom::series`]' `decode_series`
/// exactly so a written series round-trips through the native/Coeus series
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

/// Write a 3-D `Image<f32, B, 3>` with shape `[depth, rows, cols]` as a series of
/// per-slice single-frame DICOM Part 10 files.
///
/// Delegates to the substrate-free `write_series_flat` encode core; the Coeus
/// carrier only supplies the host pixel buffer and spatial geometry. Retained
/// for consumers not yet migrated off the Coeus `Image`; new native code uses
/// [`write_dicom_series_native`].
/// Input and serialization preflight cover every slice before output changes.
/// Preflight retains the serialized volume in memory; filesystem failures
/// during persistence can leave a partial series.
pub fn write_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
) -> Result<()> {
    let all_data = image.data_cow_on(&B::default()).into_owned();
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
///   `[0, 65535]`.
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
/// Substrate-free encode core shared by the Coeus and native series writers. The
/// per-slice pixel rescale, tag emission, geometry derivation, and file layout
/// are defined here so the two carriers produce pixel-and-geometry-identical
/// output for identical voxels (SSOT).
fn write_series_flat(
    path: &Path,
    all_data: &[f32],
    shape: [usize; 3],
    geom: &SeriesGeometry,
) -> Result<()> {
    let dimensions = validate_image_shape(shape, all_data.len())?;
    validate_series_geometry(geom)?;
    let pixel_plans = prepare_frame_encodings::<u16>(all_data, dimensions)?;
    validate_series_positions(geom, dimensions.depth)?;
    let mut slices = Vec::new();
    slices
        .try_reserve_exact(dimensions.depth)
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
    let series_uid = generate_series_uid();
    let study_uid = series_uid.clone();
    let series_instance_uid = format!("{}.1", series_uid);

    let [dir_x, dir_y, dir_z] = geom.direction_columns;
    // Reader convention: PixelSpacing = [row spacing, col spacing] = [Δy, Δx].
    let pixel_spacing = [geom.spacing[1], geom.spacing[0]];
    let slice_spacing = geom.spacing[2];
    let orientation = [dir_x[0], dir_x[1], dir_x[2], dir_y[0], dir_y[1], dir_y[2]];

    for z in 0..dimensions.depth {
        let slice_offset = z
            .checked_mul(dimensions.frame_samples)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let slice_end = slice_offset
            .checked_add(dimensions.frame_samples)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let slice_f32 =
            all_data
                .get(slice_offset..slice_end)
                .ok_or(DicomWriteError::PixelCountMismatch {
                    expected: dimensions.total_samples,
                    actual: all_data.len(),
                })?;
        let plan = pixel_plans
            .get(z)
            .copied()
            .ok_or(DicomWriteError::PixelCountMismatch {
                expected: dimensions.depth,
                actual: pixel_plans.len(),
            })?;
        let pixel_u16 = plan.encode(slice_f32, slice_offset)?;
        let rescale_slope = plan.rescale_slope();
        let rescale_intercept = plan.rescale_intercept();
        let sop_instance_uid = generate_instance_uid(&series_uid, z);
        let zf = f64::from_count(z);
        let image_position = [
            geom.origin[0] + zf * slice_spacing * dir_z[0],
            geom.origin[1] + zf * slice_spacing * dir_z[1],
            geom.origin[2] + zf * slice_spacing * dir_z[2],
        ];
        let mut obj = InMemDicomObject::new_empty();
        obj.put_value(
            Tag(0x0008, 0x0016),
            VR::UI,
            DICOM_SOP_CLASS_SECONDARY_CAPTURE,
        );
        obj.put_value(Tag(0x0008, 0x0018), VR::UI, sop_instance_uid.as_str());
        obj.put_value(Tag(0x0008, 0x0060), VR::CS, "OT");
        obj.put_value(Tag(0x0008, 0x0064), VR::CS, "WSD");
        obj.put_value(Tag(0x0020, 0x000D), VR::UI, study_uid.as_str());
        obj.put_value(Tag(0x0020, 0x000E), VR::UI, series_instance_uid.as_str());
        obj.put_value(Tag(0x0020, 0x0013), VR::IS, format!("{}", z + 1));
        // PS3.3 C.7.1.1 Patient Module (Type 2 => present with empty value when unknown).
        obj.put_value(Tag(0x0008, 0x0090), VR::PN, "");
        obj.put_value(Tag(0x0010, 0x0010), VR::PN, "");
        obj.put_value(Tag(0x0010, 0x0020), VR::LO, "");
        // PS3.3 C.7.2.1 General Study Module (Type 2).
        obj.put_value(Tag(0x0008, 0x0020), VR::DA, "");
        // PS3.3 C.7.3.1 General Series Module: SeriesNumber (Type 2).
        obj.put_value(Tag(0x0020, 0x0011), VR::IS, "0");
        // PS3.3 C.7.6.2 Image Plane Module: spatial geometry (round-trips
        // through the series reader; see `write_dicom_series_native` docs).
        obj.put_value(
            Tag(0x0018, 0x0050),
            VR::DS,
            format_dicom_decimal(slice_spacing)?,
        );
        obj.put_value(Tag(0x0020, 0x0032), VR::DS, format_triplet(image_position)?);
        obj.put_value(Tag(0x0020, 0x0037), VR::DS, format_six(orientation)?);
        obj.put_value(Tag(0x0028, 0x0030), VR::DS, format_pair(pixel_spacing)?);
        obj.put_value(Tag(0x0028, 0x0002), VR::US, 1_u16);
        obj.put_value(Tag(0x0028, 0x0010), VR::US, dimensions.rows_attribute);
        obj.put_value(Tag(0x0028, 0x0011), VR::US, dimensions.columns_attribute);
        emit_pixel_format_tags::<u16>(&mut obj);
        obj.put_value(
            Tag(0x0028, 0x1053),
            VR::DS,
            format_dicom_decimal(f64::from(rescale_slope))?,
        );
        obj.put_value(
            Tag(0x0028, 0x1052),
            VR::DS,
            format_dicom_decimal(f64::from(rescale_intercept))?,
        );
        obj.put_value(Tag(0x0028, 0x0004), VR::CS, MONOCHROME2);
        obj.put_value(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        );
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(DICOM_SOP_CLASS_SECONDARY_CAPTURE)
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax(EXPLICIT_VR_LE),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        slices.push(serialize_file(&file_obj)?);
    }
    write_series_files(path, &slices)
}

fn validate_series_geometry(geom: &SeriesGeometry) -> Result<()> {
    let direction = [
        geom.direction_columns[0][0],
        geom.direction_columns[0][1],
        geom.direction_columns[0][2],
        geom.direction_columns[1][0],
        geom.direction_columns[1][1],
        geom.direction_columns[1][2],
        geom.direction_columns[2][0],
        geom.direction_columns[2][1],
        geom.direction_columns[2][2],
    ];
    validate_spatial_metadata(&geom.spacing, &geom.origin, &direction)
}

fn validate_series_positions(geom: &SeriesGeometry, depth: usize) -> Result<()> {
    let dir_z = geom.direction_columns[2];
    let slice_spacing = geom.spacing[2];
    for z in 0..depth {
        let zf = f64::from_count(z);
        let position = [
            geom.origin[0] + zf * slice_spacing * dir_z[0],
            geom.origin[1] + zf * slice_spacing * dir_z[1],
            geom.origin[2] + zf * slice_spacing * dir_z[2],
        ];
        if position.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(DicomWriteError::InvalidSpatialMetadata.into());
        }
    }
    Ok(())
}
