//! DICOM RGB multiframe loading.
//!
//! This module preserves true-color multiframe pixel data in `RgbVolume<f32, B>`
//! instead of passing multi-sample frames through scalar `Image<B, 3>`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use coeus_core::MoiraiBackend;
use dicom::core::Tag;
use ritk_dicom::{
    decode_frame_with, parse_file_with, DecodeFrameRequest, DicomRsBackend, PixelLayout,
    PixelSignedness, TransferSyntaxKind,
};
use ritk_image::native::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

/// Substrate-agnostic decoded RGB multi-frame volume: a flat interleaved-RGB
/// `f32` buffer in `[frames, rows, cols, 3]` order (channel fastest) plus the
/// resolved spatial metadata (the three physical axes only).
///
/// Shared, tensor-free core of the colour multi-frame read path. The native
/// carrier constructs from the same instance produced by
/// [`load_color_multiframe_flat`].
pub struct ColorMultiFrameVolume {
    /// Interleaved RGB voxels in `[frames, rows, cols, 3]` order.
    pub data: Vec<f32>,
    /// Volume shape `[frames, rows, cols, 3]`.
    pub shape: [usize; 4],
    /// Physical coordinate of the first voxel (three spatial axes).
    pub origin: Point<3>,
    /// Inter-voxel spacing `[z, row, col]` in mm.
    pub spacing: Spacing<3>,
    /// Direction cosine matrix derived from ImageOrientationPatient.
    pub direction: Direction<3>,
}

use super::color_common::{read_optional, required_string, RGB_CHANNELS};
use super::multiframe::{read_multiframe_info, MultiFrameInfo};
use super::reader::geometry::{SliceCoverage, SpacingUniformity};

/// Read an interleaved RGB DICOM multiframe object into a native rank-4 colour
/// volume.
///
/// Native counterpart of `read_dicom_color_multiframe`: identical decode
/// (both route through [`load_color_multiframe_flat`]). Returns
/// `Image<f32, MoiraiBackend, 4>` with shape `[frames, rows, cols, 3]`,
/// interleaved RGB samples in the channel axis. The three physical axes carry
/// the frame origin/spacing/direction; the channel axis is a non-spatial
/// identity axis (origin offset 0, spacing 1, identity direction).
pub fn load_atlas_color_multiframe<P: AsRef<Path>>(
    path: P,
) -> Result<NativeImage<f32, MoiraiBackend, 4>> {
    let ColorMultiFrameVolume {
        data,
        shape,
        origin,
        spacing,
        direction: _,
    } = load_color_multiframe_flat(path)?;
    let origin4 = Point::<4>::new([origin[0], origin[1], origin[2], 0.0]);
    let spacing4 = Spacing::<4>::new([spacing[0], spacing[1], spacing[2], 1.0]);
    NativeImage::<f32, MoiraiBackend, 4>::from_flat(
        data,
        shape,
        origin4,
        spacing4,
        Direction::<4>::identity(),
    )
}

/// Decode an interleaved RGB DICOM multiframe object into a substrate-free
/// [`ColorMultiFrameVolume`].
///
/// Shared, tensor-free core: RGB validation, per-frame pixel decode, and
/// spatial-metadata resolution, without constructing any image carrier. Accepts
/// only unsigned 8-bit RGB with `PlanarConfiguration=0` (interleaved channels).
pub fn load_color_multiframe_flat<P: AsRef<Path>>(path: P) -> Result<ColorMultiFrameVolume> {
    let path = path.as_ref();
    let info = read_multiframe_info(path)?;
    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("failed to open DICOM RGB multiframe {:?}", path))?;
    let ts_uid = obj.meta().transfer_syntax();
    let transfer_syntax = TransferSyntaxKind::from_uid(ts_uid);
    validate_rgb_multiframe(path, &obj, &info, ts_uid, &transfer_syntax)?;

    let frame_samples = info
        .rows
        .checked_mul(info.cols)
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .context("DICOM RGB multiframe frame sample count overflow")?;
    let total_samples = frame_samples
        .checked_mul(info.n_frames)
        .context("DICOM RGB multiframe volume sample count overflow")?;
    // `rows`/`cols`/`n_frames` are header-derived, so a hostile or corrupt file
    // could otherwise force an up-front multi-gigabyte zero-fill before any
    // frame is decoded. Cap the speculative reservation and grow the buffer by
    // appending each validated, sequentially-decoded frame instead of
    // pre-sizing and indexing into it.
    let mut volume = Vec::with_capacity(consus_io::bounded_capacity(
        total_samples,
        std::mem::size_of::<f32>(),
    ));

    for frame_index in 0..info.n_frames {
        let frame = decode_frame_with::<DicomRsBackend>(
            &obj,
            DecodeFrameRequest {
                frame_index: u32::try_from(frame_index)
                    .context("DICOM RGB multiframe frame index exceeds u32")?,
                transfer_syntax: transfer_syntax.clone(),
                layout: PixelLayout {
                    rows: info.rows,
                    cols: info.cols,
                    samples_per_pixel: RGB_CHANNELS,
                    bits_allocated: info.bits_allocated,
                    pixel_representation: info.pixel_representation,
                    rescale_slope: 1.0,
                    rescale_intercept: 0.0,
                },
            },
        )
        .with_context(|| {
            format!(
                "DICOM backend failed for RGB frame {frame_index} in {:?}",
                path
            )
        })?
        .pixels;
        if frame.len() != frame_samples {
            bail!(
                "DICOM RGB multiframe {:?} frame {} decoded {} samples; expected {}",
                path,
                frame_index,
                frame.len(),
                frame_samples
            );
        }
        volume.extend_from_slice(&frame);
    }

    Ok(ColorMultiFrameVolume {
        data: volume,
        shape: [info.n_frames, info.rows, info.cols, RGB_CHANNELS],
        origin: Point::new(origin_from_info(&info)),
        spacing: spacing_from_info(&info)?,
        direction: direction_from_info(&info),
    })
}

fn validate_rgb_multiframe(
    path: &Path,
    obj: &dicom::object::DefaultDicomObject,
    info: &MultiFrameInfo,
    transfer_syntax_uid: &str,
    transfer_syntax: &TransferSyntaxKind,
) -> Result<()> {
    if transfer_syntax.is_compressed() && !transfer_syntax.is_codec_supported() {
        bail!(
            "DICOM RGB multiframe: compressed transfer syntax '{}' in {:?} is not supported",
            transfer_syntax_uid,
            path
        );
    }
    if transfer_syntax.is_big_endian() {
        bail!(
            "DICOM RGB multiframe: big-endian transfer syntax '{}' in {:?} is not supported",
            transfer_syntax_uid,
            path
        );
    }
    if info.n_frames == 0 || info.rows == 0 || info.cols == 0 {
        bail!(
            "DICOM RGB multiframe {:?} has invalid dimensions frames={} rows={} cols={}",
            path,
            info.n_frames,
            info.rows,
            info.cols
        );
    }
    if info.samples_per_pixel != RGB_CHANNELS {
        bail!(
            "DICOM color multiframe loader supports only RGB SamplesPerPixel=3; {:?} declares SamplesPerPixel={}",
            path,
            info.samples_per_pixel
        );
    }

    let photometric = required_string(obj, Tag(0x0028, 0x0004), "PhotometricInterpretation")?;
    if !photometric.trim().eq_ignore_ascii_case("RGB") {
        bail!(
            "DICOM color multiframe loader supports only PhotometricInterpretation=RGB; {:?} declares {}",
            path,
            photometric.trim()
        );
    }
    let planar_configuration = read_optional::<u16>(obj, Tag(0x0028, 0x0006)).unwrap_or(0);
    if planar_configuration != 0 {
        bail!(
            "DICOM RGB color multiframe loader supports only interleaved PlanarConfiguration=0; {:?} declares {}",
            path,
            planar_configuration
        );
    }
    if info.bits_allocated != 8 {
        bail!(
            "DICOM RGB color multiframe loader supports only BitsAllocated=8; {:?} declares {}",
            path,
            info.bits_allocated
        );
    }
    if info.pixel_representation != PixelSignedness::Unsigned {
        bail!(
            "DICOM RGB color multiframe loader supports only unsigned samples; {:?} declares PixelRepresentation={}",
            path,
            u16::from(info.pixel_representation)
        );
    }
    Ok(())
}

fn origin_from_info(info: &MultiFrameInfo) -> [f64; 3] {
    info.per_frame
        .first()
        .and_then(|frame| frame.image_position)
        .or(info.image_position)
        .unwrap_or([0.0, 0.0, 0.0])
}

fn spacing_from_info(info: &MultiFrameInfo) -> Result<Spacing<3>> {
    let spacing_z = spacing_z_from_info(info)?;
    let [row_spacing, col_spacing] = info
        .per_frame
        .first()
        .and_then(|frame| frame.pixel_spacing)
        .or(info.pixel_spacing)
        .unwrap_or([1.0, 1.0]);
    Ok(Spacing::new([spacing_z, row_spacing, col_spacing]))
}

fn spacing_z_from_info(info: &MultiFrameInfo) -> Result<f64> {
    if !info.per_frame.is_empty() {
        if info.per_frame.len() != info.n_frames {
            bail!(
                "DICOM RGB multiframe per-frame metadata count {} does not match NumberOfFrames {}",
                info.per_frame.len(),
                info.n_frames
            );
        }
        if info.per_frame.len() >= 2 {
            let iop = info
                .per_frame
                .iter()
                .find_map(|frame| frame.image_orientation)
                .or(info.image_orientation);
            if let Some(normal) = iop.and_then(super::reader::slice_normal_from_iop) {
                let positions = info
                    .per_frame
                    .iter()
                    .map(|frame| frame.image_position.map(|p| super::reader::dot(p, normal)))
                    .collect::<Option<Vec<f64>>>();
                if let Some(positions) = positions {
                    let report = super::reader::analyze_slice_spacing(&positions);
                    if report.spacing_uniformity == SpacingUniformity::Nonuniform
                        || report.slice_coverage == SliceCoverage::HasMissingSlices
                    {
                        bail!(
                            "DICOM RGB multiframe color loader requires uniform frame spacing; nominal={} max_relative_deviation={} missing_gaps={}",
                            report.nominal_spacing,
                            report.max_relative_deviation,
                            report.missing_between.len()
                        );
                    }
                    return Ok(report.nominal_spacing);
                }
            }
        }
        if let Some(thickness) = info
            .per_frame
            .first()
            .and_then(|frame| frame.slice_thickness)
        {
            return Ok(thickness);
        }
    }
    Ok(info.frame_thickness.unwrap_or(1.0))
}

fn direction_from_info(info: &MultiFrameInfo) -> Direction<3> {
    let Some(iop) = info
        .per_frame
        .iter()
        .find_map(|frame| frame.image_orientation)
        .or(info.image_orientation)
    else {
        return Direction::identity();
    };
    let (rx, ry, rz) = (iop[0], iop[1], iop[2]);
    let (cx, cy, cz) = (iop[3], iop[4], iop[5]);
    let normal =
        super::reader::normalize([ry * cz - rz * cy, rz * cx - rx * cz, rx * cy - ry * cx])
            .unwrap_or([0.0, 0.0, 1.0]);
    Direction::from_column_major([normal[0], normal[1], normal[2], cx, cy, cz, rx, ry, rz])
}

#[cfg(test)]
#[path = "tests_color_multiframe.rs"]
mod tests;
