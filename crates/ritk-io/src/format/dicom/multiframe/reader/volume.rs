//! Multi-frame DICOM volume decode.
//!
//! The substrate-free [`MultiFrameVolume`] and the three carriers built from
//! it. Header extraction lives in [`super::header`]; this module starts where
//! a decoded pixel is required.

use anyhow::{bail, Context, Result};
use coeus_core::MoiraiBackend;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use ritk_core::image::Image;
use ritk_dicom::{
    decode_frame_with, parse_file_with_budget, DecodeFrameRequest, DicomRsBackend, PixelLayout,
    TransferSyntaxKind,
};
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;

use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

use super::super::enhanced_pet::{BodyWeightSuvVolume, FrameMapping};
use super::super::per_frame::extract_functional_groups;
use super::super::temporal::reject_temporal_organization;
use super::header::extract_multiframe_header;
use crate::format::dicom::reader::geometry::{SliceCoverage, SpacingUniformity};
use crate::format::dicom::reader::DicomReadBudget;
/// Substrate-agnostic decoded multi-frame volume: a flat row-major `f32` buffer
/// in `[frames, rows, cols]` order plus the resolved spatial metadata.
///
/// This is the shared, tensor-free core of the multi-frame read path. Both the
/// Coeus carrier ([`load_dicom_multiframe`]) and the native Coeus carrier
/// ([`load_dicom_multiframe_native`]) construct their image from the same
/// instance, so the pixel decode and geometry analysis live in exactly one
/// place ([`load_dicom_multiframe_flat`]).
pub struct MultiFrameVolume {
    /// Row-major voxels in `[frames, rows, cols]` order, RescaleSlope/Intercept
    /// already applied.
    pub data: Vec<f32>,
    /// Enhanced PET body-weight SUV samples when quantitative mappings are present.
    pub suv_bw: Option<BodyWeightSuvVolume>,
    /// Volume shape `[frames, rows, cols]` after any gap/nonuniform resampling.
    pub shape: [usize; 3],
    /// Physical coordinate of the first voxel.
    pub origin: Point<3>,
    /// Inter-voxel spacing `[z, row, col]` in mm.
    pub spacing: Spacing<3>,
    /// Direction cosine matrix derived from ImageOrientationPatient.
    pub direction: Direction<3>,
}
/// Load a multi-frame DICOM file as a 3-D image with shape [n_frames, rows, cols].
///
/// Applies RescaleSlope and RescaleIntercept to convert stored integers to floats.
/// For Enhanced CT/MR/PET objects that carry per-frame functional groups (5200,9230),
/// per-frame slope/intercept values override the global tags for each frame in the
/// native (uncompressed) decode path.
/// When ImagePositionPatient (0020,0032) is present, the image origin is set
/// accordingly; otherwise the origin defaults to [0, 0, 0].
/// When ImageOrientationPatient (0020,0037) is present, the direction matrix is
/// derived from the row and column cosines with the normal computed as their
/// cross product; otherwise the direction defaults to identity.
pub fn load_dicom_multiframe<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B,
) -> Result<Image<f32, B, 3>> {
    let MultiFrameVolume {
        data,
        shape,
        origin,
        spacing,
        direction,
        ..
    } = load_dicom_multiframe_flat(path)?;
    let tensor = Tensor::<f32, B>::from_slice_on(shape, &data, device)?;
    Image::new(tensor, origin, spacing, direction)
}

/// Load a multi-frame DICOM file into a native Coeus-backed image.
///
/// Native counterpart of [`load_dicom_multiframe`]: identical decode and
/// geometry (both route through [`load_dicom_multiframe_flat`]), differing only
/// in the image carrier. Returns `Image<f32, MoiraiBackend, 3>` with shape
/// `[frames, rows, cols]`.
pub fn load_dicom_multiframe_native<P: AsRef<Path>>(
    path: P,
) -> Result<NativeImage<f32, MoiraiBackend, 3>> {
    let MultiFrameVolume {
        data,
        shape,
        origin,
        spacing,
        direction,
        ..
    } = load_dicom_multiframe_flat(path)?;
    NativeImage::<f32, MoiraiBackend, 3>::from_flat(data, shape, origin, spacing, direction)
}

/// Decode a multi-frame DICOM file into a substrate-free [`MultiFrameVolume`].
///
/// This is the shared, tensor-free core: pixel decode, RescaleSlope/Intercept
/// application (per-frame overrides where Enhanced functional groups are
/// present), and per-frame geometry analysis (nonuniform / missing-frame
/// resampling), without constructing any image carrier. See
/// [`load_dicom_multiframe`] for the full decode contract.
pub fn load_dicom_multiframe_flat<P: AsRef<Path>>(path: P) -> Result<MultiFrameVolume> {
    let path = path.as_ref();
    let obj = parse_file_with_budget::<DicomRsBackend, _>(path, &DicomReadBudget::DEFAULT.parser())
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;

    load_multiframe_flat_from_object(path, &obj, &DicomReadBudget::DEFAULT)
}

pub(crate) fn load_multiframe_flat_from_object(
    path: &Path,
    obj: &DefaultDicomObject,
    budget: &DicomReadBudget,
) -> Result<MultiFrameVolume> {
    if super::super::pet_dimensions::is_enhanced_pet(&obj.meta().media_storage_sop_class_uid) {
        let (geometry, mut groups) = super::pet::pet_spatial_groups(path, obj, budget)?;
        if groups.len() != 1 {
            bail!("Enhanced PET has multiple temporal or gated volumes; use load_enhanced_pet_volumes");
        }
        let group = groups.pop().expect("one PET volume");
        return decode_spatial_frames(path, obj, budget, Some(&group.frames), Some(&geometry));
    }
    reject_temporal_organization(path, obj)?;
    decode_spatial_frames(path, obj, budget, None, None)
}

pub(super) fn decode_spatial_frames(
    path: &Path,
    obj: &DefaultDicomObject,
    budget: &DicomReadBudget,
    selected: Option<&[usize]>,
    geometry: Option<&[super::super::types::PerFrameInfo]>,
) -> Result<MultiFrameVolume> {
    // Guard: compressed transfer syntaxes are not natively decodable by ritk-io.
    // Pixel data from compressed objects cannot be interpreted as raw u16/u8 samples.
    let ts_uid = obj.meta().transfer_syntax();
    let ts = TransferSyntaxKind::from_uid(ts_uid);
    if ts.is_compressed() && !ts.is_codec_supported() {
        bail!(
            "DICOM multiframe: compressed transfer syntax '{}' in {:?} is not supported \
             (not natively decoded and no codec registered); \
             decompress the file or use a supported transfer syntax",
            ts_uid,
            path
        );
    }
    if ts.is_big_endian() {
        bail!(
            "DICOM multiframe: big-endian transfer syntax '{}' in {:?} is not supported; \
             pixel decode requires little-endian byte order",
            ts_uid,
            path
        );
    }

    let mut info = extract_multiframe_header(path, obj)?;
    if info.rows == 0 || info.cols == 0 {
        bail!(
            "DICOM multiframe: rows={} cols={} must be >0 in {:?}",
            info.rows,
            info.cols,
            path
        );
    }
    if info.samples_per_pixel != 1 {
        bail!(
            "DICOM multiframe scalar volume loader supports only SamplesPerPixel=1; \
             {:?} declares SamplesPerPixel={}. \
             Decode RGB/color frames through the codec boundary or a color-volume loader",
            path,
            info.samples_per_pixel
        );
    }

    if info.n_frames == 0 {
        bail!(
            "DICOM multiframe: NumberOfFrames must be greater than zero in {:?}",
            path
        );
    }

    let owned_geometry;
    let per_frame = if let Some(geometry) = geometry {
        geometry
    } else {
        owned_geometry = extract_functional_groups(obj, info.n_frames);
        &owned_geometry
    };
    let actual_n = selected.map_or(info.n_frames, <[usize]>::len);
    let source_index = |position| selected.map_or(position, |frames| frames[position]);
    if let Some(first) = selected
        .and_then(|frames| frames.first())
        .and_then(|&index| per_frame.get(index))
    {
        info.image_position = first.image_position.or(info.image_position);
        info.image_orientation = first.image_orientation.or(info.image_orientation);
        info.pixel_spacing = first.pixel_spacing.or(info.pixel_spacing);
        info.frame_thickness = first.slice_thickness.or(info.frame_thickness);
    }

    tracing::debug!(
        n_frames = info.n_frames,
        per_frame_len = per_frame.len(),
        "load_dicom_multiframe: functional groups extracted"
    );

    let frame_pixels = info
        .rows
        .checked_mul(info.cols)
        .context("DICOM multiframe frame pixel count overflows usize")?;
    let decoded_elements = actual_n
        .checked_mul(frame_pixels)
        .context("DICOM multiframe decoded element count overflows usize")?;
    let decoded_bytes = decoded_elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("DICOM multiframe decoded byte count overflows usize")?;
    budget
        .checked_decoded_bytes(decoded_bytes)
        .context("DICOM multiframe decoded volume exceeds budget")?;
    // Cap the speculative reservation: `n_frames` and `frame_pixels` are
    // header-derived, so a hostile file could otherwise abort on a huge
    // `Vec::with_capacity`. The buffer still grows to its true size as frames
    // decode, and each frame is bounds-validated below.
    let mut floats = Vec::with_capacity(ritk_core::io_bounds::bounded_capacity(
        actual_n.saturating_mul(frame_pixels),
        std::mem::size_of::<f32>(),
    ));

    let quantitative = info.bits_allocated == 16
        && obj
            .meta()
            .media_storage_sop_class_uid
            .trim_end_matches([' ', '\0'])
            == dicom::dictionary_std::uids::ENHANCED_PET_IMAGE_STORAGE;
    if quantitative {
        budget
            .checked_decoded_bytes(
                decoded_bytes
                    .checked_mul(2)
                    .context("DICOM PET quantitative byte count overflows usize")?,
            )
            .context("DICOM PET quantitative volume exceeds budget")?;
    }
    let mut suv = quantitative.then(Vec::new);
    let padding = obj
        .element(Tag(0x0028, 0x0120))
        .ok()
        .and_then(|element| element.to_int::<i32>().ok())
        .map(|first| {
            let last = obj
                .element(Tag(0x0028, 0x0121))
                .ok()
                .and_then(|element| element.to_int::<i32>().ok())
                .unwrap_or(first);
            first.min(last)..=first.max(last)
        });

    for frame_idx in (0..actual_n).map(source_index) {
        let frame_info = per_frame.get(frame_idx);
        let slope = frame_info
            .and_then(|pfi| pfi.rescale_slope)
            .unwrap_or(info.rescale_slope) as f32;
        let intercept = frame_info
            .and_then(|pfi| pfi.rescale_intercept)
            .unwrap_or(info.rescale_intercept) as f32;

        let frame = decode_frame_with::<DicomRsBackend>(
            obj,
            DecodeFrameRequest {
                frame_index: u32::try_from(frame_idx)
                    .context("DICOM multiframe frame index exceeds u32")?,
                transfer_syntax: ts.clone(),
                layout: PixelLayout {
                    rows: info.rows,
                    cols: info.cols,
                    samples_per_pixel: 1,
                    bits_allocated: info.bits_allocated,
                    bits_stored: info.bits_stored,
                    pixel_representation: info.pixel_representation,
                    rescale_slope: if quantitative { 1.0 } else { slope },
                    rescale_intercept: if quantitative { 0.0 } else { intercept },
                },
            },
        )
        .with_context(|| format!("DICOM backend failed for frame {frame_idx} in {:?}", path))?
        .pixels;

        if frame.len() != frame_pixels {
            bail!(
                "DICOM multiframe: frame {} in {:?} decoded {} pixels; expected {}",
                frame_idx,
                path,
                frame.len(),
                frame_pixels
            );
        }

        if let Some(suv) = &mut suv {
            let mapping = FrameMapping::read(obj, frame_idx);
            suv.extend(frame.iter().map(|&stored| {
                if padding
                    .as_ref()
                    .is_some_and(|range| range.contains(&(stored as i32)))
                {
                    return f32::NAN;
                }
                mapping
                    .as_ref()
                    .and_then(|mapping| mapping.sample(stored))
                    .map_or(f32::NAN, |value| value.into_base() as f32)
            }));
            floats.extend(frame.into_iter().map(|stored| stored * slope + intercept));
        } else {
            floats.extend_from_slice(&frame);
        }
    }

    if actual_n == 0 {
        bail!("DICOM multiframe: no pixel data decoded from {:?}", path);
    }

    // Per-frame geometry analysis: derive z-spacing from actual frame positions
    // rather than the global SliceThickness tag, which may be absent or inaccurate
    // for Enhanced CT/MR/PET with varying slice distances.
    //
    // When per_frame is populated (Enhanced MF), each PerFrameInfo may carry
    // image_position. Compute per-adjacent-pair distances projected onto the
    // shared slice normal NÌ‚ = normalize(row × col). Use the median as nominal
    // spacing; warn and resample when nonuniform or missing frames are detected.
    let (final_floats, final_n, spacing_z) = 'geometry: {
        if per_frame.len() < 2 || actual_n < 2 {
            // Fewer than 2 per-frame positions: cannot derive inter-frame spacing.
            break 'geometry (floats, actual_n, info.frame_thickness.unwrap_or(1.0));
        }

        // Resolve IOP: per-frame override first, then global.
        let iop = (0..actual_n)
            .map(source_index)
            .filter_map(|index| per_frame.get(index))
            .find_map(|pf| pf.image_orientation)
            .or(info.image_orientation);

        let Some(normal) = iop.and_then(crate::format::dicom::reader::slice_normal_from_iop) else {
            break 'geometry (floats, actual_n, info.frame_thickness.unwrap_or(1.0));
        };

        let proj: Vec<Option<f64>> = (0..actual_n)
            .map(source_index)
            .map(|index| {
                per_frame
                    .get(index)
                    .and_then(|pf| pf.image_position)
                    .map(|p| crate::format::dicom::reader::dot(p, normal))
            })
            .collect();

        if proj.iter().any(|p| p.is_none()) {
            // Not all frames have positions; fall back to global thickness.
            break 'geometry (floats, actual_n, info.frame_thickness.unwrap_or(1.0));
        }

        let src_positions: Vec<f64> = proj
            .into_iter()
            .map(|p| p.expect("all frame positions verified non-None above"))
            .collect();
        let report = crate::format::dicom::reader::analyze_slice_spacing(&src_positions);

        if report.spacing_uniformity == SpacingUniformity::Nonuniform {
            tracing::warn!(
                max_relative_deviation = report.max_relative_deviation,
                nominal_spacing_mm = report.nominal_spacing,
                n_frames = actual_n,
                path = ?path,
                "DICOM multiframe: nonuniform inter-frame spacing detected \
                 (max deviation {:.2}%); resampling to uniform grid \
                 with nominal spacing {:.4} mm",
                report.max_relative_deviation * 100.0,
                report.nominal_spacing,
            );
        }
        if report.slice_coverage == SliceCoverage::HasMissingSlices {
            tracing::warn!(
                missing_between = ?report.missing_between,
                nominal_spacing_mm = report.nominal_spacing,
                path = ?path,
                "DICOM multiframe: {} gap(s) exceed 1.5× nominal spacing \
                 ({:.4} mm), indicating missing frames; \
                 resampling to fill gaps via linear interpolation",
                report.missing_between.len(),
                report.nominal_spacing,
            );
        }

        if report.spacing_uniformity == SpacingUniformity::Nonuniform
            || report.slice_coverage == SliceCoverage::HasMissingSlices
        {
            let frame_pixels = info.rows * info.cols;
            // `floats` is already exactly `actual_n` contiguous frames (each
            // extend validated to `frame_pixels` above), so it is the flat
            // source buffer directly -- no jagged intermediate.
            let new_n = crate::format::dicom::reader::resampled_frame_count(
                &src_positions,
                report.nominal_spacing,
            );
            let resampled_bytes = new_n
                .checked_mul(frame_pixels)
                .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|count| count.checked_mul(if suv.is_some() { 2 } else { 1 }))
                .context("DICOM multiframe resampling byte count overflows usize")?;
            budget
                .checked_decoded_bytes(resampled_bytes)
                .context("DICOM multiframe resampled volume exceeds budget")?;
            let new_floats = crate::format::dicom::reader::resample_frames_linear(
                &floats,
                frame_pixels,
                &src_positions,
                report.nominal_spacing,
            );
            if let Some(samples) = suv.take() {
                suv = Some(crate::format::dicom::reader::resample_frames_linear(
                    &samples,
                    frame_pixels,
                    &src_positions,
                    report.nominal_spacing,
                ));
            }
            (new_floats, new_n, report.nominal_spacing)
        } else {
            (floats, actual_n, report.nominal_spacing)
        }
    };

    let spacing = match info.pixel_spacing {
        Some([rs, cs]) => Spacing::new([spacing_z, rs, cs]),
        None => Spacing::new([spacing_z, 1.0, 1.0]),
    };

    let origin = info.image_position.unwrap_or([0.0, 0.0, 0.0]);

    let direction = if let Some(iop) = info.image_orientation {
        let (rx, ry, rz) = (iop[0], iop[1], iop[2]);
        let (cx, cy, cz) = (iop[3], iop[4], iop[5]);
        let nx = ry * cz - rz * cy;
        let ny = rz * cx - rx * cz;
        let nz = rx * cy - ry * cx;
        let n = crate::format::dicom::reader::normalize([nx, ny, nz]).unwrap_or([0.0, 0.0, 1.0]);
        let col_data: [f64; 9] = [n[0], n[1], n[2], cx, cy, cz, rx, ry, rz];
        Direction::from_column_major(col_data)
    } else {
        Direction::identity()
    };

    Ok(MultiFrameVolume {
        data: final_floats,
        suv_bw: suv.and_then(BodyWeightSuvVolume::new),
        shape: [final_n, info.rows, info.cols],
        origin: Point::new(origin),
        spacing,
        direction,
    })
}
