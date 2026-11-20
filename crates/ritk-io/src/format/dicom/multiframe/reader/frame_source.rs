//! Instance-local geometry and pixel decoding.

use std::{ops::RangeInclusive, path::Path};

use anyhow::{bail, Context, Result};
use dicom::{core::Tag, object::DefaultDicomObject};
use ritk_dicom::{
    decode_frame_with, DecodeFrameRequest, DicomRsBackend, PixelLayout, TransferSyntaxKind,
};

use super::super::{
    enhanced_pet::FrameMapping,
    per_frame::extract_functional_groups,
    types::{MultiFrameInfo, PerFrameInfo},
};
use super::header::extract_multiframe_header;
use crate::format::dicom::reader::DicomReadBudget;

#[derive(Clone, Copy)]
pub(in super::super) struct FrameAddress {
    pub source: usize,
    pub local: usize,
}

pub(in super::super) struct FrameSource<'a> {
    pub obj: &'a DefaultDicomObject,
    pub path: &'a Path,
    pub info: MultiFrameInfo,
    pub geometry: Vec<PerFrameInfo>,
    syntax: TransferSyntaxKind,
    padding: Option<RangeInclusive<i32>>,
}

impl<'a> FrameSource<'a> {
    pub(super) fn read(
        path: &'a Path,
        obj: &'a DefaultDicomObject,
        budget: &DicomReadBudget,
    ) -> Result<Self> {
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

        let info = extract_multiframe_header(path, obj)?;
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

        let source_bytes = info
            .n_frames
            .checked_mul(info.rows)
            .and_then(|count| count.checked_mul(info.cols))
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .context("DICOM multiframe decoded byte count overflows usize")?;
        budget.checked_decoded_bytes(source_bytes)?;
        let geometry = extract_functional_groups(obj, info.n_frames);
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

        Ok(Self {
            obj,
            path,
            info,
            geometry,
            syntax: ts,
            padding,
        })
    }

    pub(super) fn append_frame(
        &self,
        frame_idx: usize,
        frame_pixels: usize,
        floats: &mut Vec<f32>,
        suv: Option<&mut Vec<f32>>,
    ) -> Result<()> {
        let Self {
            obj,
            path,
            info,
            geometry,
            syntax: ts,
            padding,
        } = self;
        let quantitative = suv.is_some();
        let frame_info = geometry.get(frame_idx);
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

        if let Some(suv) = suv {
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
        Ok(())
    }
}
