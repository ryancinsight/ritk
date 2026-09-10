//! Multi-frame DICOM header extraction.
//!
//! Everything needed to answer what a multi-frame object *is* without decoding
//! a pixel: the DS parser, the tag sweep that fills a [`MultiFrameInfo`], and
//! the two entry points that open a file for it.

use anyhow::{Context, Result};
use dicom::core::Tag;
use dicom::object::InMemDicomObject;
use ritk_dicom::{parse_file_with_budget, DicomRsBackend, PixelSignedness};

use std::path::Path;

use super::super::per_frame::extract_functional_groups;
use super::super::temporal::reject_temporal_organization;
use super::super::types::MultiFrameInfo;
use crate::format::dicom::reader::types::{cs_to_arraystring, uid_to_arraystring};
use crate::format::dicom::reader::DicomReadBudget;

/// Parse a `\`-separated DICOM Decimal String (DS) field into a fixed-size array.
///
/// # Invariant
/// Returns `Some(arr)` iff the input contains at least `N` parseable `f64` values
/// separated by `\`. Non-numeric tokens are skipped. Returns `None` if fewer than
/// `N` valid numeric components exist.
pub(crate) fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]> {
    let parts: Vec<f64> = s
        .trim()
        .split('\\')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .collect();
    if parts.len() >= N {
        let mut arr = [0.0_f64; N];
        arr[..N].copy_from_slice(&parts[..N]);
        Some(arr)
    } else {
        None
    }
}

/// Extract all multi-frame header fields from an already-opened DICOM object.
///
/// # Invariants
/// - n_frames defaults to 1 when (0028,0008) is absent.
/// - bits_allocated defaults to 16 when absent.
/// - rescale_slope defaults to 1.0, rescale_intercept to 0.0 when absent.
/// - per_frame is always Vec::new(); call extract_functional_groups separately.
pub(crate) fn extract_multiframe_header(path: &Path, obj: &InMemDicomObject) -> MultiFrameInfo {
    let n_frames: usize = match obj.element(Tag(0x0028, 0x0008)) {
        Ok(element) => element
            .to_str()
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0),
        Err(_) => 1,
    };

    let rows: usize = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let cols: usize = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let bits_allocated: u16 = obj
        .element(Tag(0x0028, 0x0100))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(16);

    let samples_per_pixel: usize = obj
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);

    let pixel_representation: PixelSignedness = obj
        .element(Tag(0x0028, 0x0103))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok()) // parse u16 first
        .and_then(|v: u16| PixelSignedness::try_from(v).ok())
        .unwrap_or(PixelSignedness::Unsigned);

    let pixel_spacing = obj
        .element(Tag(0x0028, 0x0030))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| parse_ds_backslash::<2>(&s)));

    let frame_thickness = obj
        .element(Tag(0x0018, 0x0050))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok());

    let modality = obj
        .element(Tag(0x0008, 0x0060))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| cs_to_arraystring(s.trim())))
        .filter(|s| !s.is_empty());

    let sop_class_uid = obj
        .element(Tag(0x0008, 0x0016))
        .ok()
        .and_then(|e| e.to_str().ok().as_ref().and_then(|s| uid_to_arraystring(s)))
        .filter(|s| !s.is_empty());

    let image_position = obj
        .element(Tag(0x0020, 0x0032))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| parse_ds_backslash::<3>(&s)));

    let image_orientation = obj
        .element(Tag(0x0020, 0x0037))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| parse_ds_backslash::<6>(&s)));

    let rescale_slope: f64 = obj
        .element(Tag(0x0028, 0x1053))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);

    let rescale_intercept: f64 = obj
        .element(Tag(0x0028, 0x1052))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);

    MultiFrameInfo {
        path: path.to_path_buf(),
        n_frames,
        rows,
        cols,
        samples_per_pixel,
        bits_allocated,
        pixel_representation,
        pixel_spacing,
        frame_thickness,
        modality,
        sop_class_uid,
        image_position,
        image_orientation,
        rescale_slope,
        rescale_intercept,
        per_frame: Vec::new(),
    }
}

pub(crate) fn read_multiframe_info_from_object(
    path: &Path,
    obj: &InMemDicomObject,
) -> Result<MultiFrameInfo> {
    reject_temporal_organization(path, obj)?;
    let mut info = extract_multiframe_header(path, obj);
    info.per_frame = extract_functional_groups(obj, info.n_frames);
    Ok(info)
}

/// Read summary information from a multi-frame DICOM file without pixel data.
pub fn read_multiframe_info(path: impl AsRef<Path>) -> Result<MultiFrameInfo> {
    let path = path.as_ref();
    let obj = parse_file_with_budget::<DicomRsBackend, _>(path, &DicomReadBudget::DEFAULT.parser())
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;
    read_multiframe_info_from_object(path, &obj)
}
