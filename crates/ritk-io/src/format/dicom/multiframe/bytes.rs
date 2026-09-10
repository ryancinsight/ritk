//! Byte-payload entry points for DICOM multi-frame loading.

use std::path::Path;

use anyhow::{Context, Result};
use dicom::object::DefaultDicomObject;
use ritk_dicom::{parse_bytes_with_budget, DicomRsBackend};

use super::reader::{
    load_multiframe_flat_from_object, read_multiframe_info_from_object, MultiFrameVolume,
};
use super::types::MultiFrameInfo;
use crate::format::dicom::reader::DicomReadBudget;

/// Read multi-frame metadata from a named Part 10 byte payload.
///
/// `name` is a diagnostic identity supplied by the caller; it is never opened
/// as a filesystem path. The same parser budget used by RITK's scanner bounds
/// materialization before DICOM metadata is inspected.
pub fn read_multiframe_info_from_bytes(
    name: impl AsRef<Path>,
    bytes: &[u8],
) -> Result<MultiFrameInfo> {
    read_multiframe_info_from_bytes_with_budget(name, bytes, &DicomReadBudget::DEFAULT)
}

/// Read multi-frame metadata from bytes under an explicit parser budget.
pub fn read_multiframe_info_from_bytes_with_budget(
    name: impl AsRef<Path>,
    bytes: &[u8],
    budget: &DicomReadBudget,
) -> Result<MultiFrameInfo> {
    let name = name.as_ref();
    let obj = parse_bytes_with_budget::<DicomRsBackend>(bytes, &budget.parser())
        .with_context(|| format!("failed to parse DICOM multiframe bytes {:?}", name))?;
    read_multiframe_info_from_object(name, &obj)
}

/// Decode a named Part 10 byte payload into a substrate-free multi-frame
/// volume without writing it to disk.
pub fn load_dicom_multiframe_flat_from_bytes(
    name: impl AsRef<Path>,
    bytes: &[u8],
) -> Result<MultiFrameVolume> {
    load_dicom_multiframe_flat_from_bytes_with_budget(name, bytes, &DicomReadBudget::DEFAULT)
}

/// Decode a named Part 10 byte payload under an explicit DICOM read budget.
pub fn load_dicom_multiframe_flat_from_bytes_with_budget(
    name: impl AsRef<Path>,
    bytes: &[u8],
    budget: &DicomReadBudget,
) -> Result<MultiFrameVolume> {
    let name = name.as_ref();
    let obj: DefaultDicomObject =
        parse_bytes_with_budget::<DicomRsBackend>(bytes, &budget.parser())
            .with_context(|| format!("failed to parse DICOM multiframe bytes {:?}", name))?;
    load_multiframe_flat_from_object(name, &obj, budget)
}
