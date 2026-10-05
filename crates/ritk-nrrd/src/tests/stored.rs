//! Stored NRRD sample and acquisition-axis contracts.

use crate::{
    read_nrrd_document, read_nrrd_stored as read_nrrd_stored_with_budget,
    read_nrrd_stored_series as read_nrrd_stored_series_with_budget, write_nrrd_document,
    NrrdSpatialMetadataField, NrrdStoredReadError,
};
use anyhow::Result;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ImageReadBudget, IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction};
use std::io::Write;
use tempfile::tempdir;

fn read_nrrd_stored<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<StoredVolume, NrrdStoredReadError> {
    read_nrrd_stored_with_budget(path, ImageReadBudget::DEFAULT)
}

fn read_nrrd_stored_series<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<StoredSeries, NrrdStoredReadError> {
    read_nrrd_stored_series_with_budget(path, ImageReadBudget::DEFAULT)
}

fn write_header(path: &std::path::Path, fields: &[&str], bytes: &[u8]) -> Result<()> {
    let mut version = "NRRD0004";
    let mut encoding = Some("raw");
    for field in fields {
        let Some((name, _)) = field.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("measurement frame") {
            version = "NRRD0005";
        }
        if name.trim().eq_ignore_ascii_case("encoding") {
            encoding = None;
        }
    }
    write_header_with_version(path, fields, bytes, version, encoding)
}

fn write_header_with_version(
    path: &std::path::Path,
    fields: &[&str],
    bytes: &[u8],
    version: &str,
    encoding: Option<&str>,
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "{version}")?;
    for field in fields {
        writeln!(file, "{field}")?;
    }
    if let Some(encoding) = encoding {
        writeln!(file, "encoding: {encoding}")?;
    }
    writeln!(file)?;
    file.write_all(bytes)?;
    Ok(())
}

pub(super) fn sample_cases() -> Vec<(SampleType, &'static str, SampleBuffer)> {
    vec![
        (
            SampleType::U8,
            "unsigned char",
            SampleBuffer::from_samples(vec![0_u8, u8::MAX]),
        ),
        (
            SampleType::I8,
            "signed char",
            SampleBuffer::from_samples(vec![i8::MIN, i8::MAX]),
        ),
        (
            SampleType::U16,
            "unsigned short",
            SampleBuffer::from_samples(vec![0_u16, u16::MAX]),
        ),
        (
            SampleType::I16,
            "short",
            SampleBuffer::from_samples(vec![i16::MIN, i16::MAX]),
        ),
        (
            SampleType::U32,
            "unsigned int",
            SampleBuffer::from_samples(vec![0_u32, u32::MAX]),
        ),
        (
            SampleType::I32,
            "int",
            SampleBuffer::from_samples(vec![i32::MIN, i32::MAX]),
        ),
        (
            SampleType::U64,
            "unsigned long long",
            SampleBuffer::from_samples(vec![0_u64, u64::MAX]),
        ),
        (
            SampleType::I64,
            "long long",
            SampleBuffer::from_samples(vec![i64::MIN, i64::MAX]),
        ),
        (
            SampleType::F32,
            "float",
            SampleBuffer::from_samples(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_0042),
            ]),
        ),
        (
            SampleType::F64,
            "double",
            SampleBuffer::from_samples(vec![
                f64::from_bits(0x8000_0000_0000_0000),
                f64::from_bits(0x7ff8_0000_0000_0042),
            ]),
        ),
    ]
}

pub(super) fn stored_u64(values: Vec<u64>, calibration: IntensityCalibration) -> StoredVolume {
    let shape = [1, 1, values.len()];
    StoredVolume::new(
        shape,
        SampleBuffer::from_samples(values),
        ImageMetadata::default_for_shape(shape),
        CoordinateMap::Cartesian,
        calibration,
    )
    .expect("valid test volume")
}

#[cfg(test)]
mod detached;
#[cfg(test)]
mod document;
#[cfg(test)]
mod geometry;
#[cfg(test)]
mod payload;
#[cfg(test)]
mod reader;
#[cfg(test)]
mod series;
