//! Stored NRRD sample and acquisition-axis contracts.

use crate::{
    read_nrrd_stored as read_nrrd_stored_with_budget,
    read_nrrd_stored_series as read_nrrd_stored_series_with_budget, NrrdSpatialMetadataField,
    NrrdStoredReadError,
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
    write_header_exact(
        path,
        fields,
        bytes,
        !fields.iter().any(|field| {
            field
                .split_once(':')
                .is_some_and(|(name, _)| name.trim().eq_ignore_ascii_case("encoding"))
        }),
    )
}

fn write_header_exact(
    path: &std::path::Path,
    fields: &[&str],
    bytes: &[u8],
    add_raw_encoding: bool,
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "NRRD0004")?;
    for field in fields {
        writeln!(file, "{field}")?;
    }
    if add_raw_encoding {
        writeln!(file, "encoding: raw")?;
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
mod geometry;
#[cfg(test)]
mod payload;
#[cfg(test)]
mod reader;
#[cfg(test)]
mod series;
