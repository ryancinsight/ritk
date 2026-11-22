use super::*;
use crate::header::{
    HeaderAxis, HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader,
};
use anyhow::Result;
use ritk_codecs::SampleType;
use std::fs;
use tempfile::tempdir;

fn little_endian<T, const N: usize>(values: [T; 2], encode: fn(T) -> [u8; N]) -> Vec<u8> {
    values.into_iter().flat_map(encode).collect()
}

fn scalar_samples() -> [(SampleType, Vec<u8>); 10] {
    [
        (SampleType::U8, vec![0x12, 0xf0]),
        (
            SampleType::I8,
            little_endian([-117_i8, 101_i8], i8::to_le_bytes),
        ),
        (
            SampleType::U16,
            little_endian([0x9137_u16, 0xfe02_u16], u16::to_le_bytes),
        ),
        (
            SampleType::I16,
            little_endian([-32_767_i16, 32_766_i16], i16::to_le_bytes),
        ),
        (
            SampleType::U32,
            little_endian([0x9abc_def0_u32, 0x0123_4567_u32], u32::to_le_bytes),
        ),
        (
            SampleType::I32,
            little_endian([-2_000_000_001_i32, 1_234_567_890_i32], i32::to_le_bytes),
        ),
        (
            SampleType::U64,
            little_endian([u64::MAX, 0x0123_4567_89ab_cdef_u64], u64::to_le_bytes),
        ),
        (
            SampleType::I64,
            little_endian([i64::MIN + 1, 16_777_217_i64], i64::to_le_bytes),
        ),
        (
            SampleType::F32,
            little_endian(
                [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)],
                f32::to_le_bytes,
            ),
        ),
        (
            SampleType::F64,
            little_endian(
                [
                    f64::from_bits(0x8000_0000_0000_0000),
                    f64::from_bits(0x7ff8_1234_5678_9abc),
                ],
                f64::to_le_bytes,
            ),
        ),
    ]
}

fn scalar_document_bytes(
    version: HeaderVersion,
    sample_type: SampleType,
    sample_bytes: &[u8],
) -> Result<Vec<u8>> {
    let header = NiftiHeader::new_with_version(
        version,
        HeaderDims {
            nx: 2,
            ny: 1,
            nz: 1,
        },
        1,
        HeaderAxis::Volume,
        NiftiDatatype::try_from(sample_type)?,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;
    let mut bytes = header.encode();
    bytes.resize(header.vox_offset, 0);
    bytes.extend_from_slice(sample_bytes);
    Ok(bytes)
}

#[test]
fn documents_preserve_all_scalar_payload_bits_in_both_header_versions() -> Result<()> {
    let directory = tempdir()?;

    for version in [HeaderVersion::One, HeaderVersion::Two] {
        for (sample_type, sample_bytes) in scalar_samples() {
            let source = scalar_document_bytes(version, sample_type, &sample_bytes)?;
            let document = NiftiDocument::from_bytes(&source)?;
            let datatype = NiftiDatatype::try_from(sample_type)?;

            assert_eq!(document.header().datatype_code, datatype.code());
            assert_eq!(
                document.header().bits_per_sample,
                u16::try_from(sample_type.byte_width() * 8)?
            );
            assert_eq!(document.sample_bytes(), sample_bytes);

            for extension in ["nii", "nii.gz"] {
                let path = directory
                    .path()
                    .join(format!("roundtrip-{}-{extension}", datatype.code()));
                document.write(&path)?;
                let reread = NiftiDocument::read(path)?;
                assert_eq!(reread.uncompressed_bytes(), source);
            }
        }
    }

    Ok(())
}

#[test]
fn invalid_scalar_documents_leave_existing_destinations_unchanged() -> Result<()> {
    let directory = tempdir()?;
    let destination = directory.path().join("existing.nii.gz");
    let original_destination = b"keep existing destination bytes";
    fs::write(&destination, original_destination)?;

    for (version, version_name) in [
        (HeaderVersion::One, "nifti1"),
        (HeaderVersion::Two, "nifti2"),
    ] {
        let sample_bytes = little_endian([1.25_f64, -2.5_f64], f64::to_le_bytes);
        let valid = scalar_document_bytes(version, SampleType::F64, &sample_bytes)?;
        let (datatype_offset, bitpix_offset, payload_offset) = match version {
            HeaderVersion::One => (70, 72, 352),
            HeaderVersion::Two => (12, 14, 544),
        };

        let mut unsupported = valid.clone();
        unsupported[datatype_offset..datatype_offset + 2].copy_from_slice(&1536_i16.to_le_bytes());

        let mut mismatched = valid.clone();
        mismatched[datatype_offset..datatype_offset + 2].copy_from_slice(&16_i16.to_le_bytes());
        mismatched[bitpix_offset..bitpix_offset + 2].copy_from_slice(&64_i16.to_le_bytes());

        let mut truncated = valid;
        truncated.truncate(payload_offset + 8);

        enum FailureKind {
            Header,
            Payload,
        }

        for (name, invalid, failure_kind) in [
            ("unsupported", unsupported, FailureKind::Header),
            ("mismatched", mismatched, FailureKind::Header),
            ("truncated", truncated, FailureKind::Payload),
        ] {
            let source = directory
                .path()
                .join(format!("invalid-{version_name}-{name}.nii"));
            fs::write(&source, invalid)?;

            let error = transcode_nifti_document(&source, &destination)
                .expect_err("invalid scalar documents must fail before writing");
            match failure_kind {
                FailureKind::Header => assert!(matches!(error, NiftiDocumentError::Header(_))),
                FailureKind::Payload => assert!(matches!(error, NiftiDocumentError::Payload(_))),
            }
            assert_eq!(fs::read(&destination)?.as_slice(), original_destination);
        }
    }

    Ok(())
}

#[test]
fn compute_scaling_precedes_binary32_rounding() -> Result<()> {
    let backend = coeus_core::SequentialBackend;
    let values = little_endian([16_777_217_i32, 16_777_219], i32::to_le_bytes);
    for version in [HeaderVersion::One, HeaderVersion::Two] {
        let mut bytes = scalar_document_bytes(version, SampleType::I32, &values)?;
        match version {
            HeaderVersion::One => bytes[116..120].copy_from_slice(&(-16_777_216_f32).to_le_bytes()),
            HeaderVersion::Two => bytes[184..192].copy_from_slice(&(-16_777_216_f64).to_le_bytes()),
        }
        let image = crate::read_nifti_from_bytes(&bytes, &backend)?;
        assert_eq!(image.data_cow_on(&backend)?.as_ref(), [1.0, 3.0]);
        let series = crate::read_nifti_series_from_bytes(&bytes, &backend)?;
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].data_cow_on(&backend)?.as_ref(), [1.0, 3.0]);
    }
    Ok(())
}

#[test]
fn compute_scaling_handles_disabled_and_invalid_coefficients() -> Result<()> {
    let backend = coeus_core::SequentialBackend;
    let values = little_endian([-0.0_f32, -3.25], f32::to_le_bytes);
    let mut bytes = scalar_document_bytes(HeaderVersion::Two, SampleType::F32, &values)?;
    bytes[176..184].copy_from_slice(&(-0.0_f64).to_le_bytes());
    bytes[184..192].copy_from_slice(&f64::NAN.to_le_bytes());
    let image = crate::read_nifti_from_bytes(&bytes, &backend)?;
    let actual = image.data_cow_on(&backend)?;
    assert_eq!(actual[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(actual[1], -3.25);
    bytes[176..184].copy_from_slice(&(-2.0_f64).to_le_bytes());
    assert!(crate::read_nifti_from_bytes(&bytes, &backend).is_err());
    bytes[184..192].copy_from_slice(&1.0_f64.to_le_bytes());
    let image = crate::read_nifti_from_bytes(&bytes, &backend)?;
    assert_eq!(image.data_cow_on(&backend)?.as_ref(), [1.0, 7.5]);
    Ok(())
}
