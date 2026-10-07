use super::*;
use crate::header::{
    write_single_file_bytes, HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader,
};
use anyhow::Result;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
use tempfile::tempdir;

fn document_bytes(sform_x: f64) -> Vec<u8> {
    let mut header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 2,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [sform_x, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )
    .expect("valid test header");
    header.qform_code = 1;
    header.sform_code = 2;
    header.vox_offset = 368;
    let mut bytes = header.encode();
    bytes.extend_from_slice(&[1, 0, 0, 0]);
    bytes.extend_from_slice(&16_i32.to_le_bytes());
    bytes.extend_from_slice(&6_i32.to_le_bytes());
    bytes.extend_from_slice(&[11, 22, 33, 44, 55, 66, 77, 88]);
    bytes.extend_from_slice(&0x8000_0000_u32.to_le_bytes());
    bytes.extend_from_slice(&0x7fc0_1234_u32.to_le_bytes());
    bytes
}

fn scalar_document_bytes(
    datatype: NiftiDatatype,
    version: HeaderVersion,
    samples: &[u8],
) -> Vec<u8> {
    let mut header = NiftiHeader::new_with_version(
        version,
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        1,
        datatype,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )
    .expect("valid scalar header");
    header.sform_code = 1;
    let mut bytes = header.encode();
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(samples);
    bytes
}

fn nifti2_transform_bytes(configure: impl FnOnce(&mut NiftiHeader)) -> Vec<u8> {
    let mut header = NiftiHeader::new_with_version(
        HeaderVersion::Two,
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        1,
        NiftiDatatype::Float64,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )
    .expect("valid NIfTI-2 transform fixture");
    configure(&mut header);
    write_single_file_bytes(&header, &1.5_f64.to_le_bytes())
}

fn scalar_sample_payloads() -> [(SampleType, Vec<u8>); 10] {
    [
        (SampleType::U8, vec![0xfe]),
        (SampleType::I8, vec![0x80]),
        (SampleType::U16, 0xf123_u16.to_le_bytes().to_vec()),
        (SampleType::I16, i16::MIN.to_le_bytes().to_vec()),
        (SampleType::U32, 0xfedc_ba98_u32.to_le_bytes().to_vec()),
        (SampleType::I32, i32::MIN.to_le_bytes().to_vec()),
        (
            SampleType::U64,
            0xfedc_ba98_7654_3210_u64.to_le_bytes().to_vec(),
        ),
        (SampleType::I64, (i64::MIN + 1).to_le_bytes().to_vec()),
        (SampleType::F32, 0x7fc0_1234_u32.to_le_bytes().to_vec()),
        (
            SampleType::F64,
            0x7ff8_0000_0000_1234_u64.to_le_bytes().to_vec(),
        ),
    ]
}

fn assert_scalar_document_round_trip(
    sample_type: SampleType,
    expected_bytes: &[u8],
    version: HeaderVersion,
    directory: &std::path::Path,
) -> Result<()> {
    let buffer = SampleBuffer::decode(
        sample_type,
        expected_bytes,
        ByteOrder::LeastSignificantByteFirst,
    )
    .expect("complete typed sample");
    let encoded = buffer
        .encode(ByteOrder::LeastSignificantByteFirst)
        .expect("stored sample encoding");
    assert_eq!(encoded.as_slice(), expected_bytes);

    let datatype = NiftiDatatype::try_from(sample_type)
        .expect("NIfTI supports every fixed-width codec sample");
    let source = scalar_document_bytes(datatype, version, expected_bytes);
    let document = NiftiDocument::from_bytes(&source).expect("valid scalar document");
    assert_eq!(document.header().datatype_code, datatype.code());
    assert_eq!(
        document.header().bits_per_sample,
        u16::try_from(datatype.bitpix()).expect("positive NIfTI bit width")
    );
    assert_eq!(document.sample_bytes(), expected_bytes);

    for name in ["round-trip.nii", "round-trip.nii.gz"] {
        let path = directory.join(name);
        document.write(&path).expect("document write");
        let reread = NiftiDocument::read(path).expect("document read");
        assert_eq!(reread.uncompressed_bytes(), source);
        assert_eq!(reread.sample_bytes(), expected_bytes);
    }
    Ok(())
}

#[test]
fn document_round_trip_preserves_header_extension_and_sample_bits() {
    let bytes = document_bytes(1.0);
    let document = NiftiDocument::from_bytes(&bytes).expect("valid document");

    assert_eq!(document.sample_bytes(), &bytes[368..]);
    assert_eq!(document.header().qform_code, 1);
    assert_eq!(document.header().sform_code, 2);
    assert_eq!(document.header().bits_per_sample, 32);
    assert_eq!(
        document.header().spatial_forms,
        SpatialFormRelation::CompatibleHandedness
    );
    let directory = tempdir().expect("temporary directory is available");
    for name in ["roundtrip.nii", "roundtrip.nii.gz"] {
        let path = directory.path().join(name);
        document.write(&path).expect("document writes");
        let reread = NiftiDocument::read(path).expect("written document reads");
        assert_eq!(reread.uncompressed_bytes(), bytes);
    }
}

#[test]
fn documents_preserve_every_stored_scalar_type_and_sample_bit() -> Result<()> {
    let directory = tempdir().expect("temporary directory is available");

    for version in [HeaderVersion::One, HeaderVersion::Two] {
        for (sample_type, expected_bytes) in scalar_sample_payloads() {
            assert_scalar_document_round_trip(
                sample_type,
                &expected_bytes,
                version,
                directory.path(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn invalid_scalar_documents_preserve_existing_output_for_both_versions() -> Result<()> {
    #[derive(Clone, Copy)]
    enum Failure {
        Header,
        Payload,
    }

    let directory = tempdir().expect("temporary directory is available");
    let sample = 0x7ff8_0000_0000_1234_u64.to_le_bytes();
    for version in [HeaderVersion::One, HeaderVersion::Two] {
        let (datatype_offset, bitpix_offset) = match version {
            HeaderVersion::One => (70, 72),
            HeaderVersion::Two => (12, 14),
        };
        let mut unsupported = scalar_document_bytes(NiftiDatatype::Float64, version, &sample);
        unsupported[datatype_offset..datatype_offset + 2]
            .copy_from_slice(&1536_i16.to_le_bytes());

        let mut mismatched = scalar_document_bytes(NiftiDatatype::Float64, version, &sample);
        mismatched[bitpix_offset..bitpix_offset + 2].copy_from_slice(&32_i16.to_le_bytes());

        let truncated = scalar_document_bytes(
            NiftiDatatype::Float64,
            version,
            &[1, 2, 3, 4, 5, 6, 7],
        );

        for (name, bytes, failure) in [
            ("unsupported", unsupported, Failure::Header),
            ("bitpix", mismatched, Failure::Header),
            ("truncated", truncated, Failure::Payload),
        ] {
            let parsed = NiftiDocument::from_bytes(&bytes);
            match (failure, parsed) {
                (Failure::Header, Err(NiftiDocumentError::Header(_)))
                | (Failure::Payload, Err(NiftiDocumentError::Payload(_))) => {}
                (Failure::Header, other) => {
                    panic!("{name} NIfTI-{version:?} should fail during header parse: {other:?}")
                }
                (Failure::Payload, other) => {
                    panic!("{name} NIfTI-{version:?} should fail during payload validation: {other:?}")
                }
            }

            let source = directory
                .path()
                .join(format!("{version:?}-{name}.nii"));
            let destination = directory
                .path()
                .join(format!("{version:?}-{name}-existing.nii"));
            fs::write(&source, bytes).expect("invalid source fixture writes");
            fs::write(&destination, b"preserve destination")
                .expect("destination fixture writes");
            let error = transcode_nifti_document(&source, &destination)
                .expect_err("invalid scalar source must stop before writing");
            match (failure, error) {
                (Failure::Header, NiftiDocumentError::Header(_))
                | (Failure::Payload, NiftiDocumentError::Payload(_)) => {}
                (Failure::Header, other) => {
                    panic!("{name} NIfTI-{version:?} returned the wrong error: {other}")
                }
                (Failure::Payload, other) => {
                    panic!("{name} NIfTI-{version:?} returned the wrong error: {other}")
                }
            }
            assert_eq!(
                fs::read(destination).expect("destination remains readable"),
                b"preserve destination"
            );
        }
    }
    Ok(())
}

#[test]
fn gzip_round_trip_drains_trailer_and_preserves_document() {
    let bytes = document_bytes(1.0);
    let encoded = encode_gzip(&bytes).expect("gzip encoding succeeds");

    let mut corrupt = encoded;
    let last = corrupt.last_mut().expect("gzip stream has a trailer");
    *last ^= 1;
    assert!(matches!(
        NiftiDocument::from_bytes(&corrupt),
        Err(NiftiDocumentError::Compression(_))
    ));
}

#[test]
fn opposite_qform_and_sform_handedness_is_explicit() {
    let document = NiftiDocument::from_bytes(&document_bytes(-1.0))
        .expect("both valid forms remain representable");
    assert_eq!(
        document.header().spatial_forms,
        SpatialFormRelation::HandednessConflict
    );
}

#[test]
fn qform_only_accepts_valid_transform_and_rejects_nonfinite_fields() {
    let valid = nifti2_transform_bytes(|header| {
        header.qform_code = 1;
        header.sform_code = 0;
    });
    let document = NiftiDocument::from_bytes(&valid).expect("valid qform-only document");
    assert_eq!(
        document.header().spatial_forms,
        SpatialFormRelation::QformOnly
    );

    for corrupt_quaternion in [true, false] {
        let bytes = nifti2_transform_bytes(|header| {
            header.qform_code = 1;
            header.sform_code = 0;
            if corrupt_quaternion {
                header.quatern_b = f64::NAN;
            } else {
                header.quatern_x = f64::INFINITY;
            }
        });
        assert!(matches!(
            NiftiDocument::from_bytes(&bytes),
            Err(NiftiDocumentError::SpatialForms(_))
        ));
    }
}

#[test]
fn sform_only_rejects_nonfinite_and_degenerate_transforms() {
    let nonfinite = nifti2_transform_bytes(|header| {
        header.qform_code = 0;
        header.sform_code = 1;
        header.srow_y[1] = f64::NAN;
    });
    assert!(matches!(
        NiftiDocument::from_bytes(&nonfinite),
        Err(NiftiDocumentError::SpatialForms(_))
    ));

    let degenerate = nifti2_transform_bytes(|header| {
        header.qform_code = 0;
        header.sform_code = 1;
        header.srow_z = [0.0; 4];
    });
    assert!(matches!(
        NiftiDocument::from_bytes(&degenerate),
        Err(NiftiDocumentError::DegenerateSform)
    ));
}

#[test]
fn nifti2_sform_classification_handles_determinant_magnitude_extremes() {
    let encoded = nifti2_transform_bytes(|header| {
        header.qform_code = 0;
        header.sform_code = 1;
        header.srow_x = [2.0_f64.powi(1000), 0.0, 0.0, 0.0];
        header.srow_y = [0.0, 2.0_f64.powi(1000), 0.0, 0.0];
        header.srow_z = [0.0, 0.0, 2.0_f64.powi(1000), 0.0];
    });

    let document = NiftiDocument::from_bytes(&encoded)
        .expect("finite nonsingular NIfTI-2 sform is classified without forming its determinant");
    assert_eq!(
        document.header().spatial_forms,
        SpatialFormRelation::SformOnly
    );
}

#[test]
fn negative_spatial_form_codes_are_rejected_during_header_parse() {
    let bytes = nifti2_transform_bytes(|header| {
        header.qform_code = -1;
        header.sform_code = 0;
    });
    assert!(matches!(
        NiftiDocument::from_bytes(&bytes),
        Err(NiftiDocumentError::Header(_))
    ));
}

#[test]
fn invalid_source_does_not_change_existing_destination() {
    let directory = tempdir().expect("temporary directory is available");
    let source = directory.path().join("invalid.nii.gz");
    let destination = directory.path().join("existing.nii");
    fs::write(&source, [0x1f, 0x8b, 0, 0]).expect("invalid source fixture writes");
    fs::write(&destination, b"keep me").expect("destination fixture writes");

    assert!(transcode_nifti_document(&source, &destination).is_err());
    assert_eq!(
        fs::read(destination).expect("destination remains readable"),
        b"keep me"
    );
}
