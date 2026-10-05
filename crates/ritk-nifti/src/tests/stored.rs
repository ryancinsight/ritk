use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ImageReadBudget, ImageReadBudgetError, ImageReadResource, IntensityCalibration,
    LinearCalibration, LutOutputBits, ModalityLookupTable, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, SliceSeries, SliceTransform, Spacing};

use crate::header::{
    write_single_file_bytes, HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader,
};
use crate::{
    read_nifti_stored, read_nifti_stored_from_bytes, write_nifti_stored, NiftiStoredReadError,
    NiftiStoredWriteError,
};

fn sample_case(
    datatype: NiftiDatatype,
    samples: SampleBuffer,
) -> Result<(NiftiDatatype, SampleBuffer, Vec<u8>)> {
    let payload = samples.encode(ByteOrder::LeastSignificantByteFirst)?;
    Ok((datatype, samples, payload))
}

fn stored_samples() -> Result<Vec<(NiftiDatatype, SampleBuffer, Vec<u8>)>> {
    Ok(vec![
        sample_case(
            NiftiDatatype::Uint8,
            SampleBuffer::from_samples(vec![u8::MIN, u8::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Int8,
            SampleBuffer::from_samples(vec![i8::MIN, i8::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Uint16,
            SampleBuffer::from_samples(vec![u16::MIN, u16::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Int16,
            SampleBuffer::from_samples(vec![i16::MIN, i16::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Uint32,
            SampleBuffer::from_samples(vec![16_777_217_u32, u32::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Int32,
            SampleBuffer::from_samples(vec![i32::MIN, i32::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Uint64,
            SampleBuffer::from_samples(vec![9_007_199_254_740_993_u64, u64::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Int64,
            SampleBuffer::from_samples(vec![i64::MIN, i64::MAX]),
        )?,
        sample_case(
            NiftiDatatype::Float32,
            SampleBuffer::from_samples(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc1_2345),
            ]),
        )?,
        sample_case(
            NiftiDatatype::Float64,
            SampleBuffer::from_samples(vec![
                f64::from_bits(0x8000_0000_0000_0000),
                f64::from_bits(0x7ff8_0000_0000_2345),
            ]),
        )?,
    ])
}

fn stored_metadata() -> ImageMetadata<3> {
    ImageMetadata::new(
        Point::new([
            10.123_456_789_012_3,
            -20.987_654_321_123_4,
            30.111_111_111_111,
        ]),
        Spacing::new([2.5, 1.25, 0.5]),
        Direction::identity(),
    )
}

fn stored_volume(
    shape: [usize; 3],
    samples: SampleBuffer,
    coordinate_map: CoordinateMap,
    calibration: IntensityCalibration,
) -> Result<StoredVolume> {
    Ok(StoredVolume::new(
        shape,
        samples,
        stored_metadata(),
        coordinate_map,
        calibration,
    )?)
}

fn decoded_file(path: &Path) -> Result<Vec<u8>> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gz"))
    {
        let mut decoded = Vec::new();
        GzDecoder::new(File::open(path)?).read_to_end(&mut decoded)?;
        Ok(decoded)
    } else {
        Ok(std::fs::read(path)?)
    }
}

fn header(datatype: NiftiDatatype, version: HeaderVersion, volumes: usize) -> Result<NiftiHeader> {
    NiftiHeader::new_with_version(
        version,
        HeaderDims {
            nx: 2,
            ny: 1,
            nz: 1,
        },
        volumes,
        datatype,
        HeaderSpatial {
            pixdim: [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            srow_x: [-2.0, 0.0, 0.0, -10.0],
            srow_y: [0.0, -3.0, 0.0, -20.0],
            srow_z: [0.0, 0.0, 4.0, 30.0],
        },
    )
}

fn big_endian_file(header: &NiftiHeader, payload: &[u8]) -> Vec<u8> {
    let mut bytes = write_single_file_bytes(header, payload);
    let fields = match header.version {
        HeaderVersion::One => {
            let mut fields = vec![
                (0, 4),
                (70, 2),
                (72, 2),
                (108, 4),
                (112, 4),
                (116, 4),
                (252, 2),
                (254, 2),
            ];
            fields.extend((0..8).map(|index| (40 + index * 2, 2)));
            fields.extend((0..8).map(|index| (76 + index * 4, 4)));
            fields.extend((0..6).map(|index| (256 + index * 4, 4)));
            fields.extend((0..12).map(|index| (280 + index * 4, 4)));
            fields
        }
        HeaderVersion::Two => {
            let mut fields = vec![
                (0, 4),
                (12, 2),
                (14, 2),
                (168, 8),
                (176, 8),
                (184, 8),
                (344, 4),
                (348, 4),
                (500, 4),
            ];
            fields.extend((0..8).map(|index| (16 + index * 8, 8)));
            fields.extend((0..8).map(|index| (104 + index * 8, 8)));
            fields.extend((0..6).map(|index| (352 + index * 8, 8)));
            fields.extend((0..12).map(|index| (400 + index * 8, 8)));
            fields
        }
    };
    for (offset, width) in fields {
        bytes[offset..offset + width].reverse();
    }
    bytes
}

#[test]
fn stored_reader_preserves_every_supported_sample_representation() -> Result<()> {
    for version in [HeaderVersion::One, HeaderVersion::Two] {
        for (datatype, expected, payload) in stored_samples()? {
            let bytes = write_single_file_bytes(&header(datatype, version, 1)?, &payload);
            let volume = read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT)?;

            assert_eq!(volume.shape(), [1, 1, 2]);
            assert_eq!(volume.samples().sample_type(), expected.sample_type());
            assert_eq!(
                volume
                    .samples()
                    .encode(ByteOrder::LeastSignificantByteFirst)?,
                payload
            );
            assert_eq!(volume.metadata().origin().to_array(), [10.0, 20.0, 30.0]);
            assert_eq!(volume.metadata().spacing().to_array(), [4.0, 3.0, 2.0]);
            assert_eq!(volume.calibration(), &IntensityCalibration::Identity);
        }
    }
    Ok(())
}

#[test]
fn stored_reader_preserves_every_supported_sample_representation_in_big_endian() -> Result<()> {
    for version in [HeaderVersion::One, HeaderVersion::Two] {
        for (datatype, expected, _) in stored_samples()? {
            let payload = expected.encode(ByteOrder::MostSignificantByteFirst)?;
            let bytes = big_endian_file(&header(datatype, version, 1)?, &payload);
            let volume = read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT)?;

            assert_eq!(volume.samples().sample_type(), expected.sample_type());
            assert_eq!(
                volume
                    .samples()
                    .encode(ByteOrder::LeastSignificantByteFirst)?,
                expected.encode(ByteOrder::LeastSignificantByteFirst)?
            );
        }
    }
    Ok(())
}

#[test]
fn stored_reader_keeps_nifti_calibration_separate_from_sample_values() -> Result<()> {
    let mut header = header(NiftiDatatype::Int16, HeaderVersion::One, 1)?;
    header.scl_slope = 2.5;
    header.scl_inter = -1024.0;
    let payload = SampleBuffer::from_samples(vec![-3_i16, 11])
        .encode(ByteOrder::LeastSignificantByteFirst)?;
    let bytes = write_single_file_bytes(&header, &payload);

    let volume = read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT)?;
    let expected = SampleBuffer::from_samples(vec![-3_i16, 11]);
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        expected.encode(ByteOrder::LeastSignificantByteFirst)?
    );
    assert_eq!(
        volume.calibration(),
        &IntensityCalibration::Linear(LinearCalibration::new(2.5, -1024.0)?)
    );
    Ok(())
}

#[test]
fn stored_reader_converts_meter_geometry_to_lps_millimeters() -> Result<()> {
    let mut header = header(NiftiDatatype::Uint8, HeaderVersion::Two, 1)?;
    header.xyzt_units = 1;
    header.srow_x = [-0.002, 0.0, 0.0, -0.01];
    header.srow_y = [0.0, -0.003, 0.0, -0.02];
    header.srow_z = [0.0, 0.0, 0.004, 0.03];
    let bytes = write_single_file_bytes(&header, &[4, 9]);

    let volume = read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT)?;
    assert_eq!(volume.metadata().origin().to_array(), [10.0, 20.0, 30.0]);
    assert_eq!(volume.metadata().spacing().to_array(), [4.0, 3.0, 2.0]);
    Ok(())
}

#[test]
fn stored_reader_accepts_gzip_and_applies_each_budget() -> Result<()> {
    let header = header(NiftiDatatype::Uint32, HeaderVersion::Two, 1)?;
    let payload = SampleBuffer::from_samples(vec![16_777_217_u32, u32::MAX])
        .encode(ByteOrder::LeastSignificantByteFirst)?;
    let bytes = write_single_file_bytes(&header, &payload);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&bytes)?;
    let compressed = encoder.finish()?;
    let volume = read_nifti_stored_from_bytes(&compressed, ImageReadBudget::DEFAULT)?;
    let expected = SampleBuffer::from_samples(vec![16_777_217_u32, u32::MAX]);
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        expected.encode(ByteOrder::LeastSignificantByteFirst)?
    );

    let encoded_limit = ImageReadBudget::new(
        u64::try_from(compressed.len())? - 1,
        ImageReadBudget::DEFAULT.max_decoded_bytes(),
        1,
    )?;
    let error = read_nifti_stored_from_bytes(&compressed, encoded_limit)
        .expect_err("compressed bytes over the declared budget must fail");
    assert!(matches!(
        error,
        NiftiStoredReadError::Budget(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::EncodedBytes,
            ..
        })
    ));

    let decoded_limit = ImageReadBudget::new(u64::MAX, 8, 1)?;
    let error = read_nifti_stored_from_bytes(&bytes, decoded_limit)
        .expect_err("expanded file extent over budget must fail before sample allocation");
    assert!(matches!(
        error,
        NiftiStoredReadError::Budget(ImageReadBudgetError::Exceeded {
            resource: ImageReadResource::DecodedBytes,
            ..
        })
    ));
    Ok(())
}

#[test]
fn stored_reader_rejects_acquisition_axes_and_truncated_samples() -> Result<()> {
    let series_header = header(NiftiDatatype::Uint8, HeaderVersion::One, 2)?;
    let series_bytes = write_single_file_bytes(&series_header, &[1, 2, 3, 4]);
    assert!(matches!(
        read_nifti_stored_from_bytes(&series_bytes, ImageReadBudget::DEFAULT),
        Err(NiftiStoredReadError::UnsupportedRank { rank: 4 })
    ));

    let volume_header = header(NiftiDatatype::Uint8, HeaderVersion::One, 1)?;
    let truncated = write_single_file_bytes(&volume_header, &[7]);
    assert!(matches!(
        read_nifti_stored_from_bytes(&truncated, ImageReadBudget::DEFAULT),
        Err(NiftiStoredReadError::Sample(
            ritk_codecs::SampleError::TruncatedInput {
                sample_type: SampleType::U8,
                sample_count: 2,
                completed_samples: 1,
            }
        ))
    ));
    Ok(())
}

#[test]
fn stored_reader_rejects_nifti_extension_metadata_it_cannot_preserve() -> Result<()> {
    for (version, header_length) in [(HeaderVersion::One, 348), (HeaderVersion::Two, 540)] {
        let mut bytes =
            write_single_file_bytes(&header(NiftiDatatype::Uint8, version, 1)?, &[4, 9]);
        bytes[header_length] = 1;
        assert!(matches!(
            read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT),
            Err(NiftiStoredReadError::UnsupportedExtensions)
        ));
    }

    let mut bytes = write_single_file_bytes(
        &header(NiftiDatatype::Uint8, HeaderVersion::One, 1)?,
        &[4, 9],
    );
    bytes[349] = 1;
    assert!(matches!(
        read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT),
        Err(NiftiStoredReadError::InvalidExtensionFlag {
            extension_flag: [0, 1, 0, 0]
        })
    ));
    Ok(())
}

#[test]
fn stored_writer_round_trips_all_samples_in_nifti2_and_gzip() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let calibration = IntensityCalibration::Linear(LinearCalibration::new(2.5, -1024.0)?);
    let metadata = stored_metadata();

    for extension in ["nii", "nii.gz"] {
        for (index, (datatype, samples, payload)) in stored_samples()?.into_iter().enumerate() {
            let path = directory.path().join(format!("stored-{index}.{extension}"));
            let expected_sample_type = samples.sample_type();
            let volume = stored_volume(
                [1, 1, 2],
                samples,
                CoordinateMap::Cartesian,
                calibration.clone(),
            )?;
            write_nifti_stored(&path, &volume)?;

            let bytes = decoded_file(&path)?;
            let header = NiftiHeader::parse(&bytes[..540])?;
            assert_eq!(header.version, HeaderVersion::Two);
            assert_eq!(header.datatype, datatype);
            assert_eq!(header.scl_slope, 2.5);
            assert_eq!(header.scl_inter, -1024.0);
            assert_eq!(&bytes[544..], payload.as_slice());

            let decoded = read_nifti_stored(&path, ImageReadBudget::DEFAULT)?;
            assert_eq!(decoded.shape(), [1, 1, 2]);
            assert_eq!(decoded.samples().sample_type(), expected_sample_type);
            assert_eq!(
                decoded
                    .samples()
                    .encode(ByteOrder::LeastSignificantByteFirst)?,
                payload
            );
            assert_eq!(decoded.metadata().origin(), metadata.origin());
            assert_eq!(decoded.metadata().spacing(), metadata.spacing());
            assert_eq!(decoded.metadata().direction(), metadata.direction());
            assert_eq!(decoded.calibration(), &calibration);
        }
    }
    Ok(())
}

#[test]
fn stored_writer_rejects_unrepresentable_metadata_before_touching_output() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("preserved.nii");
    let original = b"existing output";
    let calibration = IntensityCalibration::Identity;

    let lookup =
        ModalityLookupTable::new(0, vec![1_u16, 2].into_boxed_slice(), LutOutputBits::Sixteen)?;
    let volume = stored_volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![3_u16, 4]),
        CoordinateMap::Cartesian,
        IntensityCalibration::ModalityLookup(lookup),
    )?;
    let new_path = directory.path().join("rejected.nii");
    assert!(matches!(
        write_nifti_stored(&new_path, &volume),
        Err(NiftiStoredWriteError::ModalityLookupCalibration)
    ));
    assert!(!new_path.exists());

    std::fs::write(&path, original)?;
    assert!(matches!(
        write_nifti_stored(&path, &volume),
        Err(NiftiStoredWriteError::ModalityLookupCalibration)
    ));
    assert_eq!(std::fs::read(&path)?, original);

    let first = LinearCalibration::new(1.0, 0.0)?;
    let second = LinearCalibration::new(2.0, 0.0)?;
    let volume = stored_volume(
        [2, 1, 1],
        SampleBuffer::from_samples(vec![3_i16, 4]),
        CoordinateMap::Cartesian,
        IntensityCalibration::PerFrameLinear(vec![first, second].into_boxed_slice()),
    )?;
    assert!(matches!(
        write_nifti_stored(&path, &volume),
        Err(NiftiStoredWriteError::VaryingFrameCalibration)
    ));
    assert_eq!(std::fs::read(&path)?, original);

    let volume = stored_volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![3_i16, 4]),
        CoordinateMap::Cartesian,
        IntensityCalibration::Linear(LinearCalibration::new(0.0, 7.0)?),
    )?;
    assert!(matches!(
        write_nifti_stored(&path, &volume),
        Err(NiftiStoredWriteError::ZeroSlopeCalibration)
    ));
    assert_eq!(std::fs::read(&path)?, original);

    let coordinate_map =
        CoordinateMap::SliceSeries(SliceSeries::try_new(vec![SliceTransform::new(
            Direction::identity(),
            [0.0, 0.0, 0.0],
        )])?);
    let volume = stored_volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![3_i16, 4]),
        coordinate_map,
        calibration,
    )?;
    assert!(matches!(
        write_nifti_stored(&path, &volume),
        Err(NiftiStoredWriteError::UnsupportedCoordinateMap)
    ));
    assert_eq!(std::fs::read(&path)?, original);
    Ok(())
}
