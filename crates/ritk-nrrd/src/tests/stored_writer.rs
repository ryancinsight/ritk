//! Stored NRRD output and conversion preflight contracts.

use super::stored::stored_u64;
use crate::{
    read_nrrd_stored, read_nrrd_stored_series, write_nrrd_stored, write_nrrd_stored_series,
    NrrdStoredWriteError,
};
use anyhow::Result;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ImageReadBudget, IntensityCalibration, LinearCalibration, SeriesAxis, StoredSeries,
    StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, SliceSeries, SliceTransform, Spacing, Vector};
use tempfile::tempdir;

#[test]
fn stored_writer_round_trips_wide_integer_and_float_payload_bits() -> Result<()> {
    let directory = tempdir()?;
    let integer_path = directory.path().join("integer.nrrd");
    let integer = stored_u64(
        vec![u64::MAX, 9_007_199_254_740_993],
        IntensityCalibration::Identity,
    );
    write_nrrd_stored(&integer_path, &integer)?;
    let decoded = read_nrrd_stored(&integer_path, ImageReadBudget::DEFAULT)?;
    assert_eq!(decoded.samples().sample_type(), SampleType::U64);
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        integer
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?
    );

    let float_path = directory.path().join("float.nrrd");
    let float_samples = SampleBuffer::from_samples(vec![
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff8_0000_0000_0042),
    ]);
    let float = StoredVolume::new(
        [1, 1, 2],
        float_samples,
        ImageMetadata::default_for_shape([1, 1, 2]),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid float volume");
    write_nrrd_stored(&float_path, &float)?;
    let decoded_float = read_nrrd_stored(&float_path, ImageReadBudget::DEFAULT)?;
    assert_eq!(decoded_float.samples().sample_type(), SampleType::F64);
    assert_eq!(
        decoded_float
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        float
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?
    );
    Ok(())
}

#[test]
fn stored_writer_round_trips_each_supported_sample_type_and_raw_payload() -> Result<()> {
    let directory = tempdir()?;
    for (sample_type, element_type, samples) in super::stored::sample_cases() {
        let path = directory.path().join(format!("{sample_type:?}.nrrd"));
        let expected_bytes = samples.encode(ByteOrder::LeastSignificantByteFirst)?;
        let shape = [1, 1, samples.len()];
        let volume = StoredVolume::new(
            shape,
            samples,
            ImageMetadata::default_for_shape(shape),
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity,
        )
        .expect("sample case matches its declared shape");

        write_nrrd_stored(&path, &volume)?;
        let encoded_file = std::fs::read(&path)?;
        let header_end = encoded_file
            .windows(2)
            .position(|window| window == b"\n\n")
            .expect("header separator");
        let header = std::str::from_utf8(&encoded_file[..header_end])?;
        assert!(header
            .lines()
            .any(|line| line == format!("type: {element_type}")));
        assert!(header.lines().any(|line| line == "endian: little"));
        assert_eq!(&encoded_file[header_end + 2..], expected_bytes);

        let decoded = read_nrrd_stored(&path, ImageReadBudget::DEFAULT)?;
        assert_eq!(decoded.samples().sample_type(), sample_type);
        assert_eq!(
            decoded
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst)?,
            expected_bytes
        );
    }
    Ok(())
}

#[test]
fn stored_writer_round_trips_finite_spacing_at_extreme_scales() -> Result<()> {
    let directory = tempdir()?;
    let direction = Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    for (name, spacing) in [
        ("small", [1.0e-200, 1.0e200, 1.0]),
        ("large", [1.0e200, 1.0e-200, 1.0]),
    ] {
        let metadata = ImageMetadata::new(
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new(spacing),
            direction,
        );
        let volume = StoredVolume::new(
            [1, 1, 1],
            SampleBuffer::from_samples(vec![17_u8]),
            metadata,
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity,
        )
        .expect("finite axis-aligned geometry");
        let path = directory.path().join(format!("{name}-spacing.nrrd"));
        write_nrrd_stored(&path, &volume)?;
        let decoded = read_nrrd_stored(&path, ImageReadBudget::DEFAULT)?;
        assert_eq!(decoded.metadata().spacing().to_array(), spacing);
        assert_eq!(decoded.metadata().direction(), &direction);
        assert_eq!(
            decoded
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst)?,
            [17]
        );
    }
    Ok(())
}

#[test]
fn stored_series_writer_round_trips_every_supported_sample_type() -> Result<()> {
    let directory = tempdir()?;
    let shape = [1, 1, 2];
    for (sample_type, element_type, first_samples) in super::stored::sample_cases() {
        let first_bytes = first_samples.encode(ByteOrder::LeastSignificantByteFirst)?;
        let sample_width = sample_type.byte_width();
        let second_bytes = first_bytes
            .chunks_exact(sample_width)
            .rev()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let second_samples = SampleBuffer::decode(
            sample_type,
            &second_bytes,
            ByteOrder::LeastSignificantByteFirst,
        )?;
        let volumes = [
            StoredVolume::new(
                shape,
                first_samples,
                ImageMetadata::default_for_shape(shape),
                CoordinateMap::Cartesian,
                IntensityCalibration::Identity,
            )
            .expect("sample case has the declared voxel count"),
            StoredVolume::new(
                shape,
                second_samples,
                ImageMetadata::default_for_shape(shape),
                CoordinateMap::Cartesian,
                IntensityCalibration::Identity,
            )
            .expect("reversed sample case has the declared voxel count"),
        ];
        let path = directory
            .path()
            .join(format!("series-{sample_type:?}.nrrd"));
        let series = StoredSeries::new(volumes.into(), SeriesAxis::List)?;
        write_nrrd_stored_series(&path, &series)?;

        let output = std::fs::read(&path)?;
        let header_end = output
            .windows(2)
            .position(|window| window == b"\n\n")
            .expect("header terminator");
        let header = std::str::from_utf8(&output[..header_end])?;
        assert!(header
            .lines()
            .any(|line| line == format!("type: {element_type}")));
        assert!(header.contains("sizes: 2 1 1 2"));
        assert!(header.contains("kinds: domain domain domain list"));

        let decoded = read_nrrd_stored_series(&path, ImageReadBudget::DEFAULT)?;
        assert_eq!(decoded.volumes().len(), 2);
        assert_eq!(decoded.axis(), &SeriesAxis::List);
        for (actual, expected_bytes) in decoded.volumes().iter().zip([first_bytes, second_bytes]) {
            assert_eq!(actual.shape(), shape);
            assert_eq!(actual.samples().sample_type(), sample_type);
            assert_eq!(actual.metadata().spacing().to_array(), [1.0; 3]);
            assert_eq!(actual.coordinate_map(), &CoordinateMap::Cartesian);
            assert_eq!(actual.calibration(), &IntensityCalibration::Identity);
            assert_eq!(
                actual
                    .samples()
                    .encode(ByteOrder::LeastSignificantByteFirst)?,
                expected_bytes
            );
        }
    }
    Ok(())
}

#[test]
fn diffusion_series_writer_preserves_the_gradient_scheme() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("diffusion.nrrd");
    let scheme = GradientScheme::from_seconds_per_square_millimeter(
        vec![
            (0.0, ritk_spatial::Vector::new([0.0, 0.0, 0.0])),
            (1_000.0, ritk_spatial::Vector::new([1.0, 0.0, 0.0])),
            (1_000.0, ritk_spatial::Vector::new([0.0, 1.0, 0.0])),
        ],
        GradientFrame::Lps,
    )?;
    let volumes = (0_u8..3)
        .map(|sample| {
            StoredVolume::new(
                [1, 1, 1],
                SampleBuffer::from_samples(vec![sample]),
                ImageMetadata::default(),
                CoordinateMap::Cartesian,
                IntensityCalibration::Identity,
            )
            .expect("one-sample volume has valid metadata")
        })
        .collect();
    let series = StoredSeries::new(volumes, SeriesAxis::Diffusion(scheme.clone()))?;

    write_nrrd_stored_series(&path, &series)?;
    let output = std::fs::read_to_string(&path)?;
    assert!(output.starts_with("NRRD0005\n"));
    assert!(output.contains("measurement frame: (1,0,0) (0,1,0) (0,0,1)"));
    assert!(output.contains("modality:=DWMRI"));
    assert!(output.contains("DWMRI_b-value:=1000"));
    let decoded = read_nrrd_stored_series(&path, ImageReadBudget::DEFAULT)?;
    assert_eq!(decoded.axis(), &SeriesAxis::Diffusion(scheme));
    assert_eq!(
        decoded
            .volumes()
            .iter()
            .map(|volume| volume
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        vec![vec![0], vec![1], vec![2]]
    );
    Ok(())
}

#[test]
fn diffusion_series_writer_preserves_nonzero_weighting_below_baseline_threshold() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("low-weighting.nrrd");
    let weighting = DiffusionWeighting::from_seconds_per_square_millimeter(25.0)?;
    let direction = GradientDirection::new(weighting, Vector::new([1.0, 0.0, 0.0]))?;
    let scheme = GradientScheme::new(vec![direction], GradientFrame::Lps)?;
    let volume = StoredVolume::new(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![17_u8]),
        ImageMetadata::default(),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )?;
    let series = StoredSeries::new(vec![volume], SeriesAxis::Diffusion(scheme.clone()))?;

    write_nrrd_stored_series(&path, &series)?;
    let decoded = read_nrrd_stored_series(&path, ImageReadBudget::DEFAULT)?;

    assert_eq!(decoded.axis(), &SeriesAxis::Diffusion(scheme));
    assert_eq!(
        decoded.volumes()[0]
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        vec![17]
    );
    Ok(())
}

#[test]
fn diffusion_header_entry_limit_preserves_existing_output() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("oversized-diffusion.nrrd");
    let original = b"preserve existing output";
    std::fs::write(&path, original)?;

    let count = 65_525_usize;
    let directions = (0..count)
        .map(|_| (0.0, ritk_spatial::Vector::new([0.0; 3])))
        .collect::<Vec<_>>();
    let scheme =
        GradientScheme::from_seconds_per_square_millimeter(directions, GradientFrame::Lps)?;
    let volumes = (0..count)
        .map(|_| {
            StoredVolume::new(
                [1, 1, 1],
                SampleBuffer::from_samples(vec![0_u8]),
                ImageMetadata::default(),
                CoordinateMap::Cartesian,
                IntensityCalibration::Identity,
            )
            .expect("one-sample volume has valid metadata")
        })
        .collect();
    let series = StoredSeries::new(volumes, SeriesAxis::Diffusion(scheme))?;

    let error = write_nrrd_stored_series(&path, &series)
        .expect_err("writer must reject a header its reader cannot parse");
    assert!(matches!(
        error,
        NrrdStoredWriteError::HeaderTooManyEntries {
            entries: 65_538,
            maximum_entries: 65_536,
        }
    ));
    assert_eq!(std::fs::read(&path)?, original);
    Ok(())
}

#[test]
fn stored_writer_rejects_unrepresentable_calibration_before_modifying_output() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("unsupported.nrrd");
    std::fs::write(&path, b"preserve existing output")?;
    let calibration = IntensityCalibration::Linear(
        LinearCalibration::new(2.0, -1024.0).expect("finite calibration"),
    );
    let volume = stored_u64(vec![1, 2], calibration);
    let error = write_nrrd_stored(&path, &volume).expect_err("calibration cannot be represented");
    assert!(matches!(
        error,
        NrrdStoredWriteError::UnsupportedCalibration
    ));
    assert_eq!(std::fs::read(&path)?, b"preserve existing output");
    Ok(())
}

#[test]
fn stored_writer_accepts_value_identity_calibration() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("identity-calibration.nrrd");
    let calibration = IntensityCalibration::Linear(
        LinearCalibration::new(1.0, 0.0).expect("finite identity calibration"),
    );
    let volume = stored_u64(vec![4, u64::MAX], calibration);
    write_nrrd_stored(&path, &volume)?;
    let decoded = read_nrrd_stored(&path, ImageReadBudget::DEFAULT)?;
    assert_eq!(decoded.calibration(), &IntensityCalibration::Identity);
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [4, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255]
    );
    Ok(())
}

#[test]
fn stored_series_writer_rejects_sample_type_mismatch_before_creating_output() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("mismatch.nrrd");
    let first = stored_u64(vec![1, 2], IntensityCalibration::Identity);
    let second = StoredVolume::new(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![1_u32, 2]),
        ImageMetadata::default_for_shape([1, 1, 2]),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid volume");
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List)?;
    let error = write_nrrd_stored_series(&path, &series)
        .expect_err("mixed sample types cannot be represented by one NRRD series");
    assert!(matches!(
        error,
        NrrdStoredWriteError::SampleTypeMismatch { index: 1 }
    ));
    assert!(!path.exists());
    Ok(())
}

#[test]
fn stored_writer_round_trips_per_slice_geometry() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("slice-series.nrrd");
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let coordinate_map = CoordinateMap::SliceSeries(
        SliceSeries::try_new(vec![
            SliceTransform::new(direction, [-12.5, 6.0, 3.0]),
            SliceTransform::new(direction, [-12.5, 6.0, 5.5]),
        ])
        .expect("two-slice sweep"),
    );
    let volume = StoredVolume::new(
        [2, 1, 1],
        SampleBuffer::from_samples(vec![13_i16, -7]),
        ImageMetadata::default_for_shape([2, 1, 1]),
        coordinate_map.clone(),
        IntensityCalibration::Identity,
    )
    .expect("valid slice-series volume");

    write_nrrd_stored(&path, &volume)?;
    let decoded = read_nrrd_stored(&path, ImageReadBudget::DEFAULT)?;
    assert_eq!(decoded.coordinate_map(), &coordinate_map);
    assert_eq!(decoded.shape(), [2, 1, 1]);
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [13, 0, 249, 255]
    );
    Ok(())
}

#[test]
fn stored_writers_reject_oversized_slice_headers_before_truncating_outputs() -> Result<()> {
    let directory = tempdir()?;
    let depth = 20_000;
    let shape = [depth, 1, 1];
    let direction = Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let coordinate_map = CoordinateMap::SliceSeries(
        SliceSeries::try_new(
            (0..depth)
                .map(|_| SliceTransform::new(direction, [f64::MAX; 3]))
                .collect(),
        )
        .expect("finite slice transforms"),
    );
    let make_volume = || {
        StoredVolume::new(
            shape,
            SampleBuffer::from_samples(vec![7_u8; depth]),
            ImageMetadata::default_for_shape(shape),
            coordinate_map.clone(),
            IntensityCalibration::Identity,
        )
        .expect("shape and slice-transform counts match")
    };

    let single_path = directory.path().join("existing-single.nrrd");
    std::fs::write(&single_path, b"preserve single destination")?;
    let single_error = write_nrrd_stored(&single_path, &make_volume())
        .expect_err("reader cannot accept a header beyond its size bound");
    assert!(matches!(
        single_error,
        NrrdStoredWriteError::HeaderTooLarge {
            header_bytes,
            maximum_bytes,
        } if header_bytes > maximum_bytes && maximum_bytes == crate::reader::MAX_HEADER_BYTES
    ));
    assert_eq!(std::fs::read(&single_path)?, b"preserve single destination");

    let series_path = directory.path().join("existing-series.nrrd");
    std::fs::write(&series_path, b"preserve series destination")?;
    let series = StoredSeries::new(vec![make_volume(), make_volume()], SeriesAxis::List)?;
    let series_error = write_nrrd_stored_series(&series_path, &series)
        .expect_err("series header also obeys the reader's size bound");
    assert!(matches!(
        series_error,
        NrrdStoredWriteError::HeaderTooLarge {
            header_bytes,
            maximum_bytes,
        } if header_bytes > maximum_bytes && maximum_bytes == crate::reader::MAX_HEADER_BYTES
    ));
    assert_eq!(std::fs::read(&series_path)?, b"preserve series destination");
    Ok(())
}
