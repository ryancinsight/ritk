//! The stored conversion admits exactly representable header values.
use super::*;

fn series_with_header(spacing: [f64; 3], origin: [f64; 3], slope: f64) -> StoredSeries {
    single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(i16_values()),
        ImageMetadata::new(
            Point::new(origin),
            Spacing::new(spacing),
            file_axes_direction(),
        ),
        IntensityCalibration::Linear(LinearCalibration::new(slope, 0.0).expect("finite slope")),
    ))
}

#[test]
fn stored_header_rejects_rounding_before_writing_either_file() -> anyhow::Result<()> {
    // binary32 has 24 significant bits. This value lies halfway between 1
    // and its successor. Both adjacent binary64 values also round on write.
    let tie = 1.0 + 2_f64.powi(-24);
    let directory = tempdir()?;
    let header_path = directory.path().join("preserved.hdr");
    let image_path = header_path.with_extension("img");
    let original = i16_document(i16_values(), b"original".to_vec());
    write_analyze_stored(&header_path, &original)?;
    let original_header = std::fs::read(&header_path)?;
    let original_image = std::fs::read(&image_path)?;

    for input in [tie.next_down(), tie, tie.next_up()] {
        for (series, field) in [
            (
                series_with_header([1.0, input, 2.0], [0.0; 3], 1.0),
                "spacing",
            ),
            (
                series_with_header([1.0; 3], [0.0, -input, 2.0], 1.0),
                "origin",
            ),
            (series_with_header([1.0; 3], [0.0; 3], -input), "scale"),
        ] {
            let document = AnalyzeStoredSeries {
                series,
                description: b"replacement".to_vec(),
            };
            let error =
                write_analyze_stored(&header_path, &document).expect_err("header would change");
            let AnalyzeStoredSeriesError::Preparation(ConversionPrepareError::Target {
                source: AnalyzeStoredSeriesRejection::HeaderEncoding(cause),
                ..
            }) = error
            else {
                panic!("expected header encoding loss, got {error}");
            };
            assert!(cause.to_string().contains(field), "{cause}");
            assert_eq!(std::fs::read(&header_path)?, original_header);
            assert_eq!(std::fs::read(&image_path)?, original_image);
        }
    }
    Ok(())
}

#[test]
fn exact_header_lattice_preserves_scale_spacing_and_origin() -> anyhow::Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("exact.hdr");
    let step = f64::from(1.0_f32.next_up());
    let spacing = [0.5, 2.0, step];
    let origin = [step * f64::from(i16::MAX), -6.0, 1.0];
    let series = series_with_header(spacing, origin, -step);
    let expected = series.volumes()[0].calibration().clone();
    let document = AnalyzeStoredSeries::from_stored_series("test", series, Vec::new(), [])?;
    write_analyze_stored(&path, &document)?;
    let restored = read_analyze_stored(&path)?;
    let actual = &restored.series().volumes()[0];
    assert_eq!(actual.metadata().spacing().to_array(), spacing);
    assert_eq!(actual.metadata().origin().to_array(), origin);
    assert_eq!(actual.calibration(), &expected);
    Ok(())
}
