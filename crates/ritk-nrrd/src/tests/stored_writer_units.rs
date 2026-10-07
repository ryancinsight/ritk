use super::stored::stored_u64;
use crate::{
    read_nrrd_stored, read_nrrd_stored_series, write_nrrd_stored, write_nrrd_stored_series,
    NrrdStoredWriteError,
};
use anyhow::Result;
use ritk_image_io::{
    ImageReadBudget, IntensityCalibration, IntensityUnit, SeriesAxis, StoredSeries,
};
use tempfile::tempdir;

#[test]
fn stored_writer_round_trips_sample_units_and_rejects_series_mismatch() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("sample-units.nrrd");
    let volume = stored_u64(vec![1, 2], IntensityCalibration::Identity)
        .with_intensity_unit(IntensityUnit::new("HU:=CT").expect("nonempty intensity unit"));
    write_nrrd_stored(&path, &volume)?;
    let decoded = read_nrrd_stored(&path, ImageReadBudget::DEFAULT)?;
    assert_eq!(decoded.intensity_unit(), volume.intensity_unit());
    let output = std::fs::read(&path)?;
    assert!(output
        .windows(b"sample units: HU:=CT\n".len())
        .any(|window| { window == b"sample units: HU:=CT\n" }));

    let series = StoredSeries::new(vec![volume], SeriesAxis::List)?;
    let series_path = directory.path().join("sample-units-series.nrrd");
    write_nrrd_stored_series(&series_path, &series)?;
    let decoded_series = read_nrrd_stored_series(&series_path, ImageReadBudget::DEFAULT)?;
    assert_eq!(
        decoded_series.volumes()[0].intensity_unit(),
        Some(&IntensityUnit::new("HU:=CT").expect("nonempty unit"))
    );

    let first = stored_u64(vec![1, 2], IntensityCalibration::Identity)
        .with_intensity_unit(IntensityUnit::new("HU").expect("nonempty unit"));
    let second = stored_u64(vec![3, 4], IntensityCalibration::Identity)
        .with_intensity_unit(IntensityUnit::new("counts").expect("nonempty unit"));
    let mismatched = StoredSeries::new(vec![first, second], SeriesAxis::List)?;
    let mismatch_path = directory.path().join("unit-mismatch.nrrd");
    assert!(matches!(
        write_nrrd_stored_series(&mismatch_path, &mismatched),
        Err(NrrdStoredWriteError::IntensityUnitMismatch { index: 1 })
    ));
    assert!(!mismatch_path.exists());

    let invalid_path = directory.path().join("unit-with-edge-space.nrrd");
    std::fs::write(&invalid_path, b"preserve existing output")?;
    let unrepresentable = stored_u64(vec![1], IntensityCalibration::Identity)
        .with_intensity_unit(IntensityUnit::new(" HU ").expect("nonempty unit"));
    assert!(matches!(
        write_nrrd_stored(&invalid_path, &unrepresentable),
        Err(NrrdStoredWriteError::UnsupportedIntensityUnit)
    ));
    assert_eq!(std::fs::read(&invalid_path)?, b"preserve existing output");
    Ok(())
}
