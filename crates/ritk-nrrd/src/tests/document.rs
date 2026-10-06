use crate::{read_nrrd_document, write_nrrd_document, NrrdDocument};
use anyhow::Result;
use ritk_codecs::SampleBuffer;
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ImageReadBudget, IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume,
};
use ritk_spatial::CoordinateMap;
use std::fs;
use tempfile::tempdir;

fn document() -> NrrdDocument {
    let volume = StoredVolume::new(
        [1, 1, 3],
        SampleBuffer::from_samples(vec![11_u16, 29, 47]),
        ImageMetadata::default(),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid stored volume");
    NrrdDocument::new(
        StoredSeries::new(vec![volume], SeriesAxis::SingleVolume).expect("single-volume series"),
        vec!["# retained note".to_owned()],
        vec![
            ("source".to_owned(), "scanner".to_owned()),
            ("source:raw\\path".to_owned(), "line\nnext".to_owned()),
        ],
    )
    .expect("valid document metadata")
}
#[test]
fn document_round_trip_retains_samples_comments_and_records() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("document.nrrd");
    write_nrrd_document(&path, &document())?;
    let decoded = read_nrrd_document(&path, ImageReadBudget::DEFAULT)?;
    assert_eq!(
        decoded.comments().last().map(String::as_str),
        Some("# retained note")
    );
    assert_eq!(decoded.records(), document().records());
    assert!(fs::read(&path)?.ends_with(&[11, 0, 29, 0, 47, 0]));
    fs::write(&path, b"sentinel")?;
    let mut invalid = document();
    invalid.records = vec![("type".to_owned(), "float".to_owned())];
    let error = write_nrrd_document(&path, &invalid);
    assert!(matches!(error, Err(crate::NrrdDocumentError::ConflictingMetadata { .. })));
    assert_eq!(fs::read(&path)?, b"sentinel");
    Ok(())
}
