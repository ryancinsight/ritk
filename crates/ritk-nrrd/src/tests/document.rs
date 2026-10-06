use crate::{read_nrrd_document, write_nrrd_document, NrrdDocument, NrrdDocumentError};
use anyhow::Result;
use ritk_codecs::{ByteOrder, SampleBuffer};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ImageReadBudget, IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use std::fs;
use tempfile::tempdir;

fn series() -> StoredSeries {
    let volume = StoredVolume::new(
        [1, 1, 3],
        SampleBuffer::from_samples(vec![11_u16, 29, 47]),
        ImageMetadata::new(
            Point::new([10.0, 20.0, 30.0]),
            Spacing::new([0.5, 1.5, 2.0]),
            Direction::identity(),
        ),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid stored volume");
    StoredSeries::new(vec![volume], SeriesAxis::SingleVolume).expect("single-volume series")
}

fn document() -> NrrdDocument {
    NrrdDocument::new(
        series(),
        vec![
            "#retained note".to_owned(),
            "# note".to_owned(),
            "##note".to_owned(),
        ],
        vec![
            ("source".to_owned(), "scanner".to_owned()),
            ("modality".to_owned(), "CT".to_owned()),
            (
                "source:raw_path".to_owned(),
                "line-continuation-test".to_owned(),
            ),
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
    let source = document();
    let source_volume = &source.series().volumes()[0];
    let decoded_volume = &decoded.series().volumes()[0];
    assert_eq!(decoded_volume.shape(), source_volume.shape());
    assert_eq!(decoded_volume.metadata(), source_volume.metadata());
    assert_eq!(
        decoded_volume.coordinate_map(),
        source_volume.coordinate_map()
    );
    assert_eq!(decoded_volume.calibration(), source_volume.calibration());
    assert_eq!(
        decoded_volume.samples().sample_type(),
        source_volume.samples().sample_type()
    );
    assert_eq!(
        decoded_volume.samples().len(),
        source_volume.samples().len()
    );
    assert_eq!(
        decoded_volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [11, 0, 29, 0, 47, 0]
    );
    assert_eq!(
        decoded.comments(),
        &[
            String::from("#retained note"),
            String::from("# note"),
            String::from("##note"),
        ]
    );
    assert_eq!(decoded.records(), source.records());
    let raw = fs::read(&path)?;
    assert!(
        raw.ends_with(&[11, 0, 29, 0, 47, 0]),
        "NRRD payload must end with the raw u16 sample bytes [11,29,47] in LE; got {raw:?}"
    );
    write_nrrd_document(&path, &decoded)?;
    let cycled = read_nrrd_document(&path, ImageReadBudget::DEFAULT)?;
    let cycled_volume = &cycled.series().volumes()[0];
    assert_eq!(cycled_volume.shape(), source_volume.shape());
    assert_eq!(cycled_volume.metadata(), source_volume.metadata());
    assert_eq!(
        cycled_volume.coordinate_map(),
        source_volume.coordinate_map()
    );
    assert_eq!(cycled_volume.calibration(), source_volume.calibration());
    assert_eq!(
        cycled_volume.samples().sample_type(),
        source_volume.samples().sample_type()
    );
    assert_eq!(cycled_volume.samples().len(), source_volume.samples().len());
    assert_eq!(
        cycled_volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [11, 0, 29, 0, 47, 0]
    );
    assert_eq!(cycled.comments(), decoded.comments());
    assert_eq!(cycled.records(), decoded.records());
    fs::write(&path, b"sentinel")?;
    let mut invalid = document();
    invalid.records = vec![("type".to_owned(), "float".to_owned())];
    assert!(matches!(
        write_nrrd_document(&path, &invalid),
        Err(NrrdDocumentError::ConflictingMetadata { .. })
    ));
    assert_eq!(fs::read(&path)?, b"sentinel");
    for key in ["dwmri_NEX", "DWMRI_B-matrix_0", "MoDaLiTy"] {
        let mut invalid = document();
        invalid.records = vec![(key.to_owned(), "1".to_owned())];
        let error = write_nrrd_document(&path, &invalid);
        assert!(matches!(
            error,
            Err(NrrdDocumentError::UnsupportedField { .. })
        ));
        assert_eq!(fs::read(&path)?, b"sentinel");
    }
    for comment in [
        "#",
        "##",
        "# ",
        "##",
        "## ",
        "# Complete NRRD file written by ritk",
    ] {
        let invalid = NrrdDocument::new(series(), vec![comment.to_owned()], Vec::new());
        assert!(matches!(
            invalid,
            Err(NrrdDocumentError::UnsupportedField { .. })
        ));
    }
    assert!(matches!(
        NrrdDocument::new(
            series(),
            Vec::new(),
            vec![("content".into(), "lost".into())]
        ),
        Err(NrrdDocumentError::UnsupportedField { .. })
    ));
    assert!(matches!(
        NrrdDocument::new(
            series(),
            Vec::new(),
            vec![("ritk_coordinate_map".into(), "Cartesian".into())]
        ),
        Err(NrrdDocumentError::UnsupportedField { .. })
    ));
    let input_path = directory.path().join("unsupported-content.nrrd");
    let source_path = directory.path().join("source.nrrd");
    write_nrrd_document(&source_path, &document())?;
    let mut input = fs::read(&source_path)?;
    let separator = input
        .windows(2)
        .position(|window| window == b"\n\n")
        .expect("writer emits a header separator");
    input.splice(
        separator + 1..separator + 1,
        b"content: lost\n".iter().copied(),
    );
    fs::write(&input_path, input)?;
    assert!(matches!(
        read_nrrd_document(&input_path, ImageReadBudget::DEFAULT),
        Err(NrrdDocumentError::UnsupportedField { .. })
    ));
    let malformed_map_path = directory.path().join("malformed-map.nrrd");
    let mut malformed_map = fs::read(&source_path)?;
    let separator = malformed_map
        .windows(2)
        .position(|window| window == b"\n\n")
        .expect("writer emits a header separator");
    malformed_map.splice(
        separator + 1..separator + 1,
        b"ritk_coordinate_map:=cartesian extra=1\n".iter().copied(),
    );
    fs::write(&malformed_map_path, malformed_map)?;
    assert!(matches!(
        read_nrrd_document(&malformed_map_path, ImageReadBudget::DEFAULT),
        Err(NrrdDocumentError::UnsupportedField { .. })
    ));
    let uppercase_map_path = directory.path().join("uppercase-map.nrrd");
    let mut uppercase_map = fs::read(&source_path)?;
    let separator = uppercase_map
        .windows(2)
        .position(|window| window == b"\n\n")
        .expect("writer emits a header separator");
    uppercase_map.splice(
        separator + 1..separator + 1,
        b"RITK_COORDINATE_MAP:=cartesian\n".iter().copied(),
    );
    fs::write(&uppercase_map_path, uppercase_map)?;
    assert!(matches!(
        read_nrrd_document(&uppercase_map_path, ImageReadBudget::DEFAULT),
        Err(NrrdDocumentError::UnsupportedField { field })
            if field == "RITK_COORDINATE_MAP"
    ));
    Ok(())
}
