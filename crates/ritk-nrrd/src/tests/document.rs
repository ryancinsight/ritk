use crate::document::read_nrrd_document_from_session;
use crate::reader::{NrrdHeaderError, NrrdReadSession, MAX_HEADER_ENTRIES};
use crate::{read_nrrd_document, write_nrrd_document, NrrdDocument, NrrdDocumentError};
use anyhow::Result;
use ritk_codecs::{ByteOrder, SampleBuffer};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ImageReadBudget, IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use std::fs;
use std::io::Write;
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
    for metadata in [
        b"DWMRI_B-VALUE:=1000\n".as_slice(),
        b"DWMRI_b-value:=1000\nDWMRI_b-value:=1000\n".as_slice(),
        b"RITK_COORDINATE_MAP:=cartesian\n".as_slice(),
        b"ritk_coordinate_map:=cartesian\nritk_coordinate_map:=cartesian\n".as_slice(),
        b"MoDaLiTy:=DWMRI\n".as_slice(),
        b"modality:=DWMRI\nmodality:=DWMRI\n".as_slice(),
        b"modality:=DWMRI\nmodality:=CT\n".as_slice(),
        b"DWMRI_GRADIENT_0000:=0 0 0\n".as_slice(),
        b"DWMRI_gradient_0000:=0 0 0\nDWMRI_gradient_0000:=0 0 0\n".as_slice(),
    ] {
        let diffusion_path = directory.path().join("invalid-diffusion.nrrd");
        let mut diffusion = fs::read(&source_path)?;
        let separator = diffusion
            .windows(2)
            .position(|window| window == b"\n\n")
            .expect("writer emits a header separator");
        diffusion.splice(separator + 1..separator + 1, metadata.iter().copied());
        fs::write(&diffusion_path, diffusion)?;
        assert!(matches!(
            read_nrrd_document(&diffusion_path, ImageReadBudget::DEFAULT),
            Err(NrrdDocumentError::UnsupportedField { .. })
        ));
    }
    Ok(())
}

#[test]
fn document_header_and_payload_stay_bound_to_one_open_source() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("document.nrrd");
    let preserved_path = directory.path().join("preserved.nrrd");
    let original = document();
    write_nrrd_document(&path, &original)?;
    let session = NrrdReadSession::open(&path)?;

    let replacement_volume = StoredVolume::new(
        [1, 1, 3],
        SampleBuffer::from_samples(vec![2_u16, 3, 5]),
        ImageMetadata::new(
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([2.0, 3.0, 4.0]),
            Direction::identity(),
        ),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid replacement volume");
    let replacement = NrrdDocument::new(
        StoredSeries::new(vec![replacement_volume], SeriesAxis::SingleVolume)
            .expect("single-volume replacement series"),
        vec!["#replacement source".to_owned()],
        vec![("source".to_owned(), "replacement".to_owned())],
    )
    .expect("valid replacement document");
    fs::rename(&path, &preserved_path)?;
    write_nrrd_document(&path, &replacement)?;

    let replacement_read = read_nrrd_document(&path, ImageReadBudget::DEFAULT)?;
    assert_eq!(replacement_read.comments(), replacement.comments());
    assert_eq!(replacement_read.records(), replacement.records());
    assert_eq!(
        replacement_read.series().volumes()[0]
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [2, 0, 3, 0, 5, 0]
    );

    let decoded = read_nrrd_document_from_session(session, ImageReadBudget::DEFAULT)?;
    let original_volume = &original.series().volumes()[0];
    let decoded_volume = &decoded.series().volumes()[0];
    assert_eq!(decoded.comments(), original.comments());
    assert_eq!(decoded.records(), original.records());
    assert_eq!(decoded_volume.shape(), original_volume.shape());
    assert_eq!(decoded_volume.metadata(), original_volume.metadata());
    assert_eq!(
        decoded_volume.coordinate_map(),
        original_volume.coordinate_map()
    );
    assert_eq!(
        decoded_volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [11, 0, 29, 0, 47, 0]
    );
    Ok(())
}

#[test]
fn document_metadata_validation_precedes_payload_decode() -> Result<()> {
    let directory = tempdir()?;
    let source_path = directory.path().join("source.nrrd");
    write_nrrd_document(&source_path, &document())?;
    let source = fs::read(&source_path)?;
    let metadata_cases: [(&[u8], &str); 2] = [
        (b"note:=line\\nnext\n", "metadata"),
        (b"MoDaLiTy:=CT\n", "MoDaLiTy"),
    ];

    for (index, (metadata, expected_field)) in metadata_cases.into_iter().enumerate() {
        let mut input = source.clone();
        let separator = input
            .windows(2)
            .position(|window| window == b"\n\n")
            .expect("writer emits a header separator");
        input.splice(separator + 1..separator + 1, metadata.iter().copied());
        input.truncate(separator + metadata.len() + 2);
        let path = directory
            .path()
            .join(format!("invalid-metadata-{index}.nrrd"));
        fs::write(&path, input)?;

        assert!(matches!(
            read_nrrd_document(&path, ImageReadBudget::DEFAULT),
            Err(NrrdDocumentError::UnsupportedField { field }) if field == expected_field
        ));
    }
    Ok(())
}

#[test]
fn canonical_header_entry_limit_precedes_payload_decode() -> Result<()> {
    let directory = tempdir()?;
    let source_path = directory.path().join("source.nrrd");
    write_nrrd_document(&source_path, &document())?;
    let source = fs::read(&source_path)?;
    let separator = source
        .windows(2)
        .position(|window| window == b"\n\n")
        .expect("writer emits a header separator");
    let header = std::str::from_utf8(&source[..separator])?;
    let lines = header
        .lines()
        .filter(|line| *line != "kinds: domain domain domain")
        .collect::<Vec<_>>();
    let entry_count = lines.len().checked_sub(1).expect("magic line exists");
    let additional_records = MAX_HEADER_ENTRIES
        .checked_sub(entry_count)
        .expect("source header is below its entry limit");
    let mut input = lines.join("\n").into_bytes();
    input.push(b'\n');
    for index in 0..additional_records {
        writeln!(&mut input, "padding_{index}:=x")?;
    }
    input.push(b'\n');
    let path = directory.path().join("entry-limit-before-payload.nrrd");
    fs::write(&path, input)?;

    assert!(matches!(
        read_nrrd_document(&path, ImageReadBudget::DEFAULT),
        Err(NrrdDocumentError::Header(NrrdHeaderError::TooManyEntries {
            maximum_entries
        })) if maximum_entries == MAX_HEADER_ENTRIES
    ));
    Ok(())
}
