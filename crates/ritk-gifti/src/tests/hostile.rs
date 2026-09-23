//! Documents that break the structure, and arbitrary input.

use proptest::prelude::*;

use super::{error_of, one_array, read};
use crate::{
    ArrayData, DataArray, DataEncoding, GiftiError, GiftiImage, GiftiLabel, Intent, MetaData,
};

const SHAPE: &str = r#"Intent="NIFTI_INTENT_SHAPE" DataType="NIFTI_TYPE_FLOAT32" ArrayIndexingOrder="RowMajorOrder" Encoding="ASCII""#;

fn structure_error(document: &str) -> (&'static str, String) {
    match error_of(document) {
        GiftiError::Structure { element, reason } => (element, reason),
        other => panic!("expected a structure error, got {other}"),
    }
}

#[test]
fn a_shape_outside_the_bounds_is_rejected_before_any_data() {
    let too_many_axes = one_array(&format!(r#"{SHAPE} Dimensionality="7" Dim0="1""#), "0");
    assert!(structure_error(&too_many_axes).1.contains("Dimensionality"));

    let overflowing = one_array(
        &format!(r#"{SHAPE} Dimensionality="2" Dim0="4294967296" Dim1="4294967296""#),
        "0",
    );
    assert!(structure_error(&overflowing).1.contains("exceeds"));

    let undeclared_axis = one_array(
        &format!(r#"{SHAPE} Dimensionality="1" Dim0="1" Dim1="1""#),
        "0",
    );
    assert!(structure_error(&undeclared_axis).1.contains("Dim1 beyond"));

    let missing_axis = one_array(&format!(r#"{SHAPE} Dimensionality="2" Dim0="1""#), "0");
    assert!(structure_error(&missing_axis).1.contains("missing Dim1"));
}

#[test]
fn a_miscounted_document_is_rejected() {
    let document = one_array(&format!(r#"{SHAPE} Dimensionality="1" Dim0="1""#), "0")
        .replace(r#"NumberOfDataArrays="1""#, r#"NumberOfDataArrays="2""#);
    assert!(structure_error(&document).1.contains("NumberOfDataArrays"));
}

#[test]
fn unsupported_versions_and_external_data_are_reported() {
    let version_two = one_array(&format!(r#"{SHAPE} Dimensionality="1" Dim0="1""#), "0")
        .replace(r#"Version="1.0""#, r#"Version="2.0""#);
    assert!(matches!(error_of(&version_two), GiftiError::Unsupported(_)));

    let external = one_array(
        r#"Intent="NIFTI_INTENT_SHAPE" DataType="NIFTI_TYPE_FLOAT32" ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="1" Encoding="ExternalFileBinary" Endian="LittleEndian" ExternalFileName="data.bin" ExternalFileOffset="0""#,
        "",
    );
    assert!(matches!(error_of(&external), GiftiError::Unsupported(_)));
}

#[test]
fn misplaced_and_unclosed_elements_are_rejected() {
    let misplaced = r#"<GIFTI Version="1.0" NumberOfDataArrays="0"><Data>1</Data></GIFTI>"#;
    assert_eq!(structure_error(misplaced).0, "Data");

    let unclosed = r#"<GIFTI Version="1.0" NumberOfDataArrays="0"><MetaData>"#;
    assert!(read(unclosed).is_err());

    let wrong_root = r#"<NIFTI Version="1.0"/>"#;
    assert!(structure_error(wrong_root).1.contains("not allowed"));
}

#[test]
fn an_undefined_entity_is_an_xml_error() {
    let document = one_array(
        &format!(r#"{SHAPE} Dimensionality="1" Dim0="1""#),
        "&bogus;",
    );
    assert!(matches!(error_of(&document), GiftiError::Xml { .. }));
}

#[test]
fn invalid_label_tables_are_rejected() {
    let label = |attributes: &str| {
        format!(
            r#"<GIFTI Version="1.0" NumberOfDataArrays="0"><LabelTable>{attributes}</LabelTable></GIFTI>"#
        )
    };
    let out_of_range = label(r#"<Label Key="1" Red="1.5" Green="0" Blue="0" Alpha="1">x</Label>"#);
    assert_eq!(structure_error(&out_of_range).0, "Label");
    let partial = label(r#"<Label Key="1" Red="0.5">x</Label>"#);
    assert!(structure_error(&partial).1.contains("without Green"));
    let repeated = label(r#"<Label Key="3">a</Label><Label Key="3">b</Label>"#);
    assert_eq!(structure_error(&repeated).0, "LabelTable");
    let negative = label(r#"<Label Key="-1">a</Label>"#);
    assert_eq!(structure_error(&negative).0, "Label");
}

#[test]
fn a_surface_with_a_dangling_triangle_is_rejected() {
    let points = DataArray::new(
        Intent::PointSet,
        vec![1, 3],
        ArrayData::Float32(vec![0.0; 3].into()),
    )
    .expect("valid");
    let triangles = DataArray::new(
        Intent::Triangle,
        vec![1, 3],
        ArrayData::Int32(vec![0, 0, 1].into()),
    )
    .expect("valid");
    let image = GiftiImage::new(
        MetaData::default(),
        Vec::new(),
        vec![points.clone(), triangles],
    )
    .expect("valid");
    let error = image.surface().expect_err("vertex 1 does not exist");
    assert!(matches!(error, GiftiError::Structure { .. }), "got {error}");

    let no_topology =
        GiftiImage::new(MetaData::default(), Vec::new(), vec![points]).expect("valid");
    let error = no_topology.surface().expect_err("no triangles");
    assert!(
        error.to_string().contains("NIFTI_INTENT_TRIANGLE"),
        "got {error}"
    );
}

#[test]
fn a_label_colour_outside_the_unit_interval_is_refused_on_construction() {
    let error = GiftiLabel::new(1, "x".to_owned(), Some([0.0, 0.0, -0.1, 1.0]))
        .expect_err("negative component");
    assert!(matches!(
        error,
        GiftiError::Structure {
            element: "Label",
            ..
        }
    ));
}

fn valid_document() -> String {
    let array = DataArray::new(
        Intent::Label,
        vec![4],
        ArrayData::Int32(vec![0, 1, 1, 2].into()),
    )
    .expect("valid");
    let labels =
        vec![GiftiLabel::new(1, "a".to_owned(), Some([1.0, 0.0, 0.0, 1.0])).expect("valid")];
    let image = GiftiImage::new(MetaData::default(), labels, vec![array]).expect("valid");
    let mut xml = Vec::new();
    image
        .write(&mut xml, DataEncoding::GZipBase64Binary)
        .expect("writes");
    String::from_utf8(xml).expect("the writer emits UTF-8")
}

/// Whatever the reader accepts must hold the model's invariants.
fn check(document: &str) {
    if let Ok(image) = read(document) {
        for array in image.arrays() {
            let count: usize = array.dims().iter().product();
            assert_eq!(array.data().len(), count);
        }
    }
}

proptest! {
    #[test]
    fn arbitrary_text_never_panics(text in "\\PC{0,400}") {
        check(&text);
    }

    #[test]
    fn truncations_of_a_valid_document_never_panic(cut in 0_usize..1000) {
        let document = valid_document();
        let cut = cut.min(document.len());
        if document.is_char_boundary(cut) {
            check(&document[..cut]);
            // Anything short of the closing root tag is incomplete.
            if cut < document.trim_end().len() {
                prop_assert!(read(&document[..cut]).is_err());
            }
        }
    }

    #[test]
    fn byte_flips_in_a_valid_document_never_panic(position in 0_usize..1000, byte in 32_u8..127) {
        let mut document = valid_document().into_bytes();
        let position = position % document.len();
        document[position] = byte;
        if let Ok(text) = String::from_utf8(document) {
            check(&text);
        }
    }
}
