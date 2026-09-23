//! Writing then reading gives back the document, in every encoding.

use crate::{
    ArrayData, CoordinateTransform, DataArray, DataEncoding, GiftiImage, GiftiLabel, IndexingOrder,
    Intent, MetaData,
};

const ENCODINGS: [DataEncoding; 3] = [
    DataEncoding::Ascii,
    DataEncoding::Base64Binary,
    DataEncoding::GZipBase64Binary,
];

/// A document using every feature the writer emits, with a column-major
/// array given in its row-major form so the expected read-back is exact.
fn document(order: IndexingOrder) -> GiftiImage {
    let metadata = MetaData::new([
        (
            "Description".to_owned(),
            "a <tricky> & \"quoted\" 'value'".to_owned(),
        ),
        ("date".to_owned(), "2026-09-23".to_owned()),
    ]);
    let points = DataArray::new(
        Intent::PointSet,
        vec![3, 3],
        ArrayData::Float32(vec![0.1, -2.5, 3.0e-7, 1.0, f32::MAX, -0.0, 7.25, 8.5, 9.75].into()),
    )
    .expect("valid")
    .with_metadata(MetaData::new([(
        "AnatomicalStructurePrimary".to_owned(),
        "CortexLeft".to_owned(),
    )]))
    .with_transforms(vec![CoordinateTransform {
        data_space: "NIFTI_XFORM_UNKNOWN".to_owned(),
        transformed_space: "NIFTI_XFORM_TALAIRACH".to_owned(),
        matrix: [
            [1.0, 0.0, 0.0, -1.5],
            [0.0, 2.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 1.0e-9],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }]);
    let triangles = DataArray::new(
        Intent::Triangle,
        vec![1, 3],
        ArrayData::Int32(vec![0, 1, 2].into()),
    )
    .expect("valid");
    // Row-major [[1, 2, 3], [4, 5, 6]] stored in the requested order.
    let stored = match order {
        IndexingOrder::RowMajor => vec![1, 2, 3, 4, 5, 6],
        IndexingOrder::ColumnMajor => vec![1, 4, 2, 5, 3, 6],
    };
    let matrix = DataArray::with_order(
        Intent::None,
        vec![2, 3],
        order,
        ArrayData::Int32(stored.into()),
    )
    .expect("valid");
    let bytes = DataArray::new(
        Intent::Other("NIFTI_INTENT_RGBA_VECTOR".to_owned()),
        vec![1, 4],
        ArrayData::UInt8(vec![0, 127, 128, 255].into()),
    )
    .expect("valid");
    let labels = vec![
        GiftiLabel::new(0, "???".to_owned(), Some([0.667, 0.667, 0.667, 0.0])).expect("valid"),
        GiftiLabel::new(7, "Left & right".to_owned(), None).expect("valid"),
    ];
    GiftiImage::new(metadata, labels, vec![points, triangles, matrix, bytes]).expect("valid")
}

#[test]
fn every_encoding_round_trips_exactly() {
    let original = document(IndexingOrder::RowMajor);
    for encoding in ENCODINGS {
        let mut xml = Vec::new();
        original.write(&mut xml, encoding).expect("writes");
        let read = GiftiImage::read(xml.as_slice()).expect("reads");
        assert_eq!(read, original, "{encoding:?}");
    }
}

/// The writer emits row-major order, so a column-major array reads back as
/// its row-major equivalent.
#[test]
fn a_column_major_array_is_written_row_major() {
    let mut xml = Vec::new();
    document(IndexingOrder::ColumnMajor)
        .write(&mut xml, DataEncoding::Base64Binary)
        .expect("writes");
    let read = GiftiImage::read(xml.as_slice()).expect("reads");
    assert_eq!(read, document(IndexingOrder::RowMajor));
}
