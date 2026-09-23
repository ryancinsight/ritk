//! Whole documents in the shape of the specification's own examples
//! (sections 14.4 and 14.6), cut down to a few values.

use super::read;
use crate::{ArrayData, IndexingOrder, Intent};

/// The layout of section 14.4: a DOCTYPE, file metadata in CDATA, an empty
/// label table, a point set with a transform, and a triangle array.
const SURFACE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/1594/gifti.dtd">
<GIFTI Version="1.0" NumberOfDataArrays="2">
  <MetaData>
    <MD>
      <Name><![CDATA[date]]></Name>
      <Value><![CDATA[Thu Nov 15 09:05:22 2007]]></Value>
    </MD>
  </MetaData>
  <LabelTable/>
  <DataArray Intent="NIFTI_INTENT_POINTSET" DataType="NIFTI_TYPE_FLOAT32"
      ArrayIndexingOrder="RowMajorOrder" Dimensionality="2" Dim0="3" Dim1="3"
      Encoding="ASCII" Endian="LittleEndian" ExternalFileName="" ExternalFileOffset="">
    <MetaData>
      <MD>
        <Name><![CDATA[AnatomicalStructurePrimary]]></Name>
        <Value><![CDATA[CortexLeft]]></Value>
      </MD>
    </MetaData>
    <CoordinateSystemTransformMatrix>
      <DataSpace><![CDATA[NIFTI_XFORM_TALAIRACH]]></DataSpace>
      <TransformedSpace><![CDATA[NIFTI_XFORM_TALAIRACH]]></TransformedSpace>
      <MatrixData>
        1.000000 0.000000 0.000000 0.000000
        0.000000 1.000000 0.000000 0.000000
        0.000000 0.000000 1.000000 0.000000
        0.000000 0.000000 0.000000 1.000000
      </MatrixData>
    </CoordinateSystemTransformMatrix>
    <Data>
      -16.072010 -66.187515 21.266994
      -16.705893 -66.054337 21.232786
      -17.614349 -65.401642 21.071466
    </Data>
  </DataArray>
  <DataArray Intent="NIFTI_INTENT_TRIANGLE" DataType="NIFTI_TYPE_INT32"
      ArrayIndexingOrder="RowMajorOrder" Dimensionality="2" Dim0="1" Dim1="3"
      Encoding="ASCII" Endian="LittleEndian" ExternalFileName="" ExternalFileOffset="">
    <Data>0 2 1</Data>
  </DataArray>
</GIFTI>
"#;

#[test]
#[expect(
    clippy::excessive_precision,
    reason = "the literals are the document's decimal text, rounded to f32 as the reader rounds it"
)]
fn a_surface_document_reads_to_exactly_its_contents() {
    let image = read(SURFACE).expect("valid document");

    assert_eq!(
        image.metadata().get("date"),
        Some("Thu Nov 15 09:05:22 2007")
    );
    assert!(image.labels().is_empty());
    assert_eq!(image.arrays().len(), 2);

    let points = &image.arrays()[0];
    assert_eq!(points.intent(), &Intent::PointSet);
    assert_eq!(points.dims(), &[3, 3]);
    assert_eq!(points.order(), IndexingOrder::RowMajor);
    assert_eq!(
        points.metadata().get("AnatomicalStructurePrimary"),
        Some("CortexLeft")
    );
    let transform = &points.transforms()[0];
    assert_eq!(transform.data_space, "NIFTI_XFORM_TALAIRACH");
    assert_eq!(transform.transformed_space, "NIFTI_XFORM_TALAIRACH");
    assert_eq!(
        transform.matrix,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    );

    let surface = image.surface().expect("a mesh");
    assert_eq!(
        surface.vertices,
        vec![
            [-16.072_01, -66.187_515, 21.266_994],
            [-16.705_893, -66.054_337, 21.232_786],
            [-17.614_349, -65.401_642, 21.071_466],
        ]
    );
    assert_eq!(surface.triangles, vec![[0, 2, 1]]);
}

/// The layout of section 14.6: a label table with colours, and per-vertex
/// keys; plus the pre-1.0 `Index` attribute the specification says to read as
/// `Key` (section 2.6.3.1).
#[test]
fn a_label_document_reads_its_table_and_keys() {
    let document = r#"<?xml version="1.0" encoding="UTF-8"?>
<GIFTI Version="1.0" NumberOfDataArrays="1">
  <LabelTable>
    <Label Key="0" Red="0.667" Green="0.667" Blue="0.667" Alpha="1.000"><![CDATA[???]]></Label>
    <Label Key="1" Red="1.000" Green="1.000" Blue="0.000" Alpha="1.000"><![CDATA[Positive]]></Label>
    <Label Index="2">Negative &amp; more</Label>
  </LabelTable>
  <DataArray Intent="NIFTI_INTENT_LABEL" DataType="NIFTI_TYPE_INT32"
      ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="10"
      Encoding="ASCII" Endian="LittleEndian">
    <Data>0 2 0 1 1 0 2 0 1 0</Data>
  </DataArray>
</GIFTI>
"#;
    let image = read(document).expect("valid document");

    let labels: Vec<(u32, &str, Option<[f32; 4]>)> = image
        .labels()
        .iter()
        .map(|label| (label.key(), label.name(), label.rgba()))
        .collect();
    assert_eq!(
        labels,
        vec![
            (0, "???", Some([0.667, 0.667, 0.667, 1.0])),
            (1, "Positive", Some([1.0, 1.0, 0.0, 1.0])),
            (2, "Negative & more", None),
        ]
    );
    assert_eq!(image.arrays()[0].intent(), &Intent::Label);
    assert_eq!(
        image.arrays()[0].data(),
        &ArrayData::Int32(vec![0, 2, 0, 1, 1, 0, 2, 0, 1, 0].into())
    );
}

/// Statistical intents outside the named set keep their name.
#[test]
fn an_unnamed_intent_is_kept_by_name() {
    let document = super::one_array(
        r#"Intent="NIFTI_INTENT_TTEST" DataType="NIFTI_TYPE_UINT8" ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="2" Encoding="ASCII""#,
        "7 255",
    );
    let image = read(&document).expect("valid document");
    assert_eq!(
        image.arrays()[0].intent(),
        &Intent::Other("NIFTI_INTENT_TTEST".to_owned())
    );
    assert_eq!(
        image.arrays()[0].data(),
        &ArrayData::UInt8(vec![7, 255].into())
    );
}
