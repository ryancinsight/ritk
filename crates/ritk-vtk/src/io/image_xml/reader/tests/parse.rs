#![allow(clippy::needless_range_loop)]

use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use crate::io::image_xml::reader::parse_vti;
use crate::io::image_xml::writer::{write_vti_image_data, write_vti_str};
use tempfile::NamedTempFile;

/// Build a minimal valid VTI XML string with the given extent, origin, spacing,
/// and optional inline PointData/CellData blocks.
fn make_vti(
    extent: &str,
    origin: &str,
    spacing: &str,
    point_data_block: &str,
    cell_data_block: &str,
) -> String {
    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    s.push_str(&format!(
        "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">\n",
        extent, origin, spacing
    ));
    s.push_str(&format!("    <Piece Extent=\"{}\">\n", extent));
    s.push_str(point_data_block);
    s.push_str(cell_data_block);
    s.push_str("    </Piece>\n");
    s.push_str("  </ImageData>\n");
    s.push_str("</VTKFile>\n");
    s
}

#[test]
fn test_parse_vti_whole_extent() {
    let vti = make_vti("0 3 0 4 0 5", "0 0 0", "1 1 1", "", "");
    let img = parse_vti(&vti).expect("parse must succeed");
    assert_eq!(img.whole_extent, [0, 3, 0, 4, 0, 5]);
}

#[test]
fn test_parse_vti_origin_and_spacing() {
    let vti = make_vti("0 1 0 1 0 1", "1.5 2.5 3.5", "0.25 0.5 0.75", "", "");
    let img = parse_vti(&vti).expect("parse must succeed");
    assert!(
        (img.origin[0] - 1.5).abs() < 1e-9,
        "origin[0] = {} expected 1.5",
        img.origin[0]
    );
    assert!(
        (img.origin[1] - 2.5).abs() < 1e-9,
        "origin[1] = {} expected 2.5",
        img.origin[1]
    );
    assert!(
        (img.origin[2] - 3.5).abs() < 1e-9,
        "origin[2] = {} expected 3.5",
        img.origin[2]
    );
    assert!(
        (img.spacing[0] - 0.25).abs() < 1e-9,
        "spacing[0] = {} expected 0.25",
        img.spacing[0]
    );
    assert!(
        (img.spacing[1] - 0.5).abs() < 1e-9,
        "spacing[1] = {} expected 0.5",
        img.spacing[1]
    );
    assert!(
        (img.spacing[2] - 0.75).abs() < 1e-9,
        "spacing[2] = {} expected 0.75",
        img.spacing[2]
    );
}

#[test]
fn test_parse_vti_point_data_scalars() {
    // extent "0 1 0 1 0 1" â†’ n_points = 2*2*2 = 8
    let pd = concat!(
        "      <PointData>\n",
        "        <DataArray type=\"Float32\" Name=\"intensity\"",
        " NumberOfComponents=\"1\" format=\"ascii\">\n",
        "          1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n",
        "        </DataArray>\n",
        "      </PointData>\n"
    );
    let vti = make_vti("0 1 0 1 0 1", "0 0 0", "1 1 1", pd, "");
    let img = parse_vti(&vti).expect("parse must succeed");
    assert_eq!(img.point_data.len(), 1, "one point array expected");
    match img
        .point_data
        .get("intensity")
        .expect("intensity must be present")
    {
        AttributeArray::Scalars {
            values,
            num_components } => {
            assert_eq!(*num_components, 1);
            assert_eq!(values.len(), 8);
            assert!(
                (values[0] - 1.0f32).abs() < 1e-5,
                "values[0] = {} expected 1.0",
                values[0]
            );
            assert!(
                (values[7] - 8.0f32).abs() < 1e-5,
                "values[7] = {} expected 8.0",
                values[7]
            );
        }
        other => panic!("expected Scalars, got {:?}", other) }
}

#[test]
fn test_parse_vti_multicomponent_vectors() {
    // extent "0 0 0 0 0 0" â†’ n_points = 1; 3-component DataArray â†’ Vectors
    let pd = concat!(
        "      <PointData>\n",
        "        <DataArray type=\"Float32\" Name=\"velocity\"",
        " NumberOfComponents=\"3\" format=\"ascii\">\n",
        "          1.0 2.0 3.0\n",
        "        </DataArray>\n",
        "      </PointData>\n"
    );
    let vti = make_vti("0 0 0 0 0 0", "0 0 0", "1 1 1", pd, "");
    let img = parse_vti(&vti).expect("parse must succeed");
    match img
        .point_data
        .get("velocity")
        .expect("velocity must be present")
    {
        AttributeArray::Vectors { values } => {
            assert_eq!(values.len(), 1, "one vector for one point");
            assert!(
                (values[0][0] - 1.0f32).abs() < 1e-5,
                "vx = {} expected 1.0",
                values[0][0]
            );
            assert!(
                (values[0][1] - 2.0f32).abs() < 1e-5,
                "vy = {} expected 2.0",
                values[0][1]
            );
            assert!(
                (values[0][2] - 3.0f32).abs() < 1e-5,
                "vz = {} expected 3.0",
                values[0][2]
            );
        }
        other => panic!("expected Vectors, got {:?}", other) }
}

#[test]
fn test_parse_vti_cell_data() {
    // extent "0 1 0 1 0 1" â†’ n_cells = 1*1*1 = 1
    let cd = concat!(
        "      <CellData>\n",
        "        <DataArray type=\"Float32\" Name=\"pressure\"",
        " NumberOfComponents=\"1\" format=\"ascii\">\n",
        "          42.0\n",
        "        </DataArray>\n",
        "      </CellData>\n"
    );
    let vti = make_vti("0 1 0 1 0 1", "0 0 0", "1 1 1", "", cd);
    let img = parse_vti(&vti).expect("parse must succeed");
    assert_eq!(img.cell_data.len(), 1, "one cell array expected");
    match img
        .cell_data
        .get("pressure")
        .expect("pressure must be present")
    {
        AttributeArray::Scalars {
            values,
            num_components } => {
            assert_eq!(*num_components, 1);
            assert_eq!(values.len(), 1);
            assert!(
                (values[0] - 42.0f32).abs() < 1e-5,
                "values[0] = {} expected 42.0",
                values[0]
            );
        }
        other => panic!("expected Scalars, got {:?}", other) }
}

#[test]
fn test_parse_vti_empty_point_data() {
    // No PointData block; point_data map must be empty.
    let vti = make_vti("0 2 0 2 0 2", "0 0 0", "1 1 1", "", "");
    let img = parse_vti(&vti).expect("parse must succeed");
    assert!(
        img.point_data.is_empty(),
        "point_data must be empty when PointData section is absent"
    );
    assert!(
        img.cell_data.is_empty(),
        "cell_data must be empty when CellData section is absent"
    );
}

#[test]
fn test_read_vti_file_roundtrip() {
    // Write via write_vti_str, parse with parse_vti, verify full round-trip.
    // extent [0,1,0,1,0,1] â†’ n_points = 8
    let mut img = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        origin: [1.0, 2.0, 3.0],
        spacing: [0.5, 0.5, 0.5],
        ..Default::default()
    };
    img.point_data.insert(
        "scalars".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            num_components: 1 },
    );
    let xml = write_vti_str(&img);
    let parsed = parse_vti(&xml).expect("round-trip parse must succeed");

    assert_eq!(parsed.whole_extent, img.whole_extent, "extent mismatch");
    for i in 0..3 {
        assert!(
            (parsed.origin[i] - img.origin[i]).abs() < 1e-5,
            "origin[{i}] mismatch: {} vs {}",
            parsed.origin[i],
            img.origin[i]
        );
        assert!(
            (parsed.spacing[i] - img.spacing[i]).abs() < 1e-5,
            "spacing[{i}] mismatch: {} vs {}",
            parsed.spacing[i],
            img.spacing[i]
        );
    }
    match parsed
        .point_data
        .get("scalars")
        .expect("scalars must be present")
    {
        AttributeArray::Scalars {
            values,
            num_components } => {
            assert_eq!(*num_components, 1);
            assert_eq!(values.len(), 8);
            let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "scalars[{i}] = {got} expected {exp}"
                );
            }
        }
        other => panic!("expected Scalars, got {:?}", other) }
}

#[test]
fn test_from_file_roundtrip() {
    // Write to NamedTempFile via write_vti_image_data, read back via read_vti_image_data.
    use crate::io::image_xml::reader::read_vti_image_data;
    let mut img = VtkImageData {
        whole_extent: [0, 2, 0, 3, 0, 1],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        ..Default::default()
    };
    // n_points = (2+1)*(3+1)*(1+1) = 3*4*2 = 24
    img.point_data.insert(
        "density".to_string(),
        AttributeArray::Scalars {
            values: (0..24).map(|i| i as f32 * 0.5).collect(),
            num_components: 1 },
    );

    let tmp = NamedTempFile::new().expect("temp file creation must succeed");
    write_vti_image_data(tmp.path(), &img).expect("write must succeed");
    let loaded = read_vti_image_data(tmp.path()).expect("read must succeed");

    assert_eq!(loaded.whole_extent, img.whole_extent, "extent round-trip");
    let loaded_vals = match loaded
        .point_data
        .get("density")
        .expect("density must exist")
    {
        AttributeArray::Scalars { values, .. } => values.clone(),
        other => panic!("expected Scalars, got {:?}", other) };
    assert_eq!(loaded_vals.len(), 24, "24 scalar values expected");
    for i in 0..24 {
        let exp = i as f32 * 0.5;
        assert!(
            (loaded_vals[i] - exp).abs() < 1e-5,
            "density[{i}] = {} expected {exp}",
            loaded_vals[i]
        );
    }
}

#[test]
fn test_missing_piece_tag_error() {
    // Valid ImageData tag but no Piece element â€” parse must return Err.
    let vti = concat!(
        "<?xml version=\"1.0\"?>\n",
        "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n",
        "  <ImageData WholeExtent=\"0 1 0 1 0 1\"",
        " Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",
        "  </ImageData>\n",
        "</VTKFile>\n"
    );
    let result = parse_vti(vti);
    assert!(result.is_err(), "missing Piece tag must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Piece"),
        "error message must mention Piece, got: {msg}"
    );
}

#[test]
fn test_nonexistent_file_error() {
    use crate::io::image_xml::reader::read_vti_image_data;
    let result = read_vti_image_data("/nonexistent/path/that/does/not/exist.vti");
    assert!(result.is_err(), "nonexistent file must return Err");
}
