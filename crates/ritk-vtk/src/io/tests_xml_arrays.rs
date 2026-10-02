//! Typed `<DataArray>` decoding of the VTK XML readers: each array decodes in
//! its declared `type`, converts to the `f32` the attribute model stores, and a
//! malformed array is refused instead of losing values.

use crate::domain::vtk_data_object::AttributeArray;
use crate::io::image_xml::reader::{parse_vti, read_vti_binary_appended_bytes};
use crate::io::unstructured_xml::reader::parse_vtu;
use anyhow::Result;
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Sample};

/// An ASCII ImageData document of two points holding `arrays`.
fn ascii_document(arrays: &str) -> String {
    format!(
        "<?xml version=\"1.0\"?>\n\
         <VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n\
         <ImageData WholeExtent=\"0 1 0 0 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n\
         <Piece Extent=\"0 1 0 0 0 0\"><PointData>{arrays}</PointData></Piece>\n\
         </ImageData></VTKFile>"
    )
}

fn scalars(document: &str, name: &str) -> Result<Vec<f32>> {
    match parse_vti(document)?.point_data.remove(name) {
        Some(AttributeArray::Scalars { values, .. }) => Ok(values),
        other => anyhow::bail!("expected Scalars '{name}', got {other:?}"),
    }
}

#[test]
fn ascii_arrays_decode_in_their_declared_type() -> Result<()> {
    let document = ascii_document(
        "<DataArray type=\"Int16\" Name=\"a\" format=\"ascii\">-32768 32767</DataArray>\
         <DataArray type=\"UInt8\" Name=\"b\" format=\"ascii\">0 255</DataArray>\
         <DataArray type=\"Float64\" Name=\"c\" format=\"ascii\">0.1 -2.5</DataArray>\
         <DataArray type=\"Int32\" Name=\"d\" format=\"ascii\">16777217 -3</DataArray>",
    );

    assert_eq!(scalars(&document, "a")?, [-32768.0, 32767.0]);
    assert_eq!(scalars(&document, "b")?, [0.0, 255.0]);
    // Cast to the f32 attribute model rounds a value f32 cannot hold.
    assert_eq!(scalars(&document, "c")?, [0.1_f32, -2.5]);
    assert_eq!(scalars(&document, "d")?, [16_777_216.0, -3.0]);
    Ok(())
}

#[test]
fn ascii_array_with_a_malformed_token_is_refused() {
    let document = ascii_document(
        "<DataArray type=\"Float32\" Name=\"a\" format=\"ascii\">1.5 oops</DataArray>",
    );

    let refused = parse_vti(&document).expect_err("a dropped token would shorten the array");

    assert!(
        format!("{refused:#}").contains("bad f32 token 'oops'"),
        "{refused:#}"
    );
}

#[test]
fn ascii_array_out_of_range_for_its_type_is_refused() {
    let document =
        ascii_document("<DataArray type=\"UInt8\" Name=\"a\" format=\"ascii\">1 256</DataArray>");

    let refused = parse_vti(&document).expect_err("256 is not a UInt8");

    assert!(
        format!("{refused:#}").contains("bad u8 token '256'"),
        "{refused:#}"
    );
}

#[test]
fn ascii_array_contradicting_its_tuple_count_is_refused() {
    let document = ascii_document(
        "<DataArray type=\"Float32\" Name=\"a\" NumberOfTuples=\"3\" \
         format=\"ascii\">1 2</DataArray>",
    );

    let refused = parse_vti(&document).expect_err("two values for three tuples");

    assert!(
        format!("{refused:#}").contains("declares 3 tuples of 1 components but holds 2 values"),
        "{refused:#}"
    );
}

#[test]
fn inline_base64_and_unknown_types_are_refused_by_name() {
    let base64 = ascii_document(
        "<DataArray type=\"Float32\" Name=\"a\" format=\"binary\">AAAAAA==</DataArray>",
    );
    let refused = parse_vti(&base64).expect_err("inline base64 is not implemented");
    assert!(
        format!("{refused:#}").contains("format=\"binary\""),
        "{refused:#}"
    );

    let text =
        ascii_document("<DataArray type=\"String\" Name=\"a\" format=\"ascii\">x</DataArray>");
    let refused = parse_vti(&text).expect_err("not a numeric type");
    assert!(
        format!("{refused:#}").contains("unsupported VTK XML DataArray type: String"),
        "{refused:#}"
    );
}

#[test]
fn malformed_geometry_attributes_are_refused() {
    let document = ascii_document("").replace("Spacing=\"1 1 1\"", "Spacing=\"1 1\"");

    let refused = parse_vti(&document).expect_err("spacing has three components");

    assert!(
        format!("{refused:#}").contains("Spacing must hold 3 values, got 2"),
        "{refused:#}"
    );
}

/// An appended ImageData document whose single array `a` of `type_name` holds
/// `payload`, behind a length prefix of `header_type` in `order`.
fn appended_document(
    type_name: &str,
    file_attributes: &str,
    header: &[u8],
    payload: &[u8],
) -> Vec<u8> {
    let mut bytes = format!(
        "<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"0.1\" {file_attributes}>\n\
         <ImageData WholeExtent=\"0 3 0 0 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n\
         <Piece Extent=\"0 3 0 0 0 0\"><PointData>\
         <DataArray type=\"{type_name}\" Name=\"a\" format=\"appended\" offset=\"0\"/>\
         </PointData></Piece></ImageData>\n<AppendedData encoding=\"raw\">\n_"
    )
    .into_bytes();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(b"\n</AppendedData></VTKFile>\n");
    bytes
}

fn appended_values<T: Sample>(type_name: &str, values: [T; 4], expected: [f32; 4]) -> Result<()> {
    let mut payload = Vec::new();
    write_samples(&values, ByteOrder::LittleEndian, &mut payload)?;
    let prefix = u32::try_from(payload.len())?.to_le_bytes();
    let document = appended_document(type_name, "byte_order=\"LittleEndian\"", &prefix, &payload);

    let grid = read_vti_binary_appended_bytes(&document)?;

    assert_eq!(
        grid.point_data.get("a"),
        Some(&AttributeArray::Scalars {
            values: expected.to_vec(),
            num_components: 1
        }),
        "{type_name}"
    );
    Ok(())
}

#[test]
fn appended_arrays_decode_in_every_declared_type() -> Result<()> {
    appended_values("UInt8", [0_u8, 1, 128, 255], [0.0, 1.0, 128.0, 255.0])?;
    appended_values(
        "Int8",
        [i8::MIN, -1, 0, i8::MAX],
        [-128.0, -1.0, 0.0, 127.0],
    )?;
    appended_values(
        "UInt16",
        [0_u16, 1, 256, u16::MAX],
        [0.0, 1.0, 256.0, 65_535.0],
    )?;
    appended_values(
        "Int16",
        [i16::MIN, -1, 0, i16::MAX],
        [-32_768.0, -1.0, 0.0, 32_767.0],
    )?;
    appended_values(
        "UInt32",
        [0_u32, 1, 65_536, 16_777_216],
        [0.0, 1.0, 65_536.0, 16_777_216.0],
    )?;
    appended_values(
        "Int32",
        [i32::MIN, -1, 0, 16_777_217],
        [-2_147_483_648.0, -1.0, 0.0, 16_777_216.0],
    )?;
    appended_values(
        "UInt64",
        [0_u64, 1, 1 << 40, 1 << 62],
        [0.0, 1.0, 1_099_511_627_776.0, 4.611_686e18],
    )?;
    appended_values(
        "Int64",
        [i64::MIN, -1, 0, 1 << 40],
        [-9.223_372e18, -1.0, 0.0, 1_099_511_627_776.0],
    )?;
    appended_values(
        "Float32",
        [0.5_f32, -0.0, 1.0 / 3.0, f32::MAX],
        [0.5, -0.0, 1.0 / 3.0, f32::MAX],
    )?;
    appended_values(
        "Float64",
        [0.1_f64, -0.0, 1.0 / 3.0, 1e10],
        [0.1, -0.0, 1.0 / 3.0, 1e10],
    )
}

#[test]
fn appended_block_honours_big_endian_order_and_a_wide_length_prefix() -> Result<()> {
    let mut payload = Vec::new();
    write_samples(&[258_i16, -2, 0, 7], ByteOrder::BigEndian, &mut payload)?;
    let prefix = u64::try_from(payload.len())?.to_be_bytes();
    let document = appended_document(
        "Int16",
        "byte_order=\"BigEndian\" header_type=\"UInt64\"",
        &prefix,
        &payload,
    );

    let grid = read_vti_binary_appended_bytes(&document)?;

    assert_eq!(
        grid.point_data.get("a"),
        Some(&AttributeArray::Scalars {
            values: vec![258.0, -2.0, 0.0, 7.0],
            num_components: 1
        })
    );
    Ok(())
}

#[test]
fn appended_block_that_is_not_whole_samples_is_refused() {
    let document = appended_document(
        "Int32",
        "byte_order=\"LittleEndian\"",
        &6_u32.to_le_bytes(),
        &[0; 6],
    );

    let refused = read_vti_binary_appended_bytes(&document).expect_err("six bytes of i32");

    assert!(
        format!("{refused:#}").contains("6 bytes is not a whole number of i32 samples"),
        "{refused:#}"
    );
}

/// `document` with `attribute` added to the opening tag of its array `a`.
fn with_array_attribute(document: &[u8], attribute: &str) -> Vec<u8> {
    let needle = b"Name=\"a\"";
    let at = document
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("the document declares array a");
    let mut bytes = document[..at].to_vec();
    bytes.extend_from_slice(attribute.as_bytes());
    bytes.push(b' ');
    bytes.extend_from_slice(&document[at..]);
    bytes
}

#[test]
fn appended_block_contradicting_its_tuple_count_is_refused() -> Result<()> {
    let mut payload = Vec::new();
    write_samples(&[7_i32, 8], ByteOrder::LittleEndian, &mut payload)?;
    let prefix = u32::try_from(payload.len())?.to_le_bytes();
    let short = with_array_attribute(
        &appended_document("Int32", "byte_order=\"LittleEndian\"", &prefix, &payload),
        "NumberOfTuples=\"4\"",
    );

    let refused = read_vti_binary_appended_bytes(&short).expect_err("two values for four tuples");

    let message = format!("{refused:#}");
    assert!(message.contains("DataArray 'a'"), "{message}");
    assert!(message.contains("appended block of 8 bytes"), "{message}");
    assert!(
        message.contains("declares 4 tuples of 1 components but holds 2 values"),
        "{message}"
    );
    Ok(())
}

#[test]
fn appended_block_matching_its_tuple_count_is_read() -> Result<()> {
    let mut payload = Vec::new();
    write_samples(&[7_i32, 8, 9, 10], ByteOrder::LittleEndian, &mut payload)?;
    let prefix = u32::try_from(payload.len())?.to_le_bytes();
    let exact = with_array_attribute(
        &appended_document("Int32", "byte_order=\"LittleEndian\"", &prefix, &payload),
        "NumberOfTuples=\"4\"",
    );

    let grid = read_vti_binary_appended_bytes(&exact)?;

    assert_eq!(
        grid.point_data.get("a"),
        Some(&AttributeArray::Scalars {
            values: vec![7.0, 8.0, 9.0, 10.0],
            num_components: 1
        })
    );
    Ok(())
}

#[test]
fn compressed_and_base64_appended_data_are_refused_by_name() {
    let compressed = appended_document(
        "Float32",
        "byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\"",
        &0_u32.to_le_bytes(),
        &[],
    );
    let refused = read_vti_binary_appended_bytes(&compressed).expect_err("zlib is not implemented");
    assert!(
        format!("{refused:#}").contains("vtkZLibDataCompressor"),
        "{refused:#}"
    );

    let base64 = String::from_utf8(appended_document("Float32", "", &0_u32.to_le_bytes(), &[]))
        .expect("ASCII document")
        .replace("encoding=\"raw\"", "encoding=\"base64\"")
        .into_bytes();
    let refused = read_vti_binary_appended_bytes(&base64).expect_err("base64 is not implemented");
    assert!(
        format!("{refused:#}").contains("encoding=\"base64\""),
        "{refused:#}"
    );
}

#[test]
fn unstructured_grid_points_decode_in_their_declared_type() -> Result<()> {
    let document = "<?xml version=\"1.0\"?>\n\
        <VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n\
        <UnstructuredGrid><Piece NumberOfPoints=\"2\" NumberOfCells=\"1\">\n\
        <Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\
        0.1 0 0  1 2 3</DataArray></Points>\n\
        <Cells><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">0 1</DataArray>\
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">2</DataArray>\
        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">3</DataArray></Cells>\n\
        </Piece></UnstructuredGrid></VTKFile>";

    let grid = parse_vtu(document)?;

    assert_eq!(grid.points, [[0.1_f32, 0.0, 0.0], [1.0, 2.0, 3.0]]);
    assert_eq!(grid.cells, [vec![0_u32, 1]]);
    Ok(())
}
