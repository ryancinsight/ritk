use super::*;

fn put_u16(bytes: &mut Vec<u8>, value: u16, little_endian: bool) {
    let encoded = if little_endian {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    bytes.extend_from_slice(&encoded);
}

fn put_u32(bytes: &mut Vec<u8>, value: u32, little_endian: bool) {
    let encoded = if little_endian {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    bytes.extend_from_slice(&encoded);
}

fn explicit_element(tag: Tag, vr: [u8; 2], value: &[u8], little_endian: bool) -> Vec<u8> {
    let parsed_vr = VR::from_binary(vr).unwrap_or(VR::UN);
    let value_length = u32::try_from(value.len()).expect("test value fits u32");
    let mut bytes = Vec::new();
    put_u16(&mut bytes, tag.group(), little_endian);
    put_u16(&mut bytes, tag.element(), little_endian);
    bytes.extend_from_slice(&vr);
    if uses_short_length(parsed_vr) {
        let short_length = u16::try_from(value_length).expect("test value fits u16");
        put_u16(&mut bytes, short_length, little_endian);
    } else {
        bytes.extend_from_slice(&[0, 0]);
        put_u32(&mut bytes, value_length, little_endian);
    }
    bytes.extend_from_slice(value);
    bytes
}

fn implicit_element(tag: Tag, value: &[u8], little_endian: bool) -> Vec<u8> {
    let value_length = u32::try_from(value.len()).expect("test value fits u32");
    let mut bytes = Vec::new();
    put_u16(&mut bytes, tag.group(), little_endian);
    put_u16(&mut bytes, tag.element(), little_endian);
    put_u32(&mut bytes, value_length, little_endian);
    bytes.extend_from_slice(value);
    bytes
}

fn explicit_sequence(tag: Tag, body: &[u8], length: u32, little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u16(&mut bytes, tag.group(), little_endian);
    put_u16(&mut bytes, tag.element(), little_endian);
    bytes.extend_from_slice(b"SQ");
    bytes.extend_from_slice(&[0, 0]);
    put_u32(&mut bytes, length, little_endian);
    bytes.extend_from_slice(body);
    bytes
}

fn explicit_undefined(tag: Tag, vr: [u8; 2], body: &[u8], little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u16(&mut bytes, tag.group(), little_endian);
    put_u16(&mut bytes, tag.element(), little_endian);
    bytes.extend_from_slice(&vr);
    bytes.extend_from_slice(&[0, 0]);
    put_u32(&mut bytes, UNDEFINED_LENGTH, little_endian);
    bytes.extend_from_slice(body);
    bytes
}

fn implicit_sequence(tag: Tag, body: &[u8], length: u32, little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u16(&mut bytes, tag.group(), little_endian);
    put_u16(&mut bytes, tag.element(), little_endian);
    put_u32(&mut bytes, length, little_endian);
    bytes.extend_from_slice(body);
    bytes
}

fn item(body: &[u8], length: u32, little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u16(&mut bytes, ITEM_GROUP, little_endian);
    put_u16(&mut bytes, ITEM_ELEMENT, little_endian);
    put_u32(&mut bytes, length, little_endian);
    bytes.extend_from_slice(body);
    bytes
}

fn item_delimiter(little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u16(&mut bytes, ITEM_GROUP, little_endian);
    put_u16(&mut bytes, ITEM_DELIMITER_ELEMENT, little_endian);
    put_u32(&mut bytes, 0, little_endian);
    bytes
}

fn sequence_delimiter(little_endian: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u16(&mut bytes, ITEM_GROUP, little_endian);
    put_u16(&mut bytes, SEQUENCE_DELIMITER_ELEMENT, little_endian);
    put_u32(&mut bytes, 0, little_endian);
    bytes
}

fn part10(dataset: &[u8], transfer_syntax: &str) -> Vec<u8> {
    let mut bytes = vec![0; 128];
    bytes.extend_from_slice(DICM_MAGIC);
    let mut transfer_syntax_value = transfer_syntax.as_bytes().to_vec();
    if !transfer_syntax_value.len().is_multiple_of(2) {
        transfer_syntax_value.push(0);
    }
    let transfer_syntax_element =
        explicit_element(TRANSFER_SYNTAX_UID, *b"UI", &transfer_syntax_value, true);
    let group_length =
        u32::try_from(transfer_syntax_element.len()).expect("test file meta fits u32");
    bytes.extend_from_slice(&explicit_element(
        FILE_META_GROUP_LENGTH,
        *b"UL",
        &group_length.to_le_bytes(),
        true,
    ));
    bytes.extend_from_slice(&transfer_syntax_element);
    bytes.extend_from_slice(dataset);
    bytes
}

#[test]
fn validates_explicit_little_endian_dataset() {
    let dataset = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE^JOHN", true);
    let data = part10(&dataset, "1.2.840.10008.1.2.1");

    let summary = validate_part10(&data, &ParseBudget::DEFAULT).expect("valid DICOM");

    assert_eq!(
        summary.transfer_syntax,
        TransferSyntaxKind::ExplicitVrLittleEndian
    );
    assert_eq!(summary.elements, 3);
    assert_eq!(summary.max_depth, 0);
    assert_eq!(summary.encoded_bytes, data.len() - 132);
}

#[test]
fn rejects_declared_value_past_input() {
    let mut dataset = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"AB", true);
    dataset[6..8].copy_from_slice(&4_u16.to_le_bytes());
    let data = part10(&dataset, "1.2.840.10008.1.2.1");

    let error = validate_part10(&data, &ParseBudget::DEFAULT).expect_err("truncated value");

    assert!(error.to_string().contains("exceeds available input"));
}

#[test]
fn rejects_input_over_byte_budget() {
    let dataset = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE", true);
    let data = part10(&dataset, "1.2.840.10008.1.2.1");
    let budget = ParseBudget::new(data.len() - 1, 100, 8);

    let error = validate_part10(&data, &budget).expect_err("byte budget must reject input");

    assert!(error.to_string().contains("input exceeds parse budget"));
}

#[test]
fn rejects_element_count_over_budget() {
    let dataset = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE^JOHN", true);
    let data = part10(&dataset, "1.2.840.10008.1.2.1");
    let budget = ParseBudget::new(data.len(), 2, 8);

    let error = validate_part10(&data, &budget).expect_err("element budget must reject data");

    assert!(error.to_string().contains("count 3 exceeds budget 2"));
}

#[test]
fn rejects_sequence_depth_over_budget() {
    let child = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE", true);
    let item_body = item(
        &child,
        u32::try_from(child.len()).expect("item fits u32"),
        true,
    );
    let sequence = explicit_sequence(
        Tag(0x0008, 0x1110),
        &item_body,
        u32::try_from(item_body.len()).expect("sequence fits u32"),
        true,
    );
    let data = part10(&sequence, "1.2.840.10008.1.2.1");
    let budget = ParseBudget::new(data.len(), 100, 0);

    let error = validate_part10(&data, &budget).expect_err("depth budget must reject sequence");

    assert!(error.to_string().contains("DICOM sequence nesting"));
}

#[test]
fn validates_undefined_sequence_and_item_delimiters() {
    let child = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE", true);
    let mut sequence_body = item(&child, UNDEFINED_LENGTH, true);
    sequence_body.extend_from_slice(&item_delimiter(true));
    sequence_body.extend_from_slice(&sequence_delimiter(true));
    let sequence = explicit_sequence(Tag(0x0008, 0x1110), &sequence_body, UNDEFINED_LENGTH, true);
    let data = part10(&sequence, "1.2.840.10008.1.2.1");

    let summary = validate_part10(&data, &ParseBudget::DEFAULT).expect("valid delimiters");

    assert_eq!(
        summary.transfer_syntax,
        TransferSyntaxKind::ExplicitVrLittleEndian
    );
    assert_eq!(summary.max_depth, 1);
    assert_eq!(summary.elements, 7);
}

#[test]
fn validates_implicit_sequence_from_standard_dictionary() {
    let child = implicit_element(Tag(0x0010, 0x0010), b"DOE", true);
    let item_body = item(
        &child,
        u32::try_from(child.len()).expect("item fits u32"),
        true,
    );
    let sequence = implicit_sequence(
        Tag(0x0008, 0x1110),
        &item_body,
        u32::try_from(item_body.len()).expect("sequence fits u32"),
        true,
    );
    let data = part10(&sequence, "1.2.840.10008.1.2");

    let summary = validate_part10(&data, &ParseBudget::DEFAULT).expect("implicit sequence");

    assert_eq!(
        summary.transfer_syntax,
        TransferSyntaxKind::ImplicitVrLittleEndian
    );
    assert_eq!(summary.max_depth, 1);
}

#[test]
fn validates_big_endian_item_headers() {
    let child = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE", false);
    let item_body = item(
        &child,
        u32::try_from(child.len()).expect("item fits u32"),
        false,
    );
    let sequence = explicit_sequence(
        Tag(0x0008, 0x1110),
        &item_body,
        u32::try_from(item_body.len()).expect("sequence fits u32"),
        false,
    );
    let data = part10(&sequence, "1.2.840.10008.1.2.2");

    let summary = validate_part10(&data, &ParseBudget::DEFAULT).expect("big endian sequence");

    assert_eq!(
        summary.transfer_syntax,
        TransferSyntaxKind::ExplicitVrBigEndian
    );
    assert_eq!(summary.max_depth, 1);
}

#[test]
fn validates_encapsulated_pixel_fragments() {
    let fragment = item(b"pixels", 6, true);
    let mut pixel_value = fragment;
    pixel_value.extend_from_slice(&sequence_delimiter(true));
    let pixel_data = explicit_undefined(PIXEL_DATA, *b"OB", &pixel_value, true);
    let data = part10(&pixel_data, "1.2.840.10008.1.2.4.50");

    let summary = validate_part10(&data, &ParseBudget::DEFAULT).expect("encapsulated pixels");

    assert_eq!(summary.max_depth, 1);
    assert_eq!(summary.transfer_syntax, TransferSyntaxKind::JpegBaseline);
}

#[test]
fn rejects_sequence_with_nonzero_delimiter_length() {
    let child = explicit_element(Tag(0x0010, 0x0010), *b"PN", b"DOE", true);
    let item_body = item(
        &child,
        u32::try_from(child.len()).expect("item fits u32"),
        true,
    );
    let mut sequence_body = item_body;
    let mut delimiter = sequence_delimiter(true);
    delimiter[4..8].copy_from_slice(&1_u32.to_le_bytes());
    sequence_body.extend_from_slice(&delimiter);
    let sequence = explicit_sequence(Tag(0x0008, 0x1110), &sequence_body, UNDEFINED_LENGTH, true);
    let data = part10(&sequence, "1.2.840.10008.1.2.1");

    let error = validate_part10(&data, &ParseBudget::DEFAULT).expect_err("bad delimiter");

    assert!(error
        .to_string()
        .contains("sequence delimiter length must be zero"));
}

#[test]
fn rejects_deflated_dataset_without_decoder() {
    let data = part10(&[], "1.2.840.10008.1.2.1.99");

    let error = validate_part10(&data, &ParseBudget::DEFAULT)
        .expect_err("deflated input needs a bounded decoder");

    assert!(error
        .to_string()
        .contains("requires a bounded deflate adapter"));
}
