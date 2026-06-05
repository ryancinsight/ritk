//! Tests for `scan_dicom_instances` and `scan_dicom_part10_bytes` — in-memory
//! SCP-received instance scanning and Part 10 byte payload scanning.

use arrayvec::ArrayString;

use super::super::scan::{scan_dicom_instances, scan_dicom_part10_bytes};
use crate::format::dicom::networking::scp::StoredInstance;

/// `scan_dicom_instances` must reject an empty slice with a descriptive error.
///
/// Analytical basis: an empty input cannot form a valid DICOM series;
/// the function must fail fast rather than producing a zero-slice series
/// that would later cause undefined behavior in pixel decode.
#[test]
fn test_scan_dicom_instances_empty_input_errors() {
    let result = scan_dicom_instances(&[]);
    let err = result.expect_err("scan_dicom_instances must fail on empty input");
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("no") || msg.contains("zero"),
        "error message must describe the empty-input condition: {msg}"
    );
}

/// `scan_dicom_instances` must reject garbage bytes that do not form valid
/// DICOM Part 10 data after `make_part10_bytes` wrapping.
///
/// Analytical basis: the Part 10 header is constructed correctly by
/// `make_part10_bytes`, but the dataset bytes are raw garbage that cannot
/// be parsed as valid DICOM data elements. The scanner must propagate the
/// parse error rather than silently producing a corrupt series.
#[test]
fn test_scan_dicom_instances_garbage_dataset_errors() {
    let inst = StoredInstance {
        sop_class_uid: ArrayString::from("1.2.840.10008.5.1.4.1.1.2").unwrap(),
        sop_instance_uid: ArrayString::from("1.2.3.4.5.6.7.999").unwrap(),
        dataset_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
        transfer_syntax_uid: ArrayString::from("1.2.840.10008.1.2.1").unwrap(),
    };

    // `scan_dicom_instances` will call `make_part10_bytes` then try to parse.
    // The garbage dataset should cause a parse failure.
    let result = scan_dicom_instances(&[inst]);
    assert!(
        result.is_err(),
        "scan_dicom_instances must fail when dataset bytes are garbage, got: {:?}",
        result.unwrap().metadata.slices.len()
    );
}

/// `scan_dicom_part10_bytes` must reject an empty input with a descriptive error.
///
/// Analytical basis: an empty input cannot form a valid DICOM series;
/// the function must fail fast rather than producing a zero-slice series.
#[test]
fn test_scan_dicom_part10_bytes_empty_input_errors() {
    let result = scan_dicom_part10_bytes(&[]);
    let err = result.expect_err("scan_dicom_part10_bytes must fail on empty input");
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("no") || msg.contains("zero"),
        "error message must describe the empty-input condition: {msg}"
    );
}

/// `scan_dicom_part10_bytes` must reject garbage bytes that do not form
/// valid DICOM Part 10 data.
///
/// Analytical basis: the garbage bytes lack the DICM preamble and cannot
/// be parsed as DICOM data. The scanner must propagate the parse error.
#[test]
fn test_scan_dicom_part10_bytes_garbage_input_errors() {
    let garbage: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF];
    let files: Vec<(&str, &[u8])> = vec![("test.dcm", garbage)];
    let result = scan_dicom_part10_bytes(&files);
    assert!(
        result.is_err(),
        "scan_dicom_part10_bytes must fail when bytes are garbage, got: {:?}",
        result.unwrap().metadata.slices.len()
    );
}

/// `scan_dicom_part10_bytes` must reject all inputs when none of the
/// provided byte payloads are parseable DICOM data.
///
/// Analytical basis: if all inputs fail to parse, the resulting slice
/// vector is empty and the function must return an error.
#[test]
fn test_scan_dicom_part10_bytes_all_unparseable_errors() {
    let garbage1: &[u8] = &[0x00; 4];
    let garbage2: &[u8] = &[0xFF; 8];
    let files: Vec<(&str, &[u8])> = vec![("bad1.dcm", garbage1), ("bad2.dcm", garbage2)];
    let result = scan_dicom_part10_bytes(&files);
    assert!(
        result.is_err(),
        "scan_dicom_part10_bytes must fail when all inputs are unparseable"
    );
}
