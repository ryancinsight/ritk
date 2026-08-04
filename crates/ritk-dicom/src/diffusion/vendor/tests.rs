use super::*;
use crate::attribute::{tags, DicomTag};
use crate::diffusion::extract_diffusion_pair;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{DefaultDicomObject, InMemDicomObject};
use ritk_spatial::Vector;

// ── Test helpers ─────────────────────────────────────────────────────────

/// Build a minimal Siemens SV10 CSA binary blob.
///
/// `entries` is a list of `(name, vr, data_bytes)` tuples.  `name` is
/// truncated / zero-padded to 64 bytes.  Each entry produces one item
/// whose length is `data_bytes.len()`.
fn make_sv10_blob(entries: &[(&str, &str, &[u8])]) -> Vec<u8> {
    let n_tags = entries.len() as u32;
    let mut blob: Vec<u8> = Vec::new();

    // Magic + padding.
    blob.extend_from_slice(b"SV10");
    blob.extend_from_slice(&[0u8; 4]);
    blob.extend_from_slice(&n_tags.to_le_bytes());
    blob.extend_from_slice(&77u32.to_le_bytes()); // standard Siemens sentinel

    for (name, vr, data) in entries {
        // Name — 64 bytes, null-padded.
        let mut name_bytes = [0u8; 64];
        let name_len = name.len().min(64);
        name_bytes[..name_len].copy_from_slice(&name.as_bytes()[..name_len]);
        blob.extend_from_slice(&name_bytes);

        // VM — always 1 for our test values.
        blob.extend_from_slice(&1u32.to_le_bytes());

        // VR — 4 bytes, space-padded.
        let mut vr_bytes = [b' '; 4];
        let vr_len = vr.len().min(4);
        vr_bytes[..vr_len].copy_from_slice(&vr.as_bytes()[..vr_len]);
        blob.extend_from_slice(&vr_bytes);

        // SyngoDT — 0 for simple values.
        blob.extend_from_slice(&0u32.to_le_bytes());

        // N_items — 1.
        blob.extend_from_slice(&1u32.to_le_bytes());

        // Padding after tag header.
        blob.extend_from_slice(&[0u8; 4]);

        // Item.
        let item_len = data.len() as u32;
        blob.extend_from_slice(&item_len.to_le_bytes());
        blob.extend_from_slice(data);
        // 4-byte alignment padding.
        let remainder = data.len() % 4;
        if remainder != 0 {
            blob.extend_from_slice(&vec![0u8; 4 - remainder]);
        }
    }

    blob
}

fn object_with_bytes(tag: DicomTag, bytes: Vec<u8>) -> DefaultDicomObject {
    let mut object = InMemDicomObject::new_empty();
    object.put(DataElement::new(
        Tag::from(tag),
        VR::OB,
        PrimitiveValue::U8(SmallVec::from_vec(bytes)),
    ));
    object
        .with_meta(
            dicom::object::meta::FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
                .media_storage_sop_instance_uid("2.25.300")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("synthetic DICOM metadata")
}

fn diffusion_elements(weighting: f64, direction: &[f64]) -> Vec<DataElement<InMemDicomObject>> {
    vec![
        DataElement::new(
            Tag::from(tags::DIFFUSION_B_VALUE),
            VR::FD,
            PrimitiveValue::from(weighting),
        ),
        DataElement::new(
            Tag::from(tags::DIFFUSION_GRADIENT_DIRECTION),
            VR::FD,
            PrimitiveValue::F64(SmallVec::from_vec(direction.to_vec())),
        ),
    ]
}

fn object_with(elements: Vec<DataElement<InMemDicomObject>>) -> DefaultDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for element in elements {
        obj.put(element);
    }
    obj.with_meta(
        dicom::object::meta::FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
            .media_storage_sop_instance_uid("2.25.400")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("synthetic DICOM metadata")
}

// ── CSA binary parser tests ──────────────────────────────────────────────

#[test]
fn csa_parser_extracts_b_value_and_direction_from_sv10_blob() {
    // B_value = 1000.0 (float64)
    let b_value = 1000f64.to_le_bytes();
    // DiffusionGradientDirection = [0.6, 0.0, 0.8] (3 × float64)
    let dir_components: [f64; 3] = [0.6, 0.0, 0.8];
    let mut dir_bytes = Vec::new();
    for c in &dir_components {
        dir_bytes.extend_from_slice(&c.to_le_bytes());
    }

    let blob = make_sv10_blob(&[
        ("B_value", "FD", &b_value),
        ("DiffusionGradientDirection", "FD", &dir_bytes),
    ]);

    let result = parse_csa_blob(&blob);
    assert!(result.is_some(), "must extract B_value and direction from valid SV10 blob");
    let (b, dir) = result.unwrap();
    assert!((b - 1000.0).abs() < 1e-9, "b-value must be 1000, got {b}");
    assert!(
        (dir.to_array()[0] - 0.6).abs() < 1e-9
            && dir.to_array()[1].abs() < 1e-9
            && (dir.to_array()[2] - 0.8).abs() < 1e-9,
        "direction must be [0.6, 0, 0.8], got {:?}",
        dir.to_array()
    );
}

#[test]
fn csa_blob_with_b0_and_no_direction_is_valid_b0() {
    let b_value = 0.0f64.to_le_bytes();
    let blob = make_sv10_blob(&[("B_value", "FD", &b_value)]);

    let result = parse_csa_blob(&blob);
    assert!(result.is_some());
    let (b, dir) = result.unwrap();
    assert_eq!(b, 0.0);
    assert_eq!(dir, Vector::new([0.0, 0.0, 0.0]));
}

#[test]
fn csa_parser_returns_none_for_non_diffusion_blob() {
    let other_value = 42f64.to_le_bytes();
    let blob = make_sv10_blob(&[("SomeOtherKey", "FD", &other_value)]);

    assert!(parse_csa_blob(&blob).is_none());
}

#[test]
fn csa_parser_rejects_non_sv10_magic() {
    let bad_blob = b"NOTSV10\x00\x00\x00\x00\x01\x00\x00\x00M\x00\x00\x00extra data here";
    assert!(parse_csa_blob(bad_blob).is_none());
}

// ── Vendor fallback integration tests ────────────────────────────────────

#[test]
fn vendor_fallback_extracts_from_csa_when_standard_tags_absent() -> anyhow::Result<()> {
    let b_value = 1000f64.to_le_bytes();
    let dir_components: [f64; 3] = [0.6, 0.0, 0.8];
    let mut dir_bytes = Vec::new();
    for c in &dir_components {
        dir_bytes.extend_from_slice(&c.to_le_bytes());
    }

    let csa_blob = make_sv10_blob(&[
        ("B_value", "FD", &b_value),
        ("DiffusionGradientDirection", "FD", &dir_bytes),
    ]);

    // Object with ONLY the CSA header — no standard diffusion tags.
    let object = object_with_bytes(
        crate::diffusion::vendor::SIEMENS_CSA_SERIES,
        csa_blob,
    );

    // extract_diffusion_pair should fall back to vendor extraction.
    let pair = extract_diffusion_pair(&object)?;
    assert!(pair.is_some(), "vendor fallback must find the CSA data");
    let (b, dir) = pair.unwrap();
    assert!((b - 1000.0).abs() < 1e-9);
    assert!(
        (dir.to_array()[0] - 0.6).abs() < 1e-9
            && dir.to_array()[1].abs() < 1e-9
            && (dir.to_array()[2] - 0.8).abs() < 1e-9,
    );
    Ok(())
}

#[test]
fn standard_tags_take_priority_over_vendor_blocks() -> anyhow::Result<()> {
    // Put standard tags at [1.0, 0.0, 0.0], but vendor CSA at [0.6, 0.0, 0.8].
    let mut elements = diffusion_elements(1_000.0, &[1.0, 0.0, 0.0]);

    // Also add a CSA blob with different values.
    let b_value_csa = 500f64.to_le_bytes();
    let dir_components: [f64; 3] = [0.6, 0.0, 0.8];
    let mut dir_bytes = Vec::new();
    for c in &dir_components {
        dir_bytes.extend_from_slice(&c.to_le_bytes());
    }
    let csa_blob = make_sv10_blob(&[
        ("B_value", "FD", &b_value_csa),
        ("DiffusionGradientDirection", "FD", &dir_bytes),
    ]);
    elements.push(DataElement::new(
        Tag::from(crate::diffusion::vendor::SIEMENS_CSA_SERIES),
        VR::OB,
        PrimitiveValue::U8(SmallVec::from_vec(csa_blob)),
    ));

    let object = object_with(elements);

    // Standard tags must win.
    let pair = extract_diffusion_pair(&object)?;
    assert!(pair.is_some());
    let (b, dir) = pair.unwrap();
    assert!((b - 1_000.0).abs() < 1e-9);
    assert_eq!(dir, Vector::new([1.0, 0.0, 0.0]));
    Ok(())
}

#[test]
fn returns_none_when_neither_standard_nor_vendor_metadata_exists() -> anyhow::Result<()> {
    let empty = object_with(vec![]);
    assert_eq!(extract_diffusion_pair(&empty)?, None);
    Ok(())
}
