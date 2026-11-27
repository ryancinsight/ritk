//! JPEG 2000 profile import and header anonymization via both AsterSlice entry points.
use super::*;

#[test]
fn jpeg2000_packet_profiles_load_from_extensionless_multiframe_files_and_bytes() -> Result<()> {
    use ritk_dicom::{dataset::PixelFragments, TransferSyntaxKind::*};
    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ritk-codecs/tests/fixtures/jpeg2000/profiles/",
                $name
            ))
            .as_slice()
        };
    }
    for (syntax, source, expected) in [
        (Jpeg2000Lossless, fixture!("classic-ppm.j2c"), None),
        (Jpeg2000Lossy, fixture!("classic-tile-override.j2c"), None),
        (HtJ2kLossless, fixture!("ht-tiles.j2c"), None),
        (HtJ2kLosslessRpcl, fixture!("ht-rpcl-parts.j2c"), None),
        (
            HtJ2k,
            fixture!("ht-irreversible.j2c"),
            Some(fixture!("ht-irreversible.u16le")),
        ),
    ] {
        let mut src = object(16, Some((12, 0)), 0x10, VR::OB, vec![])?;
        for (tag, n) in [(0x10, 33u16), (0x11, 35u16)] {
            src.put(DataElement::new(
                DicomTag::new(0x0028, tag),
                VR::US,
                PrimitiveValue::from(n),
            ));
        }
        src.update_meta(|m| m.transfer_syntax = syntax.uid().into());
        src.put(DataElement::new(
            DicomTag::new(0x7fe0, 0x10),
            VR::OB,
            PixelFragments::from_fragments([source.to_vec(), source.to_vec()]),
        ));
        check_both(&src, |v| {
            assert_eq!(v.shape, [2, 33, 35]);
            for (i, &pixel) in v.data.iter().enumerate() {
                let k = i % 1155;
                let (x, y) = (k % 35, k / 35);
                if let Some(bytes) = expected {
                    let reference = f32::from(u16::from_le_bytes([bytes[2 * k], bytes[2 * k + 1]]));
                    assert!((pixel - reference).abs() <= 1., "{syntax:?} sample {i}");
                } else {
                    assert_eq!(
                        pixel,
                        ((31 * x + 73 * y + 17 * (x ^ y)) % 4096) as f32,
                        "{syntax:?} sample {i}"
                    );
                }
            }
        })?;
        if syntax.htj2k_profile().is_some() {
            use ritk_dicom::EncapsulatedFrameSource;
            let (a, b) = source.split_at(64);
            src.put(DataElement::new(
                DicomTag::new(0x7fe0, 0x10),
                VR::OB,
                PixelFragments::from_fragments([a.to_vec(), b.to_vec(), source.to_vec()]),
            ));
            let err = src
                .encapsulated_frame(0)
                .expect_err("HTJ2K encapsulates each frame in one fragment");
            assert!(err.to_string().contains("requires one fragment per frame"));
        }
    }
    Ok(())
}

#[test]
fn wide_htj2k_survives_header_anonymization_and_extensionless_import() -> Result<()> {
    use ritk_dicom::{dataset::PixelFragments, TransferSyntaxKind};
    use ritk_io::format::dicom::anonymize::{anonymize_dicom_buffers, AnonymizeOptions};
    let mut frame = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ritk-codecs/tests/fixtures/jpeg2000/profiles/ht-deep-advanced.j2c"
    ))
    .to_vec();
    frame[42] = 37;
    let mut file = object(40, Some((38, 0)), 0x10, VR::OB, vec![])?;
    for (tag, n) in [(0x10, 33u16), (0x11, 35u16)] {
        file.put(DataElement::new(
            DicomTag::new(0x0028, tag),
            VR::US,
            PrimitiveValue::from(n),
        ));
    }
    file.put(DataElement::new(
        DicomTag::new(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Fictional^Fixture"),
    ));
    file.update_meta(|m| m.transfer_syntax = TransferSyntaxKind::HtJ2kLossless.uid().into());
    file.put(DataElement::new(
        DicomTag::new(0x7fe0, 0x10),
        VR::OB,
        PixelFragments::from_fragments([frame.clone(), frame]),
    ));
    let check = |v: &crate::LoadedVolume| {
        for f in 0..2 {
            for x in 0..35 {
                let coefficient = (1i64 << 27) + i64::try_from(x).expect("fixture width");
                let expected = (1i64 << 37)
                    + if x % 2 == 0 {
                        coefficient
                    } else {
                        -coefficient
                    };
                assert_eq!(
                    v.exact_stored_value([f, 0, x])
                        .expect("wide HTJ2K source")
                        .to_string(),
                    expected.to_string()
                );
            }
        }
    };
    check_both(&file, check)?;
    let mut bytes = Vec::new();
    file.write_all(&mut bytes)?;
    let output = anonymize_dicom_buffers(
        vec![(std::path::Path::new("IM000001"), &bytes)],
        &AnonymizeOptions::default(),
    )?;
    let v = load_volume_from_bytes(&output[0].0, &output[0].1)?;
    assert_ne!(v.patient_name.as_deref(), Some("Fictional^Fixture"));
    check(&v);
    Ok(())
}
