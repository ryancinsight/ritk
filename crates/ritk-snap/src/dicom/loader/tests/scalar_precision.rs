//! Scalar Part 10 regression tests across file discovery, byte import and display.
use super::fixtures;
use crate::dicom::{load_volume_from_bytes, load_volume_from_series_uid};
use crate::presentation::PresentationFrame;
use crate::render::{NamedColorMap, WindowLevel};
use anyhow::Result;
use ritk_dicom::dataset::{DataElement, DicomFile, PrimitiveValue, VR};
use ritk_dicom::DicomTag;

fn object(
    allocated: u16,
    stored: Option<(u16, u16)>,
    element: u16,
    vr: VR,
    data: Vec<u8>,
) -> Result<DicomFile> {
    let dir = tempfile::tempdir()?;
    let (name, _) = fixtures::write_multiframe(dir.path(), 2, None)?;
    let mut file = DicomFile::open(dir.path().join(name))?;
    let pixel_tag = |e| DicomTag::new(0x0028, e);
    for e in [0x0101, 0x0102, 0x0103, 0x1052, 0x1053] {
        file.remove_element(pixel_tag(e));
    }
    file.remove_element(DicomTag::new(0x7fe0, 0x0010));
    for (e, n) in [(0x0010, 1), (0x0011, 3), (0x0100, allocated)] {
        file.put(DataElement::new(
            pixel_tag(e),
            VR::US,
            PrimitiveValue::from(n),
        ));
    }
    if let Some((bits, sign)) = stored {
        for (e, n) in [(0x0101, bits), (0x0102, bits - 1), (0x0103, sign)] {
            file.put(DataElement::new(
                pixel_tag(e),
                VR::US,
                PrimitiveValue::from(n),
            ));
        }
    }
    file.put(DataElement::new(
        DicomTag::new(0x7fe0, element),
        vr,
        PrimitiveValue::from(data),
    ));
    Ok(file)
}

fn check_both(file: &DicomFile, check: impl Fn(&crate::LoadedVolume)) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("IM000001");
    file.write_to_file(&path)?;
    check(&load_volume_from_series_uid(
        dir.path(),
        fixtures::MULTIFRAME_SERIES_UID,
    )?);
    check(&load_volume_from_bytes("IM000001", &std::fs::read(path)?)?);
    Ok(())
}

#[test]
fn bit_packed_frames_share_a_byte() -> Result<()> {
    let src = object(1, Some((1, 0)), 0x0010, VR::OB, vec![0b1110_1101, 0])?;
    check_both(&src, |v| {
        assert_eq!(v.shape, [2, 1, 3]);
        assert_eq!(v.data.as_slice(), [1., 0., 1., 1., 0., 1.]);
    })
}

#[test]
fn wide_signed_integer_preserves_sign_and_ignores_unused_bits() -> Result<()> {
    let values = [-(1i128 << 68), -1, 0, 1, 17, (1i128 << 68) - 1];
    let raw = values
        .iter()
        .flat_map(|&n| {
            let mut word = n.to_le_bytes();
            word[8] |= 0xe0; // The top three bits of this 72-bit carrier are unused.
            word[..9].to_vec()
        })
        .collect();
    let src = object(72, Some((69, 1)), 0x0010, VR::OW, raw)?;
    check_both(&src, |v| {
        for (i, value) in values.iter().enumerate() {
            assert_eq!(
                v.exact_stored_value([i / 3, 0, i % 3])
                    .expect("source sample")
                    .to_string(),
                value.to_string()
            );
        }
    })
}

#[test]
fn neighbouring_uint64_samples_survive_orthogonal_windowing() -> Result<()> {
    let center = 1u64 << 60;
    let values = [
        center - 1,
        center,
        center + 1,
        center + 1,
        center - 1,
        center,
    ];
    let raw = values.into_iter().flat_map(u64::to_le_bytes).collect();
    let src = object(64, Some((64, 0)), 0x0010, VR::OW, raw)?;
    check_both(&src, |v| {
        assert!(v.data.windows(2).all(|w| w[0] == w[1]));
        for (axis, plane, expected) in [
            (0, 0, vec![0, 255, 255]),
            (1, 0, vec![0, 255, 255, 255, 0, 255]),
            (2, 0, vec![0, 255]),
        ] {
            let frame = PresentationFrame::from_slice(
                v,
                axis,
                plane,
                WindowLevel::new(center as f64, 1.),
                NamedColorMap::Grayscale,
            )
            .expect("render source values");
            assert_eq!(
                frame
                    .rgba()
                    .chunks_exact(4)
                    .map(|p| p[0])
                    .collect::<Vec<_>>(),
                expected
            );
        }
    })
}

#[test]
fn floating_payloads_preserve_finite_values_nan_and_infinities() -> Result<()> {
    let values = [
        16_777_216.0_f64,
        16_777_217.,
        -0.,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for (width, tag, vr, raw) in [
        (
            32,
            8,
            VR::OF,
            values
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect(),
        ),
        (
            64,
            9,
            VR::OD,
            values.into_iter().flat_map(f64::to_le_bytes).collect(),
        ),
    ] {
        let src = object(width, None, tag, vr, raw)?;
        check_both(&src, |v| {
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(v.data[i].to_bits(), (value as f32).to_bits());
            }
            if width == 64 {
                assert_eq!(
                    v.exact_stored_value([0, 0, 1]).expect("double").to_string(),
                    "16777217"
                );
                let frame = PresentationFrame::from_slice(
                    v,
                    0,
                    0,
                    WindowLevel::new(16_777_217., 1.),
                    NamedColorMap::Grayscale,
                )
                .expect("render double");
                assert_eq!((frame.rgba()[0], frame.rgba()[4]), (0, 255));
            }
        })?;
    }
    Ok(())
}

#[test]
fn conflicting_float_metadata_and_truncated_pixels_fail_import() -> Result<()> {
    let mut duplicate = object(32, None, 8, VR::OF, vec![0; 24])?;
    duplicate.put(DataElement::new(
        DicomTag::new(0x7fe0, 0x0010),
        VR::OW,
        PrimitiveValue::from(vec![0u8; 24]),
    ));
    for file in [
        duplicate,
        object(32, Some((32, 1)), 8, VR::OF, vec![0; 24])?,
        object(16, None, 8, VR::OF, vec![0; 24])?,
        object(64, None, 9, VR::OD, vec![0; 32])?,
    ] {
        let mut bytes = Vec::new();
        file.write_all(&mut bytes)?;
        assert!(load_volume_from_bytes("IM000001", &bytes).is_err());
    }
    Ok(())
}

#[test]
fn encapsulated_frames_restart_at_bit_zero_and_ignore_fragment_padding() -> Result<()> {
    use ritk_dicom::dataset::PixelFragments;
    use ritk_dicom::TransferSyntaxKind::{DeflatedImageFrame, EncapsulatedUncompressed};
    // Independent raw RFC1951 streams made by zlib for 0x05 and 0x06.
    for (syntax, fragments) in [
        (EncapsulatedUncompressed, vec![vec![5, 0], vec![6, 0]]),
        (
            DeflatedImageFrame,
            vec![vec![99, 5, 0, 0], vec![99, 3, 0, 0]],
        ),
    ] {
        let mut src = object(1, Some((1, 0)), 0x0010, VR::OB, Vec::new())?;
        src.update_meta(|meta| meta.transfer_syntax = syntax.uid().into());
        src.put(DataElement::new(
            DicomTag::new(0x7fe0, 0x0010),
            VR::OB,
            PixelFragments::from_fragments(fragments),
        ));
        check_both(&src, |v| {
            assert_eq!(v.data.as_slice(), [1., 0., 1., 0., 1., 1.])
        })?;
    }
    Ok(())
}

#[test]
fn verified_header_anonymization_reopens_precise_pixels_and_rejects_short_final_frame() -> Result<()>
{
    use ritk_io::format::dicom::anonymize::{anonymize_dicom_buffers, AnonymizeOptions};
    use std::path::Path;
    let name = Path::new("IM000001");
    let values = [16_777_217.125_f64, -0., 15., 100., 200., 3.25];
    let data = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    for length in [data.len(), data.len() - 8] {
        let mut file = object(64, None, 9, VR::OD, data[..length].to_owned())?;
        file.put(DataElement::new(
            DicomTag::new(0x0010, 0x0010),
            VR::PN,
            PrimitiveValue::from("Fictional^Fixture"),
        ));
        let mut bytes = Vec::new();
        file.write_all(&mut bytes)?;
        let export = anonymize_dicom_buffers(vec![(name, &bytes)], &AnonymizeOptions::default());
        if length < data.len() {
            assert!(export.is_err(), "verification must check the final frame");
        } else {
            let files = export?;
            let volume = load_volume_from_bytes(&files[0].0, &files[0].1)?;
            assert_ne!(volume.patient_name.as_deref(), Some("Fictional^Fixture"));
            for (i, value) in values.iter().enumerate() {
                assert_eq!(
                    volume
                        .exact_stored_value([i / 3, 0, i % 3])
                        .expect("anonymized source sample")
                        .to_string(),
                    value.to_string()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn retired_big_endian_normalizes_by_vr_before_unpacking_samples() -> Result<()> {
    // OW reverses bytes within a word even when 24-bit samples cross words.
    let raw = [
        0x801234u32,
        0x1280ef,
        0xab1200,
        0xffee01,
        0x0102ab,
        0x007755,
    ];
    let pixels = raw
        .iter()
        .flat_map(|v| v.to_le_bytes()[..3].to_owned())
        .collect();
    let mut file = object(24, Some((24, 0)), 0x0010, VR::OW, pixels)?;
    file.update_meta(|m| m.transfer_syntax = "1.2.840.10008.1.2.2".into());
    check_both(&file, |v| {
        assert_eq!(v.data.as_slice(), raw.map(|n| n as f32))
    })?;
    let values = [1e-240, 2., 3.5, 16_777_217.125, -2., f64::MAX];
    let mut double = object(
        64,
        None,
        9,
        VR::OD,
        values.into_iter().flat_map(f64::to_le_bytes).collect(),
    )?;
    double.update_meta(|m| m.transfer_syntax = "1.2.840.10008.1.2.2".into());
    check_both(&double, |v| {
        for (i, n) in values.iter().enumerate() {
            assert_eq!(
                v.exact_stored_value([i / 3, 0, i % 3])
                    .expect("big-endian double")
                    .to_string(),
                n.to_string()
            );
        }
    })
}
