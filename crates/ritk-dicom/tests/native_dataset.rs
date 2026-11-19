use std::borrow::Cow;

use bytes::Bytes;
use ritk_dicom::dataset::{DataElement, Dataset, DicomFile, FileMetaBuilder, PixelFragments, VR};
use ritk_dicom::{
    decode_frame_with, tags, DecodeFrameRequest, DicomTag, EncapsulatedFrameSource, NativeBackend,
    ParseBudget, PixelLayout, PixelSignedness, TransferSyntaxKind,
};
use tracing_subscriber::fmt::format::FmtSpan;

const SOP_CLASS: &str = "1.2.840.10008.5.1.4.1.1.2";
const SOP_INSTANCE: &str = "1.2.826.0.1.3680043.10.999.41";
const PIXEL_DATA: DicomTag = DicomTag::new(0x7fe0, 0x0010);
const PATIENT: DicomTag = DicomTag::new(0x0010, 0x0010);

fn trace_test() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

fn explicit_ui(meta: &mut Vec<u8>, element: u16, text: &str) {
    let aligned = text.len() + text.len() % 2;
    meta.extend_from_slice(&2u16.to_le_bytes());
    meta.extend_from_slice(&element.to_le_bytes());
    meta.extend_from_slice(b"UI");
    meta.extend_from_slice(&u16::try_from(aligned).unwrap().to_le_bytes());
    meta.extend_from_slice(text.as_bytes());
    if !text.len().is_multiple_of(2) {
        meta.push(0);
    }
}

fn part10_with_data(syntax: &str, data: &[u8]) -> Bytes {
    let mut meta = Vec::new();
    explicit_ui(&mut meta, 2, SOP_CLASS);
    explicit_ui(&mut meta, 3, SOP_INSTANCE);
    explicit_ui(&mut meta, 0x10, syntax);
    explicit_ui(&mut meta, 0x12, SOP_INSTANCE);
    let mut bytes = vec![0u8; 128];
    bytes.extend_from_slice(b"DICM\x02\x00\x00\x00UL\x04\x00");
    bytes.extend_from_slice(&u32::try_from(meta.len()).unwrap().to_le_bytes());
    bytes.extend(meta);
    bytes.extend_from_slice(data);
    bytes.into()
}

fn implicit_header(data: &mut Vec<u8>, group: u16, element: u16, length: u32) {
    data.extend_from_slice(&group.to_le_bytes());
    data.extend_from_slice(&element.to_le_bytes());
    data.extend_from_slice(&length.to_le_bytes());
}

#[test]
fn implicit_registry_resolves_defined_items_and_signed_padding() {
    let _trace = trace_test();
    let bytes = tracing::info_span!("arrange").in_scope(|| {
        let mut body = Vec::new();
        implicit_header(&mut body, 0x20, 0x9113, 22);
        implicit_header(&mut body, 0xfffe, 0xe000, 14);
        implicit_header(&mut body, 0x20, 0x0032, 6);
        body.extend_from_slice(b"1\\2\\3 ");
        implicit_header(&mut body, 0x28, 0x0103, 2);
        body.extend_from_slice(&1u16.to_le_bytes());
        implicit_header(&mut body, 0x28, 0x0120, 2);
        body.extend_from_slice(&(-7i16).to_le_bytes());
        part10_with_data("1.2.840.10008.1.2", &body)
    });

    let (object, emitted) = tracing::info_span!("act").in_scope(|| {
        let object = DicomFile::parse(bytes.clone(), &ParseBudget::DEFAULT).unwrap();
        let mut output = Vec::new();
        object
            .write_dataset(&mut output, &TransferSyntaxKind::ImplicitVrLittleEndian)
            .unwrap();
        (object, output)
    });

    tracing::info_span!("assert").in_scope(|| {
        let plane = object.element(DicomTag::new(0x20, 0x9113)).unwrap();
        let ritk_dicom::dataset::Value::Sequence(sequence) = plane.value() else {
            panic!("the PS3.6 plane position entry is a sequence");
        };
        assert_eq!(
            sequence.items()[0]
                .element(tags::IMAGE_POSITION_PATIENT)
                .unwrap()
                .value()
                .to_numbers::<f64>()
                .unwrap(),
            [1.0, 2.0, 3.0]
        );
        let padding = object.element(DicomTag::new(0x28, 0x0120)).unwrap();
        assert_eq!(
            (padding.vr(), padding.to_num::<i16>().unwrap()),
            (VR::SS, -7)
        );
        assert_eq!(emitted, bytes[bytes.len() - emitted.len()..]);
    });
}

fn file(syntax: TransferSyntaxKind, dataset: Dataset) -> DicomFile {
    dataset
        .with_meta(
            FileMetaBuilder::new()
                .transfer_syntax(syntax.uid())
                .media_storage_sop_class_uid(SOP_CLASS)
                .media_storage_sop_instance_uid(SOP_INSTANCE),
        )
        .unwrap()
}

#[test]
fn declared_repertoires_round_trip_through_deflate_and_its_size_limit() {
    let _trace = trace_test();
    let (input, expected) = tracing::info_span!("arrange").in_scope(|| {
        let expected = "Müller^André";
        let mut dataset = Dataset::new();
        dataset.put_value(
            tags::SPECIFIC_CHARACTER_SET,
            VR::CS,
            "ISO 2022 IR 6\\ISO 2022 IR 100",
        );
        dataset.put_value(PATIENT, VR::PN, expected);
        dataset.put_value(DicomTag::new(0x0040, 0xa160), VR::UT, "report ".repeat(300));
        let mut bytes = Vec::new();
        file(TransferSyntaxKind::DeflatedExplicitVrLittleEndian, dataset)
            .write_all(&mut bytes)
            .unwrap();
        (Bytes::from(bytes), expected)
    });
    let (valid, refusal) = tracing::info_span!("act").in_scope(|| {
        let limited = ParseBudget {
            max_alloc_bytes: 1024,
            ..ParseBudget::DEFAULT
        };
        assert!(input.len() <= limited.max_alloc_bytes);
        (
            DicomFile::parse(input.clone(), &ParseBudget::DEFAULT).unwrap(),
            DicomFile::parse(input, &limited).unwrap_err(),
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(valid.element(PATIENT).unwrap().to_str().unwrap(), expected);
        assert_eq!(
            valid
                .element(DicomTag::new(0x0040, 0xa160))
                .unwrap()
                .to_str()
                .unwrap(),
            "report ".repeat(300).trim_end()
        );
        assert!(refusal
            .to_string()
            .contains("deflated DICOM dataset exceeds parse budget"));
    });
}

fn jpeg(sample: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 100)
        .encode(&[sample; 64], 8, 8, image::ColorType::L8)
        .unwrap();
    bytes
}

#[test]
fn basic_offsets_join_fragments_and_keep_a_contiguous_frame_borrowed() {
    let _trace = trace_test();
    let (object, second) = tracing::info_span!("arrange").in_scope(|| {
        let first = jpeg(37);
        let second = jpeg(181);
        let split = (first.len() / 2) & !1;
        let tail = first.len() - split;
        let offset = u32::try_from(16 + first.len() + tail % 2).unwrap();
        let pixel = PixelFragments::new(
            vec![0, offset],
            [
                first[..split].to_vec(),
                first[split..].to_vec(),
                second.clone(),
            ],
        );
        let mut dataset = Dataset::new();
        dataset.put_value(DicomTag::new(0x28, 0x0008), VR::IS, "2");
        dataset.put(DataElement::new(PIXEL_DATA, VR::OB, pixel));
        let mut bytes = Vec::new();
        file(TransferSyntaxKind::JpegBaseline, dataset)
            .write_all(&mut bytes)
            .unwrap();
        (
            DicomFile::parse(bytes.into(), &ParseBudget::DEFAULT).unwrap(),
            second,
        )
    });

    let (first, frame) = tracing::info_span!("act").in_scope(|| {
        let first = decode_frame_with::<NativeBackend>(
            &object,
            DecodeFrameRequest {
                frame_index: 0,
                transfer_syntax: TransferSyntaxKind::JpegBaseline,
                layout: PixelLayout {
                    rows: 8,
                    cols: 8,
                    samples_per_pixel: 1,
                    bits_allocated: 8,
                    bits_stored: 8,
                    pixel_representation: PixelSignedness::Unsigned,
                    rescale_slope: 2.0,
                    rescale_intercept: -5.0,
                },
            },
        )
        .unwrap();
        (first, object.encapsulated_frame(1).unwrap())
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(first.pixels, vec![69.0; 64]);
        let Cow::Borrowed(bytes) = frame else {
            panic!("one fragment stays borrowed")
        };
        assert_eq!(&bytes[..second.len()], &second);
        assert!(object.encapsulated_frame(2).is_err());
    });
}

#[test]
fn big_endian_words_round_trip_and_decode_at_native_sample_precision() {
    let _trace = trace_test();
    let dataset = tracing::info_span!("arrange").in_scope(|| {
        b"\x00\x28\x01\x00US\x00\x02\x00\x10\x00\x28\x01\x01US\x00\x02\x00\x10\x00\x28\x01\x02US\x00\x02\x00\x0f\x7f\xe0\x00\x10OW\x00\x00\x00\x00\x00\x04\xff\xfb\x01\x01"
    });
    let (decoded, output) = tracing::info_span!("act").in_scope(|| {
        let syntax = TransferSyntaxKind::ExplicitVrBigEndian;
        let file = DicomFile::parse(
            part10_with_data(syntax.uid(), dataset),
            &ParseBudget::DEFAULT,
        )
        .unwrap();
        let decoded = decode_frame_with::<NativeBackend>(
            &file,
            DecodeFrameRequest {
                transfer_syntax: syntax.clone(),
                frame_index: 0,
                layout: PixelLayout {
                    rows: 1,
                    cols: 2,
                    samples_per_pixel: 1,
                    bits_allocated: 16,
                    bits_stored: 16,
                    pixel_representation: PixelSignedness::Signed,
                    rescale_slope: 2.0,
                    rescale_intercept: -5.0,
                },
            },
        )
        .unwrap();
        let mut output = Vec::new();
        file.write_dataset(&mut output, &syntax).unwrap();
        (decoded, output)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(decoded.pixels, [-15.0, 509.0]);
        assert_eq!(output.as_slice(), dataset);
    });
}

#[test]
fn writer_rejects_native_pixel_bytes_under_an_encapsulated_syntax_before_overwrite() {
    let _trace = trace_test();
    let (file, temp) = tracing::info_span!("arrange").in_scope(|| {
        let mut dataset = Dataset::new();
        dataset.put_value(PIXEL_DATA, VR::OB, [17, 31]);
        let file = file(TransferSyntaxKind::JpegBaseline, dataset);
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), b"original").unwrap();
        (file, temp)
    });
    let error =
        tracing::info_span!("act").in_scope(|| file.write_to_file(temp.path()).unwrap_err());
    tracing::info_span!("assert").in_scope(|| {
        assert!(format!("{error:#}")
            .contains("Pixel Data representation disagrees with transfer syntax"));
        assert_eq!(std::fs::read(temp.path()).unwrap(), b"original");
    });
}
