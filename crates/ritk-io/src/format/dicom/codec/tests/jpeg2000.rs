use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

/// Build and write a minimal JPEG 2000 Lossless DICOM Part 10 file.
///
/// Pixel data is encoded as a bare J2K codestream (`OPJ_CODEC_J2K`) using the
/// 5/3 reversible integer wavelet transform (`irreversible = 0`) via the Rust
/// `openjp2` port.
/// One resolution level is used to satisfy tile-size constraints for small images.
fn write_jpeg2000_lossless_dicom_file(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels_u16: &[u16],
) {
    use std::ffi::CString;

    assert_eq!(
        pixels_u16.len(),
        (width * height) as usize,
        "pixels_u16 length must equal width × height"
    );

    let j2k_tmp = tempfile::NamedTempFile::new().expect("NamedTempFile::new failed");
    let j2k_tmp_path = j2k_tmp.into_temp_path();
    let tmp_path_str = j2k_tmp_path
        .to_str()
        .expect("temp file path is not valid UTF-8");
    let tmp_cstr = CString::new(tmp_path_str).expect("CString::new failed");

    let j2k_bytes = unsafe {
        use openjp2::openjpeg::{
            opj_cparameters_t, opj_create_compress, opj_destroy_codec, opj_encode,
            opj_end_compress, opj_image_create, opj_image_destroy,
            opj_set_default_encoder_parameters, opj_setup_encoder, opj_start_compress,
            opj_stream_create_default_file_stream, opj_stream_destroy, CODEC_FORMAT, COLOR_SPACE,
            OPJ_BOOL, OPJ_FALSE, OPJ_TRUE,
        };
        use openjp2::opj_image_comptparm as opj_image_cmptparm_t;

        let mut params: opj_cparameters_t = std::mem::zeroed();
        opj_set_default_encoder_parameters(&mut params);
        params.irreversible = 0;
        params.numresolution = 1;

        let mut cmptparm: opj_image_cmptparm_t = std::mem::zeroed();
        cmptparm.dx = 1;
        cmptparm.dy = 1;
        cmptparm.w = width;
        cmptparm.h = height;
        cmptparm.x0 = 0;
        cmptparm.y0 = 0;
        cmptparm.prec = 16;
        cmptparm.bpp = 16;
        cmptparm.sgnd = 0;

        let image = opj_image_create(1, &mut cmptparm, COLOR_SPACE::OPJ_CLRSPC_GRAY);
        assert!(!image.is_null(), "opj_image_create returned NULL");
        (*image).x0 = 0;
        (*image).y0 = 0;
        (*image).x1 = width;
        (*image).y1 = height;

        let data_ptr = (*(*image).comps).data;
        assert!(
            !data_ptr.is_null(),
            "opj_image_comp_t data pointer is NULL after opj_image_create"
        );
        for (i, &px) in pixels_u16.iter().enumerate() {
            *data_ptr.add(i) = px as i32;
        }

        let codec = opj_create_compress(CODEC_FORMAT::OPJ_CODEC_J2K);
        assert!(!codec.is_null(), "opj_create_compress returned NULL");

        let setup_ok = opj_setup_encoder(codec, &mut params, image);
        assert_eq!(
            setup_ok, OPJ_TRUE as OPJ_BOOL,
            "opj_setup_encoder failed (returned {setup_ok})"
        );

        let stream =
            opj_stream_create_default_file_stream(tmp_cstr.as_ptr(), OPJ_FALSE as OPJ_BOOL);
        assert!(
            !stream.is_null(),
            "opj_stream_create_default_file_stream returned NULL"
        );

        let start_ok = opj_start_compress(codec, image, stream);
        assert_eq!(
            start_ok, OPJ_TRUE as OPJ_BOOL,
            "opj_start_compress failed (returned {start_ok})"
        );

        let encode_ok = opj_encode(codec, stream);
        assert_eq!(
            encode_ok, OPJ_TRUE as OPJ_BOOL,
            "opj_encode failed (returned {encode_ok})"
        );

        let end_ok = opj_end_compress(codec, stream);
        assert_eq!(
            end_ok, OPJ_TRUE as OPJ_BOOL,
            "opj_end_compress failed (returned {end_ok})"
        );

        opj_stream_destroy(stream);
        opj_image_destroy(image);
        opj_destroy_codec(codec);

        std::fs::read(&*j2k_tmp_path).expect("failed to read encoded J2K codestream")
    };

    assert!(
        !j2k_bytes.is_empty(),
        "J2K encoded codestream must not be empty"
    );

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![j2k_bytes]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.99999951"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0020),
        VR::LO,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("2.25.99999952"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.99999953"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(height as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(width as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(16u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(15u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from("1.000000"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1052),
        VR::DS,
        PrimitiveValue::from("0.000000"),
    ));
    obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, pfs));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.99999951")
                .transfer_syntax("1.2.840.10008.1.2.4.90"),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(path).expect("write_to_file failed");
}

/// JPEG 2000 Lossless round-trip: encode known 16-bit pixel values, decode via codec,
/// verify exact pixel equality (lossless invariant).
///
/// Mathematical justification:
/// JPEG 2000 Lossless (TS 1.2.840.10008.1.2.4.90) uses the 5/3 reversible integer
/// wavelet transform with lossless coding. Per ISO 15444-1 §C.5.5.1, irreversible=0
/// ⟹ exact reconstruction: |decoded[i] − original[i]| = 0 for all i.
///   Encode: J2K_Lossless(S, irreversible=0) → codestream C
///   Decode: J2K_Decode(C) → S' where S'[i] = S[i] for all i.
/// Max error = max|S[i] − S'[i]| = 0.
///
/// Pixel set spans [0, 4095] with boundary and interior samples representative of
/// a 12-bit effective range within 16-bit allocation.
#[test]
fn test_decode_compressed_frame_jpeg2000_lossless_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u16> = vec![
        0, 256, 512, 1024, 2048, 3071, 3584, 3840, 100, 200, 400, 800, 1600, 2400, 3000, 4095,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpeg2000_lossless.dcm");
    write_jpeg2000_lossless_dicom_file(&path, width, height, &original);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 16, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
        .expect("decode_compressed_frame must succeed for JPEG 2000 Lossless");

    assert_eq!(decoded.len(), 16, "decoded pixel count must equal 16");
    for (i, &v) in decoded.iter().enumerate() {
        assert!(
            (0.0..=4095.0).contains(&v),
            "decoded[{i}] = {v} is outside valid range [0, 4095]"
        );
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_error, 0.0,
        "JPEG 2000 Lossless decode error {max_error} must be exactly 0 \
         (ISO 15444-1 §C.5.5.1: irreversible=0 ⟹ |S'[i]−S[i]| = 0)"
    );
    for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig as f32,
            dec,
            "pixel[{i}]: expected {orig}, got {dec} — JPEG 2000 lossless must preserve all sample values"
        );
    }
}
