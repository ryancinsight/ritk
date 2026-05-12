//! Test-only JPEG 2000 codestream construction through the Rust `openjp2` port.
//!
//! The helper encodes a bare J2K codestream with the reversible 5/3 transform so
//! decoder tests can verify exact ISO 15444-1 lossless reconstruction without
//! linking the OpenJPEG C library.

use openjp2::openjpeg as opj;

pub(super) fn encode_grayscale_j2k(
    pixels: &[i32],
    rows: u32,
    cols: u32,
    precision: u32,
    signed: bool,
) -> Vec<u8> {
    assert_eq!(
        pixels.len(),
        (rows * cols) as usize,
        "pixels length must equal rows × cols"
    );

    unsafe {
        let mut out: Vec<u8> = Vec::new();
        let write_stream = create_memory_write_stream(&mut out);
        let image = create_image(pixels, rows, cols, precision, signed);
        let codec = opj::opj_create_compress(opj::CODEC_FORMAT::OPJ_CODEC_J2K);
        assert!(!codec.is_null(), "opj_create_compress returned null");

        opj::opj_set_info_handler(codec, None, std::ptr::null_mut());
        opj::opj_set_warning_handler(codec, None, std::ptr::null_mut());
        opj::opj_set_error_handler(codec, None, std::ptr::null_mut());

        let mut params: opj::opj_cparameters_t = std::mem::zeroed();
        opj::opj_set_default_encoder_parameters(&mut params);
        params.irreversible = 0;
        params.numresolution = 1;

        assert_eq!(
            opj::opj_setup_encoder(codec, &mut params, image),
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_setup_encoder failed"
        );
        assert_eq!(
            opj::opj_start_compress(codec, image, write_stream),
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_start_compress failed"
        );
        assert_eq!(
            opj::opj_encode(codec, write_stream),
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_encode failed"
        );
        assert_eq!(
            opj::opj_end_compress(codec, write_stream),
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_end_compress failed"
        );

        opj::opj_stream_destroy(write_stream);
        opj::opj_destroy_codec(codec);
        opj::opj_image_destroy(image);

        out
    }
}

unsafe fn create_memory_write_stream(out: &mut Vec<u8>) -> *mut opj::opj_stream_t {
    unsafe extern "C" fn write_fn(
        buffer: *mut std::ffi::c_void,
        nb_bytes: opj::OPJ_SIZE_T,
        user_data: *mut std::ffi::c_void,
    ) -> opj::OPJ_SIZE_T {
        let out = &mut *(user_data as *mut Vec<u8>);
        let slice = std::slice::from_raw_parts(buffer as *const u8, nb_bytes);
        out.extend_from_slice(slice);
        nb_bytes
    }

    unsafe extern "C" fn seek_out_fn(
        nb_bytes: opj::OPJ_OFF_T,
        _user_data: *mut std::ffi::c_void,
    ) -> opj::OPJ_BOOL {
        if nb_bytes >= 0 {
            opj::OPJ_TRUE as opj::OPJ_BOOL
        } else {
            opj::OPJ_FALSE as opj::OPJ_BOOL
        }
    }

    unsafe extern "C" fn skip_out_fn(
        nb_bytes: opj::OPJ_OFF_T,
        _user_data: *mut std::ffi::c_void,
    ) -> opj::OPJ_OFF_T {
        nb_bytes.max(0)
    }

    let write_stream = opj::opj_stream_default_create(opj::OPJ_FALSE as opj::OPJ_BOOL);
    assert!(
        !write_stream.is_null(),
        "opj_stream_default_create returned null"
    );
    opj::opj_stream_set_write_function(write_stream, Some(write_fn));
    opj::opj_stream_set_skip_function(write_stream, Some(skip_out_fn));
    opj::opj_stream_set_seek_function(write_stream, Some(seek_out_fn));
    opj::opj_stream_set_user_data(
        write_stream,
        out as *mut Vec<u8> as *mut std::ffi::c_void,
        None,
    );
    write_stream
}

unsafe fn create_image(
    pixels: &[i32],
    rows: u32,
    cols: u32,
    precision: u32,
    signed: bool,
) -> *mut opj::opj_image_t {
    let mut cmptparm: openjp2::opj_image_comptparm = std::mem::zeroed();
    cmptparm.dx = 1;
    cmptparm.dy = 1;
    cmptparm.w = cols;
    cmptparm.h = rows;
    cmptparm.prec = precision;
    cmptparm.bpp = precision;
    cmptparm.sgnd = u32::from(signed);

    let image = opj::opj_image_create(1, &mut cmptparm, opj::COLOR_SPACE::OPJ_CLRSPC_GRAY);
    assert!(!image.is_null(), "opj_image_create returned null");

    (*image).x0 = 0;
    (*image).y0 = 0;
    (*image).x1 = cols;
    (*image).y1 = rows;

    let comp = &mut *(*image).comps;
    assert!(
        !comp.data.is_null(),
        "opj_image_create returned component with null data"
    );
    for (i, &pixel) in pixels.iter().enumerate() {
        *comp.data.add(i) = pixel;
    }

    image
}
