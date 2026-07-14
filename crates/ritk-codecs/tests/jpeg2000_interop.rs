//! Differential interoperability tests: RITK-native JPEG 2000 codec versus
//! the `openjp2` reference implementation (the pure-Rust c2rust port of
//! OpenJPEG, dev-dependency only).
//!
//! # Evidence tier
//! Cross-implementation differential validation of the lossless contract:
//! - `openjp2` encode → RITK decode: every sample exact.
//! - RITK encode → `openjp2` decode: every sample exact.

use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, WaveletTransform};
use ritk_codecs::{decode_jpeg2000_fragment, PixelLayout, PixelSignedness};
use std::ffi::CString;
use std::io::Write;

fn layout(rows: usize, cols: usize, bits: u16) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: bits,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

/// Deterministic CT-like content: gradient + LCG noise within `2^prec`.
fn synthetic(rows: u32, cols: u32, prec: u32) -> Vec<i32> {
    let mut state = 0xC0FF_EE00_DEAD_F00Du64;
    let amplitude = 1i64 << prec;
    (0..(rows * cols) as usize)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) % 64) as i64;
            (((i as i64 * 5) + noise) % amplitude) as i32
        })
        .collect()
}

/// Encode a grayscale image to a bare J2K codestream with `openjp2`,
/// `numresolution = num_decomp_levels + 1`. `irreversible` selects the wavelet:
/// 0 → 5/3 reversible (lossless), 1 → 9/7 irreversible (lossy). With the
/// default rate allocation (no `tcp_rates`) the encoder retains every coding
/// pass, so a lossy stream loses precision only through scalar quantization.
fn openjp2_encode(
    pixels: &[i32],
    width: u32,
    height: u32,
    prec: u32,
    numres: i32,
    irreversible: i32,
) -> Vec<u8> {
    let tmp = tempfile::NamedTempFile::new().expect("NamedTempFile::new failed");
    let tmp_path = tmp.into_temp_path();
    let tmp_cstr = CString::new(tmp_path.to_str().expect("temp path utf-8")).expect("CString");

    unsafe {
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
        params.irreversible = irreversible; // 0 → 5/3 reversible, 1 → 9/7 irreversible
        params.numresolution = numres;

        let mut cmptparm: opj_image_cmptparm_t = std::mem::zeroed();
        cmptparm.dx = 1;
        cmptparm.dy = 1;
        cmptparm.w = width;
        cmptparm.h = height;
        cmptparm.x0 = 0;
        cmptparm.y0 = 0;
        cmptparm.prec = prec;
        cmptparm.bpp = prec;
        cmptparm.sgnd = 0;

        let image = opj_image_create(1, &mut cmptparm, COLOR_SPACE::OPJ_CLRSPC_GRAY);
        assert!(!image.is_null(), "opj_image_create returned NULL");
        (*image).x0 = 0;
        (*image).y0 = 0;
        (*image).x1 = width;
        (*image).y1 = height;

        let data_ptr = (*(*image).comps).data;
        assert!(!data_ptr.is_null(), "component data NULL");
        for (i, &px) in pixels.iter().enumerate() {
            *data_ptr.add(i) = px;
        }

        let codec = opj_create_compress(CODEC_FORMAT::OPJ_CODEC_J2K);
        assert!(!codec.is_null(), "opj_create_compress returned NULL");
        assert_eq!(
            opj_setup_encoder(codec, &mut params, image),
            OPJ_TRUE as OPJ_BOOL,
            "opj_setup_encoder failed"
        );

        let stream =
            opj_stream_create_default_file_stream(tmp_cstr.as_ptr(), OPJ_FALSE as OPJ_BOOL);
        assert!(!stream.is_null(), "stream NULL");
        assert_eq!(
            opj_start_compress(codec, image, stream),
            OPJ_TRUE as OPJ_BOOL,
            "opj_start_compress failed"
        );
        assert_eq!(
            opj_encode(codec, stream),
            OPJ_TRUE as OPJ_BOOL,
            "opj_encode failed"
        );
        assert_eq!(
            opj_end_compress(codec, stream),
            OPJ_TRUE as OPJ_BOOL,
            "opj_end_compress failed"
        );

        opj_stream_destroy(stream);
        opj_image_destroy(image);
        opj_destroy_codec(codec);
    }

    let bytes = std::fs::read(&tmp_path).expect("read encoded J2K");
    assert!(!bytes.is_empty(), "openjp2 produced an empty codestream");
    bytes
}

/// Decode a bare J2K codestream with the public `openjp2` C-compatible API.
/// The temporary file uses the reference crate's real file stream path and
/// keeps this differential oracle independent of a second wrapper crate.
fn openjp2_decode(j2k: &[u8]) -> Vec<i32> {
    let mut tmp = tempfile::NamedTempFile::new().expect("NamedTempFile::new failed");
    tmp.write_all(j2k).expect("write encoded J2K");
    let tmp_path = tmp.into_temp_path();
    let tmp_cstr = CString::new(tmp_path.to_str().expect("temp path utf-8"))
        .expect("temporary path cannot contain NUL");

    // SAFETY: every pointer is created by openjp2 or points to storage owned
    // by this function; the reference API is torn down after the decoded
    // component is copied into an owned Vec.
    unsafe {
        use openjp2::openjpeg::{
            opj_create_decompress, opj_decode, opj_destroy_codec, opj_dparameters_t,
            opj_end_decompress, opj_image_destroy, opj_read_header,
            opj_setup_decoder, opj_stream_create_default_file_stream, opj_stream_destroy,
            CODEC_FORMAT, OPJ_BOOL, OPJ_TRUE,
        };
        use openjp2::opj_image as opj_image_t;

        let mut parameters = opj_dparameters_t::default();
        let codec = opj_create_decompress(CODEC_FORMAT::OPJ_CODEC_J2K);
        assert!(!codec.is_null(), "opj_create_decompress returned NULL");
        assert_eq!(
            opj_setup_decoder(codec, &mut parameters),
            OPJ_TRUE as OPJ_BOOL,
            "opj_setup_decoder failed"
        );

        let stream = opj_stream_create_default_file_stream(
            tmp_cstr.as_ptr(),
            OPJ_TRUE as OPJ_BOOL,
        );
        assert!(!stream.is_null(), "decoder stream is NULL");
        let mut image = std::ptr::null_mut::<opj_image_t>();
        assert_eq!(
            opj_read_header(stream, codec, &mut image),
            OPJ_TRUE as OPJ_BOOL,
            "opj_read_header failed"
        );
        assert!(!image.is_null(), "decoder image is NULL");
        assert_eq!((*image).numcomps, 1, "single grayscale component expected");
        assert_eq!(
            opj_decode(codec, stream, image),
            OPJ_TRUE as OPJ_BOOL,
            "opj_decode failed"
        );
        assert_eq!(
            opj_end_decompress(codec, stream),
            OPJ_TRUE as OPJ_BOOL,
            "opj_end_decompress failed"
        );

        let component = &*(*image).comps;
        assert!(!component.data.is_null(), "decoded component data is NULL");
        let width = usize::try_from(component.w).expect("component width exceeds usize");
        let height = usize::try_from(component.h).expect("component height exceeds usize");
        let sample_count = width.checked_mul(height).expect("component shape overflows usize");
        let data = std::slice::from_raw_parts(component.data, sample_count).to_vec();

        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
        opj_image_destroy(image);
        data
    }
}

/// Reference → RITK: every sample must reconstruct exactly.
fn assert_ritk_decodes_openjp2(rows: u32, cols: u32, prec: u32, numres: i32) {
    let pixels = synthetic(rows, cols, prec);
    let j2k = openjp2_encode(&pixels, cols, rows, prec, numres, 0);
    let bits = if prec <= 8 { 8u16 } else { 16 };
    let decoded = decode_jpeg2000_fragment(&j2k, layout(rows as usize, cols as usize, bits))
        .unwrap_or_else(|e| {
            panic!("RITK decode of openjp2 stream failed ({rows}×{cols}, prec {prec}, numres {numres}): {e:#}")
        });
    let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    assert_eq!(
        decoded, expected,
        "openjp2 → RITK must be exact ({rows}×{cols}, prec {prec}, numres {numres})"
    );
}

/// RITK → reference: every sample must reconstruct exactly.
fn assert_openjp2_decodes_ritk(rows: u32, cols: u32, prec: u32, levels: u8) {
    let pixels = synthetic(rows, cols, prec);
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        prec,
        PixelSignedness::Unsigned,
        levels,
        WaveletTransform::Reversible,
    );
    let data = openjp2_decode(&j2k);
    assert_eq!(data.len(), (rows * cols) as usize);
    for (i, (&orig, &dec)) in pixels.iter().zip(data.iter()).enumerate() {
        assert_eq!(
            dec, orig,
            "RITK → openjp2 sample[{i}] mismatch ({rows}×{cols}, prec {prec}, L{levels})"
        );
    }
}

// ── openjp2 → RITK ───────────────────────────────────────────────────────────

#[test]
fn openjp2_to_ritk_64x64_8bit_no_dwt() {
    assert_ritk_decodes_openjp2(64, 64, 8, 1);
}

#[test]
fn openjp2_to_ritk_64x64_16bit_three_levels() {
    assert_ritk_decodes_openjp2(64, 64, 16, 4);
}

#[test]
fn openjp2_to_ritk_150x100_8bit_five_levels_multi_cblk() {
    assert_ritk_decodes_openjp2(100, 150, 8, 6);
}

#[test]
fn openjp2_to_ritk_64x80_16bit_five_levels() {
    // Matches the GDCM/OpenJPEG DICOM writer defaults (numres = 6).
    assert_ritk_decodes_openjp2(64, 80, 16, 6);
}

/// Full openjp2 → RITK matrix: sizes × {8, 12, 16}-bit × all resolution
/// counts. Regression coverage for the §B.10.1 header bit-stuffing defect
/// (we byte-stuffed 0x00 after 0xFF instead of 7-bit stuffing), which only
/// manifested when a packet header happened to contain 0xFF.
#[test]
fn openjp2_to_ritk_matrix() {
    for &(rows, cols) in &[(64u32, 64u32), (64, 80), (80, 64), (100, 150)] {
        for &prec in &[8u32, 12, 16] {
            for numres in 1..=6i32 {
                assert_ritk_decodes_openjp2(rows, cols, prec, numres);
            }
        }
    }
}

/// Full RITK → openjp2 matrix (reverse direction of [`openjp2_to_ritk_matrix`]).
#[test]
fn ritk_to_openjp2_matrix() {
    for &(rows, cols) in &[(64u32, 64u32), (64, 80), (100, 150)] {
        for &prec in &[8u32, 12, 16] {
            for levels in 0..=5u8 {
                assert_openjp2_decodes_ritk(rows, cols, prec, levels);
            }
        }
    }
}

// ── RITK → openjp2 ───────────────────────────────────────────────────────────

#[test]
fn ritk_to_openjp2_64x64_8bit_no_dwt() {
    assert_openjp2_decodes_ritk(64, 64, 8, 0);
}

#[test]
fn ritk_to_openjp2_64x64_16bit_three_levels() {
    assert_openjp2_decodes_ritk(64, 64, 16, 3);
}

#[test]
fn ritk_to_openjp2_150x100_8bit_two_levels_multi_cblk() {
    assert_openjp2_decodes_ritk(100, 150, 8, 2);
}
/// Extract the tile body (after SOD, before EOC) from a bare J2K codestream.
fn tile_body(j2k: &[u8]) -> &[u8] {
    let sod = j2k
        .windows(2)
        .position(|w| w == [0xFF, 0x93])
        .expect("SOD present");
    let end = j2k.len() - 2; // strip EOC
    &j2k[sod + 2..end]
}

/// Escalation harness: byte-compare our encoder against openjp2 for a series
/// of increasingly complex 8×8 images; reports the first differing byte.
#[test]
fn escalation_byte_compare_with_openjp2() {
    // 128 is the 8-bit unsigned DC level: pixel 128 → coefficient 0, so these
    // are true coefficient impulses after the encoder's DC shift.
    let mut impulse = vec![128i32; 64];
    impulse[0] = 138; // coefficient +10 at (0,0)
    let mut impulse_mid = vec![128i32; 64];
    impulse_mid[3 * 8 + 1] = 100; // coefficient −28 at (1,3)
    let mut two = vec![128i32; 64];
    two[0] = 200; // +72
    two[9] = 100; // −28
    let ramp: Vec<i32> = (0..64).map(|i| (i % 8) * 30).collect();
    let vramp: Vec<i32> = (0..64).map(|i| (i / 8) * 30).collect();
    let synth = synthetic(8, 8, 8);
    // Bit-plane bisection: +1 → exactly one cleanup pass; +3 → CUP then one
    // SPP/MRP/CUP round.
    let mut v1 = vec![128i32; 64];
    v1[0] = 129;
    let mut v1_mid = vec![128i32; 64];
    v1_mid[4 * 8 + 4] = 129;
    let mut v3 = vec![128i32; 64];
    v3[0] = 131;
    let mut v3_neg = vec![128i32; 64];
    v3_neg[4 * 8 + 3] = 125;
    let cases: [(&str, &[i32]); 10] = [
        ("v1_corner", &v1),
        ("v1_mid", &v1_mid),
        ("v3_corner", &v3),
        ("v3_neg_mid", &v3_neg),
        ("impulse00", &impulse),
        ("impulse31", &impulse_mid),
        ("two_px", &two),
        ("hramp", &ramp),
        ("vramp", &vramp),
        ("synthetic", &synth),
    ];
    let mut failures = Vec::new();
    for (name, px) in cases {
        let ours_stream = encode_grayscale_j2k(
            px,
            8,
            8,
            8,
            PixelSignedness::Unsigned,
            0,
            WaveletTransform::Reversible,
        );
        let refs_stream = openjp2_encode(px, 8, 8, 8, 1, 0);
        let ours = tile_body(&ours_stream);
        let refs = tile_body(&refs_stream);
        let diff = ours
            .iter()
            .zip(refs.iter())
            .position(|(a, b)| a != b)
            .filter(|_| true)
            .or(if ours.len() != refs.len() {
                Some(ours.len().min(refs.len()))
            } else {
                None
            });
        eprintln!(
            "case {name}: ours {} bytes, ref {} bytes, first diff {:?}",
            ours.len(),
            refs.len(),
            diff
        );
        if let Some(k) = diff {
            let lo = k.saturating_sub(2);
            eprintln!(
                "  ours[{lo}..] = {:02X?}\n  refs[{lo}..] = {:02X?}",
                &ours[lo..(k + 6).min(ours.len())],
                &refs[lo..(k + 6).min(refs.len())]
            );
            failures.push(name);
        }
    }
    assert!(
        failures.is_empty(),
        "byte divergence in cases: {failures:?}"
    );
}
/// Regression (J2K-INTEROP): single-impulse 8×8 image, no DWT — the minimized
/// case for the MQ probability-estimation defect (state advanced on every MPS
/// instead of only on renormalisation, ISO 15444-1 §C.2.6/Figure C.7). Both
/// cross-decode directions must reconstruct every sample exactly.
#[test]
fn cross_decode_impulse_8x8_regression() {
    let mut v1_mid = vec![128i32; 64];
    v1_mid[4 * 8 + 4] = 129;

    // openjp2 stream → RITK decoder.
    let theirs = openjp2_encode(&v1_mid, 8, 8, 8, 1, 0);
    let dec = decode_jpeg2000_fragment(&theirs, layout(8, 8, 8))
        .expect("RITK decode of openjp2 impulse stream");
    let expected: Vec<f32> = v1_mid.iter().map(|&p| p as f32).collect();
    assert_eq!(dec, expected, "openjp2 → RITK impulse reconstruction");

    // RITK stream → openjp2 decoder.
    let ours = encode_grayscale_j2k(
        &v1_mid,
        8,
        8,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    );
    let data = openjp2_decode(&ours);
    assert_eq!(data, v1_mid, "RITK → openjp2 impulse reconstruction");
}

// ── Lossy 9/7 differential decode (openjp2 → RITK vs the reference) ────────────
//
// The reversible interop tests prove the MQ coder, EBCOT, tier-2 packets, the
// 5/3 wavelet, and the codestream parser are openjp2-conformant. The remaining
// unverified surface of the *lossy* path is exactly the 9/7 inverse lifting and
// the QCD scalar-quantization step-size parsing. This test isolates them: a
// 9/7-irreversible stream is encoded once by openjp2 (default rate allocation,
// so every coding pass is retained and the only precision loss is scalar
// quantization), then decoded by **both** RITK and openjp2.
//
// The oracle is *fidelity to the original image*, not per-sample agreement with
// openjp2. JPEG 2000 leaves the dequantization reconstruction bias r (ISO 15444-1
// §E.1.1.2) to the decoder — RITK uses the midpoint r = 0.5, openjp2 a different
// value — and after the inverse 9/7 (whose synthesis filters have gain > 1) a
// sub-step bias difference propagates to several sample units. So the two
// conformant reconstructions legitimately differ by more than one level; what a
// *correct* 9/7-inverse + QCD-parse guarantees is that RITK reconstructs the
// original about as faithfully as the reference. We assert RITK's PSNR is within
// 1 dB of openjp2's: a robust discriminator, since a correct inverse tracks the
// reference to a small fraction of a dB while any structural error in the 9/7
// coefficients or step-size decoding collapses PSNR by tens of dB.

/// Mean-squared error of a reconstruction against the original integer samples.
fn mse_vs_original(recon: impl Iterator<Item = f64>, original: &[i32]) -> f64 {
    let sum: f64 = recon
        .zip(original)
        .map(|(r, &p)| {
            let e = r - f64::from(p);
            e * e
        })
        .sum();
    sum / original.len() as f64
}

/// Peak-signal-to-noise ratio in dB for `prec`-bit samples (∞ when `mse == 0`).
fn psnr(mse: f64, prec: u32) -> f64 {
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    let peak = f64::from((1u32 << prec) - 1);
    10.0 * (peak * peak / mse).log10()
}

/// openjp2 9/7-irreversible encode → {RITK decode, openjp2 decode}; RITK must
/// reconstruct the original within 1 dB of the reference decoder's PSNR.
fn assert_ritk_matches_openjp2_lossy(rows: u32, cols: u32, prec: u32, numres: i32) {
    let pixels = synthetic(rows, cols, prec);
    let j2k = openjp2_encode(&pixels, cols, rows, prec, numres, 1);

    let bits = if prec <= 8 { 8u16 } else { 16 };
    let ritk = decode_jpeg2000_fragment(&j2k, layout(rows as usize, cols as usize, bits))
        .unwrap_or_else(|e| {
            panic!("RITK decode of openjp2 9/7 stream failed ({rows}×{cols}, prec {prec}, numres {numres}): {e:#}")
        });

    let openjp2 = openjp2_decode(&j2k);

    assert_eq!(
        ritk.len(),
        openjp2.len(),
        "sample-count mismatch ({rows}×{cols}, prec {prec}, numres {numres})"
    );

    let ritk_mse = mse_vs_original(ritk.iter().map(|&r| f64::from(r)), &pixels);
    let openjp2_mse = mse_vs_original(openjp2.iter().map(|&o| f64::from(o)), &pixels);
    let (ritk_psnr, openjp2_psnr) = (psnr(ritk_mse, prec), psnr(openjp2_mse, prec));

    eprintln!(
        "lossy {rows}×{cols} prec{prec} numres{numres}: RITK {ritk_psnr:.2} dB, openjp2 {openjp2_psnr:.2} dB \
         (mse {ritk_mse:.3} vs {openjp2_mse:.3})"
    );
    // A correct 9/7 inverse + QCD parse tracks the reference to a fraction of a
    // dB; the 1 dB floor is orders of magnitude away from the collapse a
    // structural defect would produce.
    assert!(
        ritk_psnr >= openjp2_psnr - 1.0,
        "RITK lossy PSNR {ritk_psnr:.2} dB is more than 1 dB below openjp2 {openjp2_psnr:.2} dB \
         ({rows}×{cols}, prec {prec}, numres {numres}) — 9/7 inverse or QCD parse defect"
    );
}

#[test]
fn lossy_openjp2_to_ritk_64x64_8bit_three_levels() {
    assert_ritk_matches_openjp2_lossy(64, 64, 8, 4);
}

#[test]
fn lossy_openjp2_to_ritk_64x80_16bit_five_levels() {
    assert_ritk_matches_openjp2_lossy(64, 80, 16, 6);
}

/// Lossy 9/7 differential matrix across sizes, precisions, and resolution counts.
///
/// Covers `numres = 1..=6`, including the degenerate zero-decomposition-level
/// case (no wavelet, pure scalar quantization). With the bitplane-aware
/// reconstruction (a fully decoded index reconstructs at q·Δ, matching the
/// OpenJPEG reference), RITK tracks openjp2 within 1 dB PSNR across the whole
/// matrix — the 9/7 inverse lifting, the QCD step-size parsing, and the
/// dequantization reconstruction are all validated against the reference.
#[test]
fn lossy_openjp2_to_ritk_matrix() {
    for &(rows, cols) in &[(64u32, 64u32), (64, 80), (100, 150)] {
        for &prec in &[8u32, 12, 16] {
            for numres in 1..=6i32 {
                assert_ritk_matches_openjp2_lossy(rows, cols, prec, numres);
            }
        }
    }
}
