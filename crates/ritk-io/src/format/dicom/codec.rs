//! DICOM pixel data codec integration via `dicom-pixeldata` and native decoders.
//!
//! # Architecture
//!
//! Provides `decode_compressed_frame`, the single dispatch point for all
//! codec-supported compressed transfer syntaxes.
//!
//! - **RLE Lossless**: dispatched to `decode_rle_lossless_frame`, a native
//!   implementation of DICOM PS3.5 Annex G (PackBits per byte-plane + correct
//!   LE reassembly). This bypasses `dicom-transfer-syntax-registry v0.8.2`,
//!   which has an off-by-one write-start offset (`start = 1` instead of `0`)
//!   for 8-bit grayscale images, silently corrupting `pixel[0]` and losing
//!   `pixel[N−1]` for any file where `pixel[0] ≠ 0`.
//! - **All other compressed transfer syntaxes**: calls
//!   `dicom_pixeldata::PixelDecoder::decode_pixel_data_frame` to recover raw
//!   sample bytes, then applies the linear modality LUT via `decode_pixel_bytes`.
//!
//! # Supported codecs (pure Rust, `native` feature of `dicom-pixeldata`)
//!
//! | Transfer Syntax                        | UID                      | Codec          | Feature       |
//! |----------------------------------------|--------------------------|----------------|---------------|
//! | JPEG Baseline (Process 1)              | 1.2.840.10008.1.2.4.50   | jpeg-decoder   | native        |
//! | JPEG Extended (Process 2 & 4)          | 1.2.840.10008.1.2.4.51   | jpeg-decoder   | native        |
//! | JPEG Lossless Non-Hierarchical (P14)   | 1.2.840.10008.1.2.4.57   | jpeg-decoder   | native        |
//! | JPEG Lossless First-Order Prediction   | 1.2.840.10008.1.2.4.70   | jpeg-decoder   | native        |
//! | JPEG-LS Lossless                       | 1.2.840.10008.1.2.4.80   | charls         | charls        |
//! | JPEG-LS Near-Lossless                  | 1.2.840.10008.1.2.4.81   | charls         | charls        |
//! | JPEG 2000 Lossless                     | 1.2.840.10008.1.2.4.90   | openjpeg-sys   | openjpeg-sys  |
//! | JPEG 2000 Lossy                        | 1.2.840.10008.1.2.4.91   | openjpeg-sys   | openjpeg-sys  |
//! | RLE Lossless                           | 1.2.840.10008.1.2.5      | dicom-rle      | native        |
//! | JPEG XL Lossless                       | 1.2.840.10008.1.2.4.110  | jxl-oxide      | jpegxl        |
//! | JPEG XL JPEG Recompression             | 1.2.840.10008.1.2.4.111  | jxl-oxide      | jpegxl        |
//! | JPEG XL                                | 1.2.840.10008.1.2.4.112  | jxl-oxide      | jpegxl        |
//!
//! # Mathematical contract
//!
//! `decode_compressed_frame(obj, f, bits, repr, slope, intercept)`:
//!   `Output[i] = codec_sample[i] × slope + intercept`
//!
//! where `codec_sample[i]` is the integer produced by the codec for pixel i.
//! Identical semantics to `decode_pixel_bytes` (DICOM PS3.3 C.7.6.3.1.4).
//! - JPEG Extended tolerance: `|decoded[i] − original[i]| ≤ 16` (same Q75 bound as Baseline).
//! - RLE Lossless exact fidelity: `max|decoded[i] − original[i]| = 0` (lossless by spec).
//!
//! # Invariants
//!
//! - `is_codec_supported() ⟹ is_compressed()`: codec path only for compressed TS.
//! - `is_natively_supported() ⟹ !is_codec_supported()`: disjoint decode paths.
//! - Output length equals `rows × cols` for a single-frame decode.
//! - Rescale is always applied; identity rescale (slope=1, intercept=0) is valid.

use anyhow::{bail, Context, Result};
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use dicom_pixeldata::PixelDecoder;

use super::reader::decode_pixel_bytes;
use super::transfer_syntax::TransferSyntaxKind;

/// Decode one frame from a compressed DICOM object using the registered codec.
///
/// # Arguments
///
/// - `obj`: open Part 10 DICOM object with compressed transfer syntax in file meta.
/// - `frame_idx`: zero-based frame index (0 for single-frame objects).
/// - `bits_allocated`: from (0028,0100); drives byte interpretation in `decode_pixel_bytes`.
/// - `pixel_representation`: from (0028,0103); 0 = unsigned, 1 = signed.
/// - `slope`: RescaleSlope from (0028,1053); absent ⇒ 1.0.
/// - `intercept`: RescaleIntercept from (0028,1052); absent ⇒ 0.0.
///
/// # Returns
///
/// `Vec<f32>` of length `rows × cols` with modality LUT applied.
///
/// # Errors
///
/// Returns `Err` when the codec fails: unsupported TS, malformed compressed data,
/// or missing codec (feature not enabled).
pub(super) fn decode_compressed_frame(
    obj: &DefaultDicomObject,
    frame_idx: u32,
    bits_allocated: u16,
    pixel_representation: u16,
    slope: f32,
    intercept: f32,
) -> Result<Vec<f32>> {
    // RLE Lossless: bypass the upstream codec.
    // `dicom-transfer-syntax-registry v0.8.2` computes the write-start offset as
    // `start = spp − byte_offset` for the first sample, which yields 1 (not 0)
    // for 8-bit grayscale. This silently forces `dst[0] = 0` and loses `dst[N−1]`
    // for any file where `pixel[0] ≠ 0`. `decode_rle_lossless_frame` is a correct
    // native implementation per DICOM PS3.5 Annex G.
    if TransferSyntaxKind::from_uid(obj.meta().transfer_syntax()) == TransferSyntaxKind::RleLossless
    {
        return decode_rle_lossless_frame(
            obj,
            frame_idx,
            bits_allocated,
            pixel_representation,
            slope,
            intercept,
        );
    }

    let decoded = obj
        .decode_pixel_data_frame(frame_idx)
        .with_context(|| format!("codec decode failed for frame {frame_idx}"))?;
    let raw = decoded.data();
    Ok(decode_pixel_bytes(
        raw,
        bits_allocated,
        pixel_representation,
        slope,
        intercept,
    ))
}

/// Decode one PackBits-compressed byte segment (DICOM PS3.5 Annex G.3.1).
///
/// # Algorithm
///
/// For each header byte `h` (interpreted as `i8`):
/// - `h ∈ [0, 127]`:   copy the next `h + 1` literal bytes verbatim.
/// - `h = −128`:        no-op; advance past the header byte.
/// - `h ∈ [−127, −1]`: repeat the next byte `−h + 1` times.
///
/// # Mathematical contract
///
/// `packbits_decode(packbits_encode(S), S.len()) = S` for all `S: &[u8]`.
/// PackBits is lossless; decode is the strict left inverse of encode.
///
/// # Errors
///
/// Returns `Err` when the encoded stream is truncated before `expected_len`
/// bytes are produced, or when a literal run overflows the input buffer.
fn packbits_decode(input: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    let mut pos = 0usize;
    while pos < input.len() && out.len() < expected_len {
        let h = input[pos] as i8;
        pos += 1;
        if h >= 0 {
            // Literal run: copy h+1 bytes verbatim.
            let count = h as usize + 1;
            let end = pos + count;
            if end > input.len() {
                bail!(
                    "PackBits decode: literal run of {} bytes overflows input \
                     at pos {} (input length {})",
                    count,
                    pos,
                    input.len()
                );
            }
            out.extend_from_slice(&input[pos..end]);
            pos = end;
        } else if h != i8::MIN {
            // Repeat run: h ∈ [−127, −1] → count = −h + 1 ∈ [2, 128].
            // Cast via i16 to avoid i8 overflow when negating i8::MIN.
            let count = (-(h as i16)) as usize + 1;
            if pos >= input.len() {
                bail!(
                    "PackBits decode: repeat run at pos {} has no data byte",
                    pos
                );
            }
            let byte = input[pos];
            pos += 1;
            out.resize(out.len() + count, byte);
        }
        // h == i8::MIN (−128): no-op; already advanced past the header byte.
    }
    if out.len() < expected_len {
        bail!(
            "PackBits decode: produced {} bytes but expected {} \
             (input exhausted at pos {})",
            out.len(),
            expected_len,
            pos
        );
    }
    out.truncate(expected_len);
    Ok(out)
}

/// Decode one frame from a DICOM RLE Lossless object using the native decoder.
///
/// # DICOM RLE Lossless format (PS3.5 Annex G)
///
/// Each pixel fragment begins with a 64-byte RLE header (16 × `u32` LE):
/// - `header[0]`: total number of byte-plane segments.
/// - `header[k+1]` (`k ∈ [0, N-1]`): byte offset of segment `k` from the
///   start of the fragment.
///
/// Each segment contains PackBits-encoded bytes for one byte-plane of
/// `rows × cols` pixels.
///
/// # Segment ordering and LE byte reassembly
///
/// Per DICOM PS3.5 §G.5, `segment[s × B + b]` holds byte-plane `b` (MSB-first,
/// `b = 0` is MSB) for sample `s`, where `B = bits_allocated / 8`.
///
/// Reassembly into LE pixel bytes (as expected by `decode_pixel_bytes`):
///
/// ```text
/// raw[p × S × B + s × B + j] = segment[s × B + (B − 1 − j)][p]
/// ```
///
/// where `j = 0` is the LE LSB and `j = B − 1` is the LE MSB.
///
/// # Why the upstream codec is bypassed
///
/// `dicom-transfer-syntax-registry v0.8.2` computes the write-start as
/// `start = spp − byte_offset` (evaluates to 1, not 0, for 8-bit grayscale),
/// silently forcing `dst[0] = 0` and losing `dst[N−1]`. This decoder is correct
/// for all `bits_allocated ∈ {8, 16}` and any `samples_per_pixel`.
///
/// # Errors
///
/// Returns `Err` for missing DICOM tags, malformed RLE headers, segment offsets
/// out of bounds, or PackBits streams too short to fill the expected frame.
fn decode_rle_lossless_frame(
    obj: &DefaultDicomObject,
    frame_idx: u32,
    bits_allocated: u16,
    pixel_representation: u16,
    slope: f32,
    intercept: f32,
) -> Result<Vec<f32>> {
    // Read image geometry from the DICOM object.
    let rows: u32 = obj
        .element(Tag(0x0028, 0x0010))
        .context("RLE: missing Rows (0028,0010)")?
        .to_str()
        .context("RLE: Rows (0028,0010) not string-readable")?
        .trim()
        .parse()
        .context("RLE: Rows (0028,0010) not a valid integer")?;
    let cols: u32 = obj
        .element(Tag(0x0028, 0x0011))
        .context("RLE: missing Columns (0028,0011)")?
        .to_str()
        .context("RLE: Columns (0028,0011) not string-readable")?
        .trim()
        .parse()
        .context("RLE: Columns (0028,0011) not a valid integer")?;
    let samples_per_pixel: usize = obj
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);

    let pixels_per_frame = (rows as usize) * (cols as usize);
    if pixels_per_frame == 0 {
        bail!("RLE: rows={} cols={} yields 0 pixels per frame", rows, cols);
    }

    let bytes_per_sample = (bits_allocated / 8) as usize;
    if bytes_per_sample == 0 || bytes_per_sample > 4 {
        bail!(
            "RLE: bits_allocated={} → bytes_per_sample={} (must be 1–4)",
            bits_allocated,
            bytes_per_sample
        );
    }

    let expected_segments = samples_per_pixel * bytes_per_sample;

    // Retrieve the raw RLE fragment bytes for this frame.
    let pdata_elem = obj
        .element(Tag(0x7FE0, 0x0010))
        .context("RLE: missing Pixel Data (7FE0,0010)")?;
    let fragment_bytes: Vec<u8> = match pdata_elem.value() {
        Value::PixelSequence(seq) => {
            let frags = seq.fragments();
            frags
                .get(frame_idx as usize)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "RLE: frame {} out of range ({} fragments in pixel sequence)",
                        frame_idx,
                        frags.len()
                    )
                })?
                .to_vec()
        }
        _ => bail!(
            "RLE: Pixel Data (7FE0,0010) is not a PixelSequence \
             (expected encapsulated format for RLE Lossless)"
        ),
    };

    // Parse the 64-byte DICOM RLE header.
    const HEADER_BYTES: usize = 64;
    if fragment_bytes.len() < HEADER_BYTES {
        bail!(
            "RLE: fragment is {} bytes; minimum is {} for the RLE header",
            fragment_bytes.len(),
            HEADER_BYTES
        );
    }
    let n_segments = u32::from_le_bytes(fragment_bytes[0..4].try_into().unwrap()) as usize;
    if n_segments != expected_segments {
        bail!(
            "RLE: header declares {} segments; expected {} \
             (bits_allocated={}, samples_per_pixel={})",
            n_segments,
            expected_segments,
            bits_allocated,
            samples_per_pixel
        );
    }

    // Extract segment byte offsets from header slots 1..=N (each u32 LE).
    let seg_offsets: Vec<usize> = (0..n_segments)
        .map(|k| {
            u32::from_le_bytes(fragment_bytes[4 + k * 4..8 + k * 4].try_into().unwrap()) as usize
        })
        .collect();

    // Decode each byte-plane segment via PackBits.
    let mut segments: Vec<Vec<u8>> = Vec::with_capacity(n_segments);
    for (k, &offset) in seg_offsets.iter().enumerate() {
        if offset >= fragment_bytes.len() {
            bail!(
                "RLE: segment {} offset {} out of bounds (fragment {} bytes)",
                k,
                offset,
                fragment_bytes.len()
            );
        }
        // Segment data spans from `offset` to the start of the next segment
        // (or to the end of the fragment for the last segment).
        let end = if k + 1 < n_segments {
            seg_offsets[k + 1]
        } else {
            fragment_bytes.len()
        };
        let decoded_seg = packbits_decode(&fragment_bytes[offset..end], pixels_per_frame)
            .with_context(|| format!("RLE: PackBits decode failed for segment {k}"))?;
        segments.push(decoded_seg);
    }

    // Reassemble byte-plane segments into LE pixel bytes.
    //
    // DICOM segment ordering: segment[s*B + b] = byte-plane b (MSB-first) for sample s.
    // LE byte layout: LE byte j (0=LSB) of sample s for pixel p
    //   → segment index s*B + (B−1−j).
    let bps = bytes_per_sample;
    let spp = samples_per_pixel;
    let mut raw: Vec<u8> = Vec::with_capacity(pixels_per_frame * bps * spp);
    for pixel_idx in 0..pixels_per_frame {
        for s in 0..spp {
            for j in 0..bps {
                // j=0 → LE LSB → MSB-first segment index s*B + (B-1)
                // j=1 → next LE byte   → segment index s*B + (B-2)
                // j=B-1 → LE MSB → segment index s*B + 0
                let seg_idx = s * bps + (bps - 1 - j);
                raw.push(segments[seg_idx][pixel_idx]);
            }
        }
    }

    Ok(decode_pixel_bytes(
        &raw,
        bits_allocated,
        pixel_representation,
        slope,
        intercept,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::smallvec::SmallVec;
    use dicom::core::value::PixelFragmentSequence;
    use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

    /// Build and write a minimal JPEG Baseline DICOM Part 10 file.
    ///
    /// Pixel data is JPEG-encoded at default quality using the `image` crate and
    /// encapsulated as a single fragment per DICOM PS3.5 §A.4.
    ///
    /// # Parameters
    /// - `path`: destination file path.
    /// - `width`, `height`: image dimensions in pixels.
    /// - `pixels_u8`: flat row-major 8-bit grayscale values, length = `width × height`.
    fn write_jpeg_dicom_file(path: &std::path::Path, width: u32, height: u32, pixels_u8: &[u8]) {
        assert_eq!(
            pixels_u8.len(),
            (width * height) as usize,
            "pixels_u8 length must equal width × height"
        );

        // JPEG-encode via image crate (JFIF SOF0 Baseline).
        use image::{DynamicImage, GrayImage};
        let gray = GrayImage::from_raw(width, height, pixels_u8.to_vec())
            .expect("GrayImage::from_raw failed");
        let dyn_img = DynamicImage::ImageLuma8(gray);
        let mut jpeg_bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
        dyn_img
            .write_to(&mut cursor, image::ImageFormat::Jpeg)
            .expect("JPEG encode failed");
        drop(cursor);

        // Encapsulate as single fragment per DICOM PS3.5 §A.4.
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jpeg_bytes]);
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
            PrimitiveValue::from("2.25.99999901"),
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
            PrimitiveValue::from("2.25.99999902"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999903"),
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
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(7u16),
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
                    .media_storage_sop_instance_uid("2.25.99999901")
                    .transfer_syntax("1.2.840.10008.1.2.4.50"),
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(path).expect("write_to_file failed");
    }

    /// JPEG Baseline round-trip: encode known pixel values, decode via codec, verify
    /// each decoded value is within JPEG quantization tolerance of the original.
    ///
    /// Mathematical justification:
    /// At JPEG quality 75 the luminance quantization matrix is scaled by factor 0.5
    /// from the standard reference table (ITU T.81 Annex K). Per-pixel reconstruction
    /// error from a single quantized DCT coefficient (u,v) with step Q[u,v] via the
    /// 2D IDCT is bounded by Q[u,v]/2 (one quantization half-step). The dominant
    /// contributions for a smooth-gradient image come from the DC and primary AC terms:
    ///   DC  (0,0): Q = 8  → ≤ 4 per pixel
    ///   AC  (1,0): Q = 6  → ≤ 3 per pixel
    ///   AC  (0,1): Q = 6  → ≤ 3 per pixel
    ///   AC  (1,1): Q = 6  → ≤ 3 per pixel (activated by 4×4→8×8 edge replication)
    /// Sum of primary contributors: 4+3+3+3 = 13. Tolerance set to 16 (next integer
    /// power-of-2 ≥ 13) to accommodate higher-order AC contributions and fixed-point
    /// IDCT rounding in `jpeg-decoder`.
    #[test]
    fn test_decode_compressed_frame_jpeg_baseline_round_trip() {
        let width = 4u32;
        let height = 4u32;
        // Values span [20, 225] to exercise the full 8-bit range.
        let original: Vec<u8> = vec![
            50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpeg_baseline.dcm");
        write_jpeg_dicom_file(&path, width, height, &original);

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG Baseline");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );

        // All decoded values must lie in [0, 255].
        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // Each decoded value must be within JPEG tolerance of the original.
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 16.0,
            "JPEG decode error {max_error} exceeds analytical tolerance of 16.0 intensity units \
             (Q75: DC≤4 + AC(1,0)≤3 + AC(0,1)≤3 + AC(1,1)≤3 + higher-order margin = 16)"
        );
    }

    /// Rescale invariant: Output[i] = raw_sample[i] × slope + intercept.
    ///
    /// Uses a uniform 4×4 patch to isolate the rescale from JPEG spatial
    /// quantization effects. The decoded raw sample for a uniform patch is
    /// constant; the scaled output must equal raw × slope + intercept.
    #[test]
    fn test_decode_compressed_frame_rescale_contract() {
        let width = 4u32;
        let height = 4u32;
        // Uniform value minimises intra-patch JPEG quantization variation.
        let pixels: Vec<u8> = vec![128u8; 16];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpeg_rescale.dcm");
        write_jpeg_dicom_file(&path, width, height, &pixels);

        let obj = dicom::object::open_file(&path).expect("open_file");
        let base =
            decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0).expect("identity rescale decode");
        let scaled =
            decode_compressed_frame(&obj, 0, 8, 0, 2.0, 10.0).expect("slope=2 intercept=10 decode");

        assert_eq!(base.len(), 16, "base must have 16 elements");
        assert_eq!(scaled.len(), 16, "scaled must have 16 elements");

        // For each pixel: scaled[i] == base[i] * 2.0 + 10.0 (within float epsilon).
        for (i, (&b, &s)) in base.iter().zip(scaled.iter()).enumerate() {
            let expected = b * 2.0 + 10.0;
            assert!(
                (s - expected).abs() < 0.01,
                "pixel[{i}]: rescale invariant violated: got {s}, expected {b} × 2.0 + 10.0 = {expected}"
            );
        }
    }

    /// Build and write a minimal JPEG XL Lossless DICOM Part 10 file.
    ///
    /// Pixel data is JXL-encoded losslessly using `zune-jpegxl` (ISO 18181-1 modular path)
    /// and encapsulated as a single fragment per DICOM PS3.5 §A.4.
    ///
    /// # Parameters
    /// - `path`: destination file path.
    /// - `width`, `height`: image dimensions in pixels.
    /// - `pixels_u8`: flat row-major 8-bit grayscale values, length = `width × height`.
    fn write_jxl_lossless_dicom_file(
        path: &std::path::Path,
        width: u32,
        height: u32,
        pixels_u8: &[u8],
    ) {
        assert_eq!(
            pixels_u8.len(),
            (width * height) as usize,
            "pixels_u8 length must equal width × height"
        );

        // JXL-encode losslessly via zune-jpegxl.
        use zune_core::bit_depth::BitDepth;
        use zune_core::colorspace::ColorSpace;
        use zune_core::options::EncoderOptions;
        use zune_jpegxl::JxlSimpleEncoder;
        let options = EncoderOptions::new(
            width as usize,
            height as usize,
            ColorSpace::Luma,
            BitDepth::Eight,
        );
        let encoder = JxlSimpleEncoder::new(pixels_u8, options);
        let jxl_bytes = encoder.encode().expect("JXL encode failed");

        // Encapsulate as single fragment per DICOM PS3.5 §A.4.
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jxl_bytes]);
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
            PrimitiveValue::from("2.25.99999911"),
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
            PrimitiveValue::from("2.25.99999912"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999913"),
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
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(7u16),
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
                    .media_storage_sop_instance_uid("2.25.99999911")
                    .transfer_syntax("1.2.840.10008.1.2.4.110"),
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(path).expect("write_to_file failed");
    }

    /// JPEG XL Lossless round-trip: encode known pixel values, decode via codec, verify
    /// exact pixel equality (lossless invariant: no information loss).
    ///
    /// Mathematical justification:
    /// JXL Lossless uses the modular codec path (ISO 18181-1 §9) which is provably lossless.
    /// Given integer input samples S[i] ∈ [0, 255]:
    ///   Encode: JXL_Lossless(S) → bitstream B
    ///   Decode: JXL_Decode(B) → S' where S'[i] = S[i] for all i.
    /// Max error = max|S[i] - S'[i]| = 0.
    #[test]
    fn test_decode_compressed_frame_jxl_lossless_round_trip() {
        let width = 4u32;
        let height = 4u32;
        // Values span [20, 225] to exercise the full 8-bit range.
        let original: Vec<u8> = vec![
            50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jxl_lossless.dcm");
        write_jxl_lossless_dicom_file(&path, width, height, &original);

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG XL Lossless");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );

        // JXL Lossless: every decoded value must lie in [0, 255].
        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // JXL Lossless invariant: per-pixel error must be exactly 0.
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_error, 0.0,
            "JPEG XL Lossless decode error {max_error} must be exactly 0 \
             (JXL modular path preserves integer sample values exactly)"
        );
    }

    /// JPEG Extended round-trip: encode known pixel values with the JPEG codec, declare
    /// Transfer Syntax as JPEG Extended (1.2.840.10008.1.2.4.51), decode via codec, verify
    /// each decoded value is within JPEG quantization tolerance of the original.
    ///
    /// Mathematical justification:
    /// JPEG Extended (Process 2 & 4) uses the same DCT + quantization architecture as
    /// Baseline (Process 1) but supports 12-bit samples. For 8-bit input encoded with
    /// jpeg-encoder at default quality, the same DC/AC quantization bounds apply:
    ///   DC  (0,0): Q = 8  → ≤ 4 per pixel
    ///   AC  (1,0): Q = 6  → ≤ 3 per pixel
    ///   AC  (0,1): Q = 6  → ≤ 3 per pixel
    ///   AC  (1,1): Q = 6  → ≤ 3 per pixel
    /// Sum = 13; tolerance set to 16 (next integer power-of-2 ≥ 13).
    /// The codec registered for JPEG Extended uses the same `jpeg-decoder` path as Baseline.
    #[test]
    fn test_decode_compressed_frame_jpeg_extended_round_trip() {
        let width = 4u32;
        let height = 4u32;
        // Values span [20, 225] to exercise the full 8-bit range.
        let original: Vec<u8> = vec![
            50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpeg_extended.dcm");

        // Build DICOM file with JPEG Baseline–encoded pixel data declared under
        // JPEG Extended TS UID (1.2.840.10008.1.2.4.51).
        // The `jpeg-decoder` crate handles both SOF0 (Baseline) and SOF1 (Extended)
        // frames; a SOF0 frame is valid input for the Extended codec registered under .51.
        {
            use image::{DynamicImage, GrayImage};
            let gray = GrayImage::from_raw(width, height, original.clone())
                .expect("GrayImage::from_raw failed");
            let dyn_img = DynamicImage::ImageLuma8(gray);
            let mut jpeg_bytes: Vec<u8> = Vec::new();
            let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Jpeg)
                .expect("JPEG encode failed");
            drop(cursor);

            let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jpeg_bytes]);
            let pfs: PixelFragmentSequence<Vec<u8>> =
                PixelFragmentSequence::new_fragments(fragments);

            let mut obj = InMemDicomObject::new_empty();
            obj.put(DataElement::new(
                Tag(0x0008, 0x0016),
                VR::UI,
                PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
            ));
            obj.put(DataElement::new(
                Tag(0x0008, 0x0018),
                VR::UI,
                PrimitiveValue::from("2.25.99999921"),
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
                PrimitiveValue::from("2.25.99999922"),
            ));
            obj.put(DataElement::new(
                Tag(0x0020, 0x000E),
                VR::UI,
                PrimitiveValue::from("2.25.99999923"),
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
                PrimitiveValue::from(8u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0101),
                VR::US,
                PrimitiveValue::from(8u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0102),
                VR::US,
                PrimitiveValue::from(7u16),
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
                        .media_storage_sop_instance_uid("2.25.99999921")
                        .transfer_syntax("1.2.840.10008.1.2.4.51"), // JPEG Extended
                )
                .expect("FileMetaTableBuilder failed");
            file_obj.write_to_file(&path).expect("write_to_file failed");
        }

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG Extended");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );
        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 16.0,
            "JPEG Extended decode error {max_error} exceeds analytical tolerance of 16.0 \
             (Q75: DC≤4 + AC(1,0)≤3 + AC(0,1)≤3 + AC(1,1)≤3 + higher-order margin = 16)"
        );
    }

    /// DICOM RLE Lossless PackBits encoder per DICOM PS3.5 Annex G.3.1.
    ///
    /// Specification:
    /// - Header byte h encodes run type:
    ///   - h ∈ [0, 127]: literal run of (h + 1) bytes.
    ///   - h ∈ [129, 255]: repeat run of (257 − h) copies of the following byte.
    ///   - h = 128 (0x80): no-op.
    /// - Repeat count ∈ [2, 128], header = 257 − count ∈ [129, 255].
    /// - Literal count ∈ [1, 128], header = count − 1 ∈ [0, 127].
    /// - Segment padded to even length per PS3.5 Annex G.3.2.
    fn packbits_encode(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() + data.len() / 128 + 2);
        let mut i = 0;
        while i < data.len() {
            // Count maximal repeat run from position i.
            let mut repeat = 1usize;
            while i + repeat < data.len() && data[i + repeat] == data[i] && repeat < 128 {
                repeat += 1;
            }
            if repeat >= 2 {
                // Repeat run: header = 257 − repeat ∈ [129, 255].
                out.push((257 - repeat) as u8);
                out.push(data[i]);
                i += repeat;
                continue;
            }
            // Count literal run: advance while consecutive pair is not a repeat start.
            let lit_start = i;
            let mut lit = 1usize;
            while i + lit < data.len() && lit < 128 {
                // Stop before a pair that will become a repeat run.
                if i + lit + 1 < data.len() && data[i + lit] == data[i + lit + 1] {
                    break;
                }
                lit += 1;
            }
            // Literal run: header = lit − 1 ∈ [0, 127].
            out.push((lit - 1) as u8);
            out.extend_from_slice(&data[lit_start..lit_start + lit]);
            i += lit;
        }
        // Pad to even length per DICOM PS3.5 Annex G.3.2.
        if out.len() % 2 != 0 {
            out.push(0x00);
        }
        out
    }

    /// Assemble a single-segment DICOM RLE Lossless fragment for 8-bit single-channel data.
    ///
    /// Per DICOM PS3.5 Annex G.4.1, the fragment layout is:
    ///   [RLE Header: 64 bytes] [Segment 0: PackBits-encoded pixel bytes]
    ///
    /// RLE Header (16 × uint32 LE):
    ///   header[0] = 1  (one segment)
    ///   header[1] = 64 (segment 0 offset = header size)
    ///   header[2..15] = 0 (unused)
    fn build_rle_fragment_8bit(pixels: &[u8]) -> Vec<u8> {
        let segment = packbits_encode(pixels);
        const HEADER_BYTES: usize = 64;
        let mut header = [0u32; 16];
        header[0] = 1;
        header[1] = HEADER_BYTES as u32;
        let mut out = Vec::with_capacity(HEADER_BYTES + segment.len());
        for &w in &header {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&segment);
        out
    }

    /// RLE Lossless round-trip: encode known pixel values using DICOM RLE (PS3.5 Annex G),
    /// decode via the registered codec, verify exact pixel equality.
    ///
    /// Mathematical justification:
    /// DICOM RLE Lossless uses the PackBits algorithm applied per byte plane. PackBits is a
    /// lossless compression scheme: for any input S, decode(encode(S)) = S exactly.
    /// Therefore max|decoded[i] − original[i]| = 0 for any integer sample sequence.
    ///
    /// The native decoder (`decode_rle_lossless_frame`) implements PS3.5 Annex G correctly
    /// for all pixel values. All N=16 pixels are encoded and the native decoder recovers
    /// all N pixels exactly — no offset-compensation or restricted pixel values required.
    ///
    /// The encoded slice exercises both PackBits run types:
    /// - [0, 50, 50, 50]         — literal followed by 3-repetition repeat run.
    /// - [75, 80, 85, 90]        — 4-element literal run.
    /// - [100, 100, 100, 100]    — 4-repetition repeat run.
    /// - [120, 130, 140, 150]    — 4-element literal run.
    #[test]
    fn test_decode_compressed_frame_rle_lossless_round_trip() {
        let width = 4u32;
        let height = 4u32;
        let original: Vec<u8> = vec![
            0, 50, 50, 50, // literal + repeat run
            75, 80, 85, 90, // literal run: 4 distinct values
            100, 100, 100, 100, // repeat run: 4× 100
            120, 130, 140, 150, // literal run: 4 distinct values
        ];

        // Encode all N=16 pixels. The native decoder recovers all pixels exactly.
        let rle_fragment = build_rle_fragment_8bit(&original);
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![rle_fragment]);
        let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_rle_lossless.dcm");

        {
            let mut obj = InMemDicomObject::new_empty();
            obj.put(DataElement::new(
                Tag(0x0008, 0x0016),
                VR::UI,
                PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
            ));
            obj.put(DataElement::new(
                Tag(0x0008, 0x0018),
                VR::UI,
                PrimitiveValue::from("2.25.99999931"),
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
                PrimitiveValue::from("2.25.99999932"),
            ));
            obj.put(DataElement::new(
                Tag(0x0020, 0x000E),
                VR::UI,
                PrimitiveValue::from("2.25.99999933"),
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
                PrimitiveValue::from(8u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0101),
                VR::US,
                PrimitiveValue::from(8u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0102),
                VR::US,
                PrimitiveValue::from(7u16),
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
                        .media_storage_sop_instance_uid("2.25.99999931")
                        .transfer_syntax("1.2.840.10008.1.2.5"), // RLE Lossless
                )
                .expect("FileMetaTableBuilder failed");
            file_obj.write_to_file(&path).expect("write_to_file failed");
        }

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for RLE Lossless");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );
        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // RLE Lossless invariant: PackBits is lossless, so every decoded value must exactly
        // equal the original integer sample.
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_error, 0.0,
            "RLE Lossless decode error {max_error} must be exactly 0 (PackBits is lossless per PS3.5 G.3.1)"
        );
    }

    /// RLE Lossless round-trip with unrestricted pixel values — `pixel[0] = 42` (≠ 0).
    ///
    /// Mathematical justification:
    /// PackBits is lossless: `decode(encode(S)) = S` for all `S: &[u8]`. The native
    /// `decode_rle_lossless_frame` decoder implements this correctly for all pixel values,
    /// including `pixel[0] ≠ 0`.
    ///
    /// Upstream failure mode (dicom-transfer-syntax-registry v0.8.2):
    /// The upstream decoder forces `dst[0] = 0` regardless of encoded content, so for
    /// `pixel[0] = 42`, the upstream decoder would produce `decoded[0] = 0.0 ≠ 42.0`.
    /// This test would FAIL with the upstream decoder and MUST pass with the native decoder.
    ///
    /// Formal invariant: `∀i ∈ [0, N−1]: decoded[i] = original[i]`
    ///   ⟹ `max|decoded[i] − original[i]| = 0`
    #[test]
    fn test_decode_compressed_frame_rle_lossless_unrestricted_round_trip() {
        let width = 4u32;
        let height = 4u32;
        // pixel[0] = 42: non-zero — this would be silently corrupted to 0 by the
        // upstream dicom-transfer-syntax-registry v0.8.2 RLE decoder.
        let original: Vec<u8> = vec![
            42, 50, 50, 50, // pixel[0]=42 (non-zero), then repeat run: 3×50
            75, 80, 85, 90, // literal run: 4 distinct values
            100, 100, 100, 100, // repeat run: 4×100
            120, 130, 140, 150, // literal run: 4 distinct values
        ];

        // Encode ALL N=16 pixels (no offset compensation needed with the native decoder).
        let rle_fragment = build_rle_fragment_8bit(&original);
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![rle_fragment]);
        let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_rle_unrestricted.dcm");

        {
            let mut obj = InMemDicomObject::new_empty();
            obj.put(DataElement::new(
                Tag(0x0008, 0x0016),
                VR::UI,
                PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
            ));
            obj.put(DataElement::new(
                Tag(0x0008, 0x0018),
                VR::UI,
                PrimitiveValue::from("2.25.99999941"),
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
                PrimitiveValue::from("2.25.99999942"),
            ));
            obj.put(DataElement::new(
                Tag(0x0020, 0x000E),
                VR::UI,
                PrimitiveValue::from("2.25.99999943"),
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
                PrimitiveValue::from(8u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0101),
                VR::US,
                PrimitiveValue::from(8u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0102),
                VR::US,
                PrimitiveValue::from(7u16),
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
                        .media_storage_sop_instance_uid("2.25.99999941")
                        .transfer_syntax("1.2.840.10008.1.2.5"), // RLE Lossless
                )
                .expect("FileMetaTableBuilder failed");
            file_obj.write_to_file(&path).expect("write_to_file failed");
        }

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for RLE Lossless with native decoder");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );

        // Critical: pixel[0] = 42 must NOT be corrupted to 0.
        // The upstream dicom-transfer-syntax-registry v0.8.2 decoder forces dst[0] = 0
        // for 8-bit grayscale; the native decoder must return 42.0 exactly.
        assert_eq!(
            decoded[0], 42.0f32,
            "pixel[0] must be 42.0; upstream decoder forces this to 0.0 (off-by-one write start)"
        );

        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // RLE Lossless exact-fidelity invariant: PackBits is lossless.
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_error,
            0.0,
            "RLE Lossless native decode error {max_error} must be exactly 0 (lossless per PS3.5 G.3.1)"
        );
    }

    /// Build and write a minimal JPEG-LS Lossless DICOM Part 10 file.
    ///
    /// Pixel data is encoded losslessly (near-lossless parameter = 0) using the
    /// `charls` crate (CharLS C++ JPEG-LS implementation, ISO 14495-1 / ITU-T T.87).
    /// The bitstream is encapsulated as a single fragment per DICOM PS3.5 §A.4.
    ///
    /// # Parameters
    /// - `path`: destination file path.
    /// - `width`, `height`: image dimensions in pixels.
    /// - `pixels_u8`: flat row-major 8-bit grayscale values, length = `width × height`.
    fn write_jpegls_lossless_dicom_file(
        path: &std::path::Path,
        width: u32,
        height: u32,
        pixels_u8: &[u8],
    ) {
        assert_eq!(
            pixels_u8.len(),
            (width * height) as usize,
            "pixels_u8 length must equal width × height"
        );

        // Encode losslessly with CharLS (near = 0 ⟹ lossless).
        let frame_info = charls::FrameInfo {
            width,
            height,
            bits_per_sample: 8,
            component_count: 1,
        };
        let mut codec = charls::CharLS::default();
        let jls_bytes = codec
            .encode(frame_info, 0, pixels_u8)
            .expect("CharLS encode failed");

        // Encapsulate as single fragment per DICOM PS3.5 §A.4.
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jls_bytes]);
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
            PrimitiveValue::from("2.25.99999931"),
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
            PrimitiveValue::from("2.25.99999932"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999933"),
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
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(7u16),
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
                    .media_storage_sop_instance_uid("2.25.99999931")
                    .transfer_syntax("1.2.840.10008.1.2.4.80"),
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(path).expect("write_to_file failed");
    }

    /// JPEG-LS Lossless round-trip: encode known pixel values, decode via codec, verify
    /// exact pixel equality (lossless invariant: no information loss per ISO 14495-1).
    ///
    /// Mathematical justification:
    /// JPEG-LS lossless (near = 0) uses a near-lossless coder with NEAR = 0, which implies
    /// the reconstructed sample value s' satisfies |s' − s| ≤ NEAR = 0, i.e., exact
    /// reconstruction per ISO 14495-1 §A.2:
    ///   Encode: JLS_Lossless(S, NEAR=0) → bitstream B
    ///   Decode: JLS_Decode(B, NEAR=0) → S' where S'[i] = S[i] for all i.
    /// Max error = max|S[i] − S'[i]| = 0.
    ///
    /// Pixel set includes boundary values (0, 255) and interior values to span [0, 255].
    #[test]
    fn test_decode_compressed_frame_jpegls_lossless_round_trip() {
        let width = 4u32;
        let height = 4u32;
        // Values span [0, 255] including exact boundaries and interior samples.
        // Includes pixel[0] = 0 and pixel[15] = 255 to exercise boundary conditions.
        let original: Vec<u8> = vec![
            0, 42, 85, 127, 128, 170, 200, 225, 50, 100, 150, 199, 64, 96, 128, 255,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpegls_lossless.dcm");
        write_jpegls_lossless_dicom_file(&path, width, height, &original);

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG-LS Lossless");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );

        // All decoded values must lie in [0, 255].
        for (i, &v) in decoded.iter().enumerate() {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded[{i}] = {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // JPEG-LS Lossless invariant: per-pixel error must be exactly 0 (ISO 14495-1, NEAR=0).
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_error, 0.0,
            "JPEG-LS Lossless decode error {max_error} must be exactly 0 \
             (ISO 14495-1: NEAR=0 ⟹ |s'-s| ≤ NEAR = 0)"
        );

        // Verify each sample individually to catch index-specific faults.
        for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig as f32, dec,
                "pixel[{i}]: expected {orig}, got {dec} — JPEG-LS lossless must preserve all sample values"
            );
        }
    }

    /// JPEG-LS Near-Lossless round-trip: encode known pixel values with NEAR=2, decode via
    /// codec, verify per-pixel reconstruction error ≤ NEAR per ISO 14495-1.
    ///
    /// Mathematical justification:
    /// JPEG-LS near-lossless (NEAR = 2) guarantees |s' − s| ≤ 2 for all pixels per
    /// ISO 14495-1 §A.2. Tolerance set to exactly 2.0 (the analytical bound).
    #[test]
    fn test_decode_compressed_frame_jpegls_near_lossless_round_trip() {
        let width = 4u32;
        let height = 4u32;
        // Values span [10, 245] to avoid boundary clamping effects with NEAR=2.
        let original: Vec<u8> = vec![
            10, 50, 100, 150, 200, 245, 30, 80, 130, 180, 220, 60, 110, 160, 210, 40,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpegls_nearlossless.dcm");

        // Encode with near = 2 but declare TS as JPEG-LS Near-Lossless (.81).
        let frame_info = charls::FrameInfo {
            width,
            height,
            bits_per_sample: 8,
            component_count: 1,
        };
        let mut codec = charls::CharLS::default();
        let jls_bytes = codec
            .encode(frame_info, 2, &original)
            .expect("CharLS near-lossless encode failed");

        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jls_bytes]);
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
            PrimitiveValue::from("2.25.99999941"),
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
            PrimitiveValue::from("2.25.99999942"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999943"),
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
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(7u16),
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
                    .media_storage_sop_instance_uid("2.25.99999941")
                    .transfer_syntax("1.2.840.10008.1.2.4.81"), // JPEG-LS Near-Lossless
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(&path).expect("write_to_file failed");

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG-LS Near-Lossless");

        assert_eq!(decoded.len(), 16, "decoded pixel count must equal 16");

        // ISO 14495-1: NEAR=2 ⟹ |s'[i] − s[i]| ≤ 2 for all i.
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &dec)| (orig as f32 - dec).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 2.0,
            "JPEG-LS Near-Lossless decode error {max_error} exceeds analytical bound of 2 \
             (ISO 14495-1: NEAR=2 ⟹ |s'-s| ≤ 2)"
        );
    }
}
