//! Native JPEG 2000 (ISO 15444-1) decoder for DICOM encapsulated frames.
//!
//! # Architecture
//! This module provides a **pure-Rust** implementation of JPEG 2000 decoding,
//! eliminating all C/FFI dependencies (`jpeg2k`, `openjp2`, `openjpeg-sys`).
//!
//! Sub-modules:
//! - `marker`     – ISO 15444-1 marker constants and byte-read utilities.
//! - `codestream` – Main-header parser (SIZ, COD, QCD, SOT).
//! - `mq_coder`   – MQ arithmetic coder — encoder and decoder (Annex C).
//! - `ebcot`      – EBCOT tier-1 encoder and decoder (Annex D).
//! - `packet`     – Tier-2 packet encoder and decoder (Annex B).
//! - `wavelet`    – Forward and inverse 5/3 reversible DWT (Annex F).
//! - `wavelet_9_7`– Forward and inverse 9/7 irreversible DWT (Annex F, lossy).
//! - `quantization`– Scalar dead-zone quantization for the 9/7 path (Annex E).
//! - `subband`    – Mallat subband geometry (Annex B.5).
//! - `tag_tree`   – Quad-tree inclusion/MSB coding (Annex B.10.2).
//! - `image`      – Full codestream decoder and DICOM pixel extractor.
//! - [`encoder`]  – Pure-Rust encoder (produces conformant codestreams).
//!
//! # Specification (ISO 15444-1 / DICOM PS3.5)
//! DICOM JPEG 2000 encapsulates a raw J2K codestream (not a JP2 file wrapper):
//! - Transfer Syntax 1.2.840.10008.1.2.4.90: JPEG 2000 Lossless Only.
//! - Transfer Syntax 1.2.840.10008.1.2.4.91: JPEG 2000 lossy or lossless.
//!
//! # Current limitations
//! - One precinct per resolution/band (no precinct partitioning; code-blocks
//!   are 64×64 within each subband).
//! - Native pixel extraction currently accepts one grayscale component, LRCP
//!   progression without POC or tile coding overrides, inline packet headers,
//!   default precinct and code-block styles, 64×64 nominal code-blocks, one
//!   tile-part per tile, and no multiple-component transform. Unsupported
//!   profiles are rejected before output allocation rather than decoded with
//!   incorrect packet traversal.
//! - Lossy 9/7 irreversible encode and decode are supported (scalar quantization,
//!   unit-step near-lossless encoder); a rate-controlled quality knob is pending.
//!
//! # Untrusted-input bounds
//! SIZ image/tile geometry and SOT tile-part fields are validated before packet
//! decode. Tile buffers use the image-domain intersection defined by T.800
//! Annex B.3 rather than the full reference tile, and both pixel and
//! sample counts share a fixed allocation cap. Tile headers are parsed by marker
//! boundaries before output allocation; every segment must fit its byte extent,
//! EOC is mandatory, every declared tile must be present, and every expected
//! LRCP packet header must be consumed.
//!
//! # Interop validation
//! The reversible (5/3 lossless) path is validated against a captured OpenJPEG
//! 2.5.4 corpus in both directions: 72 OpenJPEG-produced streams decode exactly,
//! 54 RITK streams remain byte-identical to outputs accepted and decoded exactly
//! by OpenJPEG, and ten MQ/EBCOT patterns match tile bodies byte-for-byte. The
//! irreversible (9/7 lossy) corpus covers 54 geometry, precision, and resolution
//! combinations. RITK reconstruction must remain within 1 dB PSNR of each
//! captured OpenJPEG decoder baseline, validating inverse lifting, QCD step-size
//! parsing, and dequantization. Tests execute only the RITK-native Rust codec.
//! Internal round-trips additionally cover lossy encode→decode.
//!
//! Reconstruction (ISO 15444-1 §E.1.1.2) is source-aware: a transformed subband
//! coefficient (`num_decomp_levels ≥ 1`) is a continuous value with sub-step
//! uncertainty, reconstructed at the interval midpoint (the standard half-step at
//! full decode); a zero-level LL band carries the original integer samples
//! captured losslessly (`Δ = 1`), reconstructed exactly with no bias — recovering
//! them bit-for-bit, where a fixed half-step would offset every sample by `Δ/2`.

pub(crate) mod codestream;
pub(crate) mod ebcot;
pub mod encoder;
pub(crate) mod image;
pub(crate) mod marker;
pub(crate) mod mq_coder;
pub(crate) mod packet;
pub(crate) mod quantization;
pub(crate) mod subband;
pub(crate) mod tag_tree;
pub(crate) mod wavelet;
pub(crate) mod wavelet_9_7;

use anyhow::{bail, Result};

use crate::PixelLayout;
use image::{decode_j2k_fragment, is_soc};

/// J2K Start of Codestream marker (ISO 15444-1 §A.3): bytes `0xFF 0x4F`.
#[allow(dead_code)] // Tested via soc_marker_constant_matches_iso_15444_1
pub(crate) const SOC: u16 = 0xFF4F;

/// JPEG / JFIF Start of Image marker (0xFFD8), distinct from SOC.
#[allow(dead_code)] // Tested via soi_constant_matches_jpeg_start_of_image
pub(crate) const SOI: u16 = 0xFFD8;

/// Decode a DICOM-encapsulated JPEG 2000 J2K codestream fragment.
///
/// # Arguments
/// - `fragment`: raw bytes of the encapsulated pixel data item.
/// - `layout`: pixel geometry and DICOM rescale parameters.
///
/// # Errors
/// Returns an error if:
/// - `fragment` does not begin with the SOC marker (0xFF4F).
/// - SIZ or SOT geometry is invalid, outside the supported allocation bound,
///   or identifies a tile outside the image's tile grid.
/// - the stream is not single-component grayscale LRCP without coding
///   overrides, packed packet headers, multi-part tiles, or MCT.
/// - packet or coefficient decoding fails.
/// - a marker segment or packet is truncated, EOC or a declared tile is
///   missing, or decoded component metadata does not match `layout`.
pub fn decode_jpeg2000_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    if !is_jpeg2000_codestream(fragment) {
        bail!(
            "JPEG 2000 fragment does not begin with SOC marker 0xFF4F \
             (first 2 bytes: {:02X?})",
            &fragment[..fragment.len().min(2)]
        );
    }
    decode_j2k_fragment(fragment, layout)
}

/// Returns `true` if `fragment` begins with the J2K SOC marker (`0xFF 0x4F`).
///
/// A bare DICOM JPEG 2000 codestream always starts with SOC (ISO 15444-1 §A.3).
#[inline]
pub(crate) fn is_jpeg2000_codestream(fragment: &[u8]) -> bool {
    is_soc(fragment)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
