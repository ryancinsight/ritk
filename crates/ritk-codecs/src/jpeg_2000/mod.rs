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
//! # High-throughput JPEG 2000 and packet organization
//! DICOM PS3.5 §10.18 specifies the Part 15 lossless (.201), lossless RPCL
//! (.202) and unrestricted (.203) syntaxes. The decoder supports 1–38 stored
//! bits, all five packet progressions, tiled and multi-part images, packed
//! packet headers, progression changes and component/tile coding overrides.
//! RITK reconstructs ordinary scalar tiles directly, including reversible
//! HT code-blocks. General packet organizations and component transforms use
//! `j2k-native` with native-width output. The HT entropy kernel and the
//! unsigned 16-bit grayscale HT writer use `htj2k`.
//!
//! # Frame boundaries
//! The caller's read budget bounds the encoded fragment and decoded output.
//! A structural pass validates SIZ, header marker lengths, tile-part indices
//! and lengths, complete tile coverage, and EOC before reconstruction.
//! A final zero-length Psot extends to EOC. DICOM may pad an odd-length
//! fragment with a single zero byte. The progressive lossless profile
//! requires RPCL order, a thumbnail axis of at most 64 samples and a TLM
//! index consistent with the actual tile-parts.
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
mod decode;
mod decode_error;
pub(crate) mod ebcot;
pub mod encoder;
mod extended;
mod ht_encoder;
pub use ht_encoder::encode_grayscale_htj2k;
pub(crate) mod image;
pub(crate) mod marker;
pub(crate) mod mq_coder;
mod organization;
pub(crate) mod packet;
pub(crate) mod quantization;
mod read_limits;
pub(crate) mod subband;
pub(crate) mod tag_tree;
mod tile_lengths;
pub(crate) mod wavelet;
pub(crate) mod wavelet_9_7;

pub use decode::{
    decode_jpeg2000_fragment, decode_jpeg2000_stored_fragment, decode_jpeg2000_stored_with_profile,
    decode_jpeg2000_with_profile, grayscale_component_precision,
};
pub use decode_error::Jpeg2000DecodeError;
pub use organization::{HtJ2kProfile, Jpeg2000Profile};

/// J2K Start of Codestream marker (ISO 15444-1 §A.3): bytes `0xFF 0x4F`.
#[cfg(test)]
pub(crate) const SOC: u16 = 0xFF4F;

/// JPEG / JFIF Start of Image marker (0xFFD8), distinct from SOC.
#[cfg(test)]
pub(crate) const SOI: u16 = 0xFFD8;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod profile_tests;
#[cfg(test)]
mod tests;
