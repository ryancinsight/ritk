//! ISO 15444-1 marker codes and big-endian byte-reading helpers.
//!
//! All JPEG 2000 markers are 2-byte codes with high byte 0xFF (§A.4–§A.9).
//! Constants are named exactly as in the standard.

#![allow(dead_code)] // Marker constants are spec-mandated; used selectively by parsers.

/// Start of Codestream (§A.4.1).
pub const SOC: u16 = 0xFF4F;
/// Image and tile size (§A.5.1).
pub const SIZ: u16 = 0xFF51;
/// Coding style default (§A.6.1).
pub const COD: u16 = 0xFF52;
/// Coding style component (§A.6.2).
pub const COC: u16 = 0xFF53;
/// Quantization default (§A.6.4).
pub const QCD: u16 = 0xFF5C;
/// Quantization component (§A.6.5).
pub const QCC: u16 = 0xFF5D;
/// Region of interest (§A.6.6).
pub const RGN: u16 = 0xFF5E;
/// Progression order change (§A.6.7).
pub const POC: u16 = 0xFF5F;
/// Tile-part lengths (§A.7.1).
pub const TLM: u16 = 0xFF55;
/// Packed packet headers, main header (§A.7.4).
pub const PPM: u16 = 0xFF60;
/// Packed packet headers, tile-part header (§A.7.4).
pub const PPT: u16 = 0xFF61;
/// Component registration (§A.9.1).
pub const CRG: u16 = 0xFF63;
/// Comment (§A.9.2).
pub const COM: u16 = 0xFF64;
/// Start of Tile-part (§A.4.2).
pub const SOT: u16 = 0xFF90;
/// Start of Data (§A.4.3).
pub const SOD: u16 = 0xFF93;
/// End of Codestream (§A.4.4).
pub const EOC: u16 = 0xFFD9;

// ── Byte-reading helpers ──────────────────────────────────────────────────────

/// Read a big-endian `u8` from `data[pos]`.
#[inline]
pub fn read_u8(data: &[u8], pos: usize) -> anyhow::Result<u8> {
    data.get(pos)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("J2K: read_u8 at {pos} beyond {}-byte buffer", data.len()))
}

/// Read a big-endian `u16` from `data[pos]`.
#[inline]
pub fn read_u16(data: &[u8], pos: usize) -> anyhow::Result<u16> {
    data.get(pos..pos + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .ok_or_else(|| anyhow::anyhow!("J2K: read_u16 at {pos} beyond {}-byte buffer", data.len()))
}

/// Read a big-endian `u32` from `data[pos]`.
#[inline]
pub fn read_u32(data: &[u8], pos: usize) -> anyhow::Result<u32> {
    data.get(pos..pos + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| anyhow::anyhow!("J2K: read_u32 at {pos} beyond {}-byte buffer", data.len()))
}
