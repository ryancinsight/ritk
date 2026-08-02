//! Vendor-specific DICOM private-block extraction for diffusion metadata.
//!
//! When standard DICOM tags `(0018,9087)` Diffusion b-value and `(0018,9089)`
//! Diffusion Gradient Orientation are absent, this module attempts to extract
//! the equivalent metadata from manufacturer-specific private blocks.
//!
//! Currently supported:
//!
//! * **Siemens CSA header** — binary blob in `(0029,1020)` (series) or
//!   `(0029,1010)` (image).  Keys `B_value` and `DiffusionGradientDirection`
//!   are extracted with the same semantics as the standard tags.
//!
//! # Extension
//!
//! GE and Philips private groups are recognised as present but return
//! `Ok(None)`.  Adding a new vendor requires:
//!
//! 1. A private element lookup (via [`crate::attribute::DicomAttributeRead::optional_bytes`]).
//! 2. A format-specific decoder in this module.
//! 3. A new arm in [`try_vendor_pair`].

use anyhow::{bail, Context, Result};
use ritk_spatial::Vector;

use crate::attribute::{DicomAttributeRead, DicomTag};

/// DICOM private element carrying the Siemens CSA series header.
///
/// Private creator: `SIEMENS CSA HEADER` in `(0029,1008)`.
const SIEMENS_CSA_SERIES: DicomTag = DicomTag::new(0x0029, 0x1020);

/// Fallback Siemens CSA element at the image level.
const SIEMENS_CSA_IMAGE: DicomTag = DicomTag::new(0x0029, 0x1010);

// ── Public entry point ───────────────────────────────────────────────────

/// Attempt to extract a diffusion (b-value, gradient-direction) pair from
/// vendor private blocks when the standard top-level elements are absent.
///
/// Returns `Ok(None)` when no recognised vendor metadata is present.
pub(super) fn try_vendor_pair(
    object: &impl DicomAttributeRead,
) -> Result<Option<(f64, Vector<3>)>> {
    if let Some(pair) = try_siemens_csa(object)? {
        return Ok(Some(pair));
    }
    // GE and Philips are recognised as present but not yet decoded.
    // Add new arm here when a vendor decoder is implemented.
    Ok(None)
}

// ── Siemens CSA binary header ────────────────────────────────────────────

/// Siemens SV10 CSA magic bytes.
const CSA_SV10_MAGIC: &[u8; 4] = b"SV10";

/// A single key-value pair parsed from a CSA element table.
#[derive(Debug)]
struct CsaEntry {
    name: String,
    data: Vec<u8>,
}

/// Read diffusion metadata from a Siemens CSA binary header.
///
/// Tries `(0029,1020)` (series level) first, then `(0029,1010)`
/// (image level).  The CSA blob must start with the `SV10` magic.
fn try_siemens_csa(object: &impl DicomAttributeRead) -> Result<Option<(f64, Vector<3>)>> {
    for tag in [SIEMENS_CSA_SERIES, SIEMENS_CSA_IMAGE] {
        let Some(blob) = object.optional_bytes(tag, "Siemens CSA header")? else {
            continue;
        };
        if let Some(pair) = parse_csa_blob(&blob) {
            return Ok(Some(pair));
        }
    }
    Ok(None)
}

/// Parse a complete CSA binary blob into a diffusion pair.
fn parse_csa_blob(blob: &[u8]) -> Option<(f64, Vector<3>)> {
    let entries = parse_csa_entries(blob).ok()?;

    let b_value = entries
        .iter()
        .find(|e| e.name == "B_value")
        .and_then(|e| decode_csa_float(&e.data));

    let direction = entries
        .iter()
        .find(|e| e.name == "DiffusionGradientDirection")
        .and_then(|e| decode_csa_triplet(&e.data));

    match (b_value, direction) {
        (Some(b), Some(dir)) => Some((b, Vector::new(dir))),
        // b=0 with no explicit direction is a valid b0 volume.
        (Some(b), None) if b == 0.0 => Some((b, Vector::new([0.0, 0.0, 0.0]))),
        _ => None,
    }
}

// ── CSA binary parser ────────────────────────────────────────────────────

/// Parse all key-value entries from a Siemens SV10 CSA binary blob.
fn parse_csa_entries(blob: &[u8]) -> Result<Vec<CsaEntry>> {
    if blob.len() < 16 || &blob[..4] != CSA_SV10_MAGIC {
        bail!("CSA header does not start with SV10 magic");
    }

    let n_tags = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
    // Sanity: a DICOM element is ≤ 4 GB, so n_tags must be reasonable.
    if n_tags == 0 {
        bail!("CSA header declares zero tags — malformed");
    }
    if n_tags > 10_000 {
        bail!("CSA header declares {n_tags} tags — unreasonable");
    }

    // Tag headers start at byte offset 16.
    let mut offset: usize = 16;
    let mut entries = Vec::with_capacity(n_tags);

    for _tag_index in 0..n_tags {
        if offset + 84 > blob.len() {
            bail!(
                "CSA tag header at offset {offset} extends past blob end ({})",
                blob.len()
            );
        }

        // Name — first 64 bytes, null-terminated.
        let name_bytes = &blob[offset..offset + 64];
        let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(64);
        let name = String::from_utf8_lossy(&name_bytes[..name_len]).into_owned();

        let vm = u32::from_le_bytes(blob[offset + 64..offset + 68].try_into().unwrap());
        let _vr = &blob[offset + 68..offset + 72]; // 4-char ASCII, e.g. "FD  "
        let _syngo_dt =
            u32::from_le_bytes(blob[offset + 72..offset + 76].try_into().unwrap());
        let n_items = u32::from_le_bytes(blob[offset + 76..offset + 80].try_into().unwrap())
            as usize;

        // Skip the 4-byte pad after the tag header.
        offset += 84;

        let mut data = Vec::new();
        for _ in 0..n_items {
            if offset + 4 > blob.len() {
                bail!("CSA item header at offset {offset} extends past blob end");
            }
            let item_len =
                u32::from_le_bytes(blob[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + item_len > blob.len() {
                bail!(
                    "CSA item data of {item_len} bytes at offset {offset} extends past blob end"
                );
            }
            // For simple values (no leading array count), just collect.
            data.extend_from_slice(&blob[offset..offset + item_len]);
            offset += item_len;

            // Align to 4-byte boundary.
            let remainder = offset % 4;
            if remainder != 0 {
                offset += 4 - remainder;
            }
        }

        // Only keep entries we actually need.
        if name == "B_value" || name == "DiffusionGradientDirection" {
            entries.push(CsaEntry { name, data });
        }

        let _ = vm; // used for future multi-valued decoding
    }

    Ok(entries)
}

// ── Value decoders ───────────────────────────────────────────────────────

/// Decode a single little-endian `f64` from a CSA element payload.
fn decode_csa_float(data: &[u8]) -> Option<f64> {
    if data.len() == 8 {
        let raw = data[..8].try_into().ok()?;
        let value = f64::from_le_bytes(raw);
        if value.is_finite() {
            return Some(value);
        }
    }
    // Try parsing as a decimal string (some Siemens versions store strings).
    decode_csa_float_string(data)
}

/// Decode a float from a CSA payload stored as an ASCII decimal string.
fn decode_csa_float_string(data: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(data).ok()?;
    let trimmed = text.trim_end_matches('\0').trim();
    trimmed.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Decode three `f64` values from a CSA element payload (gradient direction).
fn decode_csa_triplet(data: &[u8]) -> Option<[f64; 3]> {
    // 3 × f64 = 24 bytes
    if data.len() >= 24 {
        let x = f64::from_le_bytes(data[0..8].try_into().ok()?);
        let y = f64::from_le_bytes(data[8..16].try_into().ok()?);
        let z = f64::from_le_bytes(data[16..24].try_into().ok()?);
        if x.is_finite() && y.is_finite() && z.is_finite() {
            return Some([x, y, z]);
        }
    }
    // Fall back to whitespace-separated ASCII.
    let text = std::str::from_utf8(data).ok()?;
    let trimmed = text.trim_end_matches('\0').trim();
    let parts: Vec<f64> = trimmed
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    if parts.len() == 3 && parts.iter().all(|v| v.is_finite()) {
        return Some([parts[0], parts[1], parts[2]]);
    }
    None
}

#[cfg(test)]
mod tests;
