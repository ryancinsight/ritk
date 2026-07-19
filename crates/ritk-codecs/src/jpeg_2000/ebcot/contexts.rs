pub(crate) const CTX_ZC_BASE: usize = 0; // 9 significance contexts (0..8)
pub(crate) const CTX_SC_BASE: usize = 9; // 5 sign contexts (9..13)
pub(crate) const CTX_MR_BASE: usize = 14; // 3 magnitude-refinement contexts (14..16)
pub(crate) const CTX_UNI: usize = 17; // uniform
pub(crate) const CTX_AGG: usize = 18; // aggregation / run-length

/// Subband orientation, used to select the significance context function.
#[allow(dead_code)] // Hl and Hh needed when DWT support is added (J2K-DECODE-DWT)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubbandOrientation {
    /// LL (lowest frequency) or LH (horizontal high-pass, vertical low-pass).
    LlOrLh,
    /// HL (horizontal low-pass, vertical high-pass).
    Hl,
    /// HH (both high-pass).
    Hh,
}

// â”€â”€ Context helper functions (ISO 15444-1 Â§D.3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Significance context for LL / LH subbands (ISO 15444-1 Table D.1, columns H/V/D).
#[inline]
pub(crate) fn zc_ll_lh(h: u32, v: u32, d: u32) -> usize {
    CTX_ZC_BASE
        + match (h, v, d) {
            (2, _, _) => 8,
            (1, v, _) if v >= 1 => 7,
            (1, 0, d) if d >= 1 => 6,
            (1, 0, 0) => 5,
            (0, 2, _) => 4,
            (0, 1, _) => 3,
            (0, 0, d) if d >= 2 => 2,
            (0, 0, 1) => 1,
            _ => 0,
        }
}

/// Significance context for HL subband (H and V roles swapped).
#[inline]
pub(crate) fn zc_hl(h: u32, v: u32, d: u32) -> usize {
    zc_ll_lh(v, h, d)
}

/// Significance context for HH subband (ISO 15444-1 Table D.2).
#[inline]
pub(crate) fn zc_hh(h: u32, v: u32, d: u32) -> usize {
    CTX_ZC_BASE
        + match (d, h + v) {
            (d, _) if d >= 3 => 8,
            (2, hv) if hv >= 1 => 7,
            (2, _) => 6,
            (1, hv) if hv >= 2 => 5,
            (1, 1) => 4,
            (1, 0) => 3,
            (0, hv) if hv >= 2 => 2,
            (0, 1) => 1,
            _ => 0,
        }
}

/// Choose the significance context given orientation and neighbour counts.
#[inline]
pub(crate) fn zc_context(orient: SubbandOrientation, h: u32, v: u32, d: u32) -> usize {
    match orient {
        SubbandOrientation::LlOrLh => zc_ll_lh(h, v, d),
        SubbandOrientation::Hl => zc_hl(h, v, d),
        SubbandOrientation::Hh => zc_hh(h, v, d),
    }
}

/// Sign context from horizontal/vertical sign contributions (ISO 15444-1 Table D.3).
///
/// Returns `(ctx_index, xor_bit)` where `xor_bit` inverts the coded sign.
#[inline]
pub(crate) fn sc_context(kh: i32, kv: i32) -> (usize, u32) {
    match (kh, kv) {
        (1, 1) => (CTX_SC_BASE + 4, 0),
        (1, 0) => (CTX_SC_BASE + 3, 0),
        (1, -1) => (CTX_SC_BASE + 2, 0),
        (0, 1) => (CTX_SC_BASE + 1, 0),
        (0, 0) => (CTX_SC_BASE, 0),
        (0, -1) => (CTX_SC_BASE + 1, 1),
        (-1, 1) => (CTX_SC_BASE + 2, 1),
        (-1, 0) => (CTX_SC_BASE + 3, 1),
        (-1, -1) => (CTX_SC_BASE + 4, 1),
        _ => (CTX_SC_BASE, 0), // unreachable in practice
    }
}

/// Magnitude refinement context (ISO 15444-1 Â§D.3.3).
#[inline]
pub(crate) fn mr_context(has_sig_other: bool, refined_before: bool) -> usize {
    if refined_before {
        CTX_MR_BASE + 2
    } else if has_sig_other {
        CTX_MR_BASE + 1
    } else {
        CTX_MR_BASE
    }
}
