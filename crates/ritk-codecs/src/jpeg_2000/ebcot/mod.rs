//! EBCOT tier-1 encoder and decoder (ISO 15444-1 Annex D).

pub mod contexts;
pub mod decoder;
pub mod encoder;

pub use contexts::SubbandOrientation;
#[expect(unused_imports)]
pub use decoder::{decode_code_block, DecodedBlock};
#[expect(unused_imports)]
pub use encoder::{encode_code_block, EncodedBlock};

#[cfg(test)]
pub(crate) use contexts::{sc_context as sc_context_for_test, zc_context as zc_context_for_test};

#[cfg(test)]
mod tests;

// ── Per-sample state flags ────────────────────────────────────────────────────

/// Bit-packed per-sample state used during EBCOT processing.
///
/// Tier-1 repeatedly scans an entire 64×64 code-block, so storing four flags
/// in one byte keeps the state plane cache-resident without changing its
/// semantics.
#[derive(Clone, Copy, Default)]
pub(crate) struct SampleState(u8);

impl SampleState {
    const SIGNIFICANT: u8 = 1 << 0;
    const NEGATIVE: u8 = 1 << 1;
    const VISITED: u8 = 1 << 2;
    const REFINED: u8 = 1 << 3;

    #[inline(always)]
    pub(crate) fn is_significant(self) -> bool {
        self.0 & Self::SIGNIFICANT != 0
    }

    #[inline(always)]
    pub(crate) fn set_significant(&mut self) {
        self.0 |= Self::SIGNIFICANT;
    }

    #[inline(always)]
    pub(crate) fn is_negative(self) -> bool {
        self.0 & Self::NEGATIVE != 0
    }

    #[inline(always)]
    pub(crate) fn mark_negative(&mut self) {
        self.0 |= Self::NEGATIVE;
    }

    #[inline(always)]
    pub(crate) fn was_visited(self) -> bool {
        self.0 & Self::VISITED != 0
    }

    #[inline(always)]
    pub(crate) fn set_visited(&mut self) {
        self.0 |= Self::VISITED;
    }

    #[inline(always)]
    pub(crate) fn clear_visited(&mut self) {
        self.0 &= !Self::VISITED;
    }

    #[inline(always)]
    pub(crate) fn was_refined(self) -> bool {
        self.0 & Self::REFINED != 0
    }

    #[inline(always)]
    pub(crate) fn set_refined(&mut self) {
        self.0 |= Self::REFINED;
    }
}

const _: () = assert!(std::mem::size_of::<SampleState>() == 1);

// ── Neighbour utilities ───────────────────────────────────────────────────────

/// Count significant horizontal (H), vertical (V) and diagonal (D) neighbours.
#[inline]
pub(crate) fn neighbour_sig_counts(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> (u32, u32, u32) {
    let mut h = 0u32;
    let mut v = 0u32;
    let mut d = 0u32;
    if x > 0 && state[y * width + x - 1].is_significant() {
        h += 1;
    }
    if x + 1 < width && state[y * width + x + 1].is_significant() {
        h += 1;
    }
    if y > 0 && state[(y - 1) * width + x].is_significant() {
        v += 1;
    }
    if y + 1 < height && state[(y + 1) * width + x].is_significant() {
        v += 1;
    }
    if x > 0 && y > 0 && state[(y - 1) * width + x - 1].is_significant() {
        d += 1;
    }
    if x + 1 < width && y > 0 && state[(y - 1) * width + x + 1].is_significant() {
        d += 1;
    }
    if x > 0 && y + 1 < height && state[(y + 1) * width + x - 1].is_significant() {
        d += 1;
    }
    if x + 1 < width && y + 1 < height && state[(y + 1) * width + x + 1].is_significant() {
        d += 1;
    }
    (h, v, d)
}

/// Total significant neighbour count (H + V + D).
#[inline]
pub(crate) fn neighbour_sig_total(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> u32 {
    let (h, v, d) = neighbour_sig_counts(state, width, height, x, y);
    h + v + d
}

/// `true` if any of the 8 neighbours is significant (for MR context).
#[inline]
pub(crate) fn any_neighbour_sig(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> bool {
    neighbour_sig_total(state, width, height, x, y) > 0
}

/// Net sign contributions κ_h and κ_v (ISO 15444-1 Table D.3).
///
/// Each significant horizontal/vertical neighbour contributes +1 (positive)
/// or −1 (negative) based on its sign.  κ = signum(sum of contributions).
#[inline]
pub(crate) fn sign_contributions(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> (i32, i32) {
    let mut h_raw = 0i32;
    let mut v_raw = 0i32;

    if x > 0 {
        let s = &state[y * width + x - 1];
        if s.is_significant() {
            h_raw += if s.is_negative() { -1 } else { 1 };
        }
    }
    if x + 1 < width {
        let s = &state[y * width + x + 1];
        if s.is_significant() {
            h_raw += if s.is_negative() { -1 } else { 1 };
        }
    }
    if y > 0 {
        let s = &state[(y - 1) * width + x];
        if s.is_significant() {
            v_raw += if s.is_negative() { -1 } else { 1 };
        }
    }
    if y + 1 < height {
        let s = &state[(y + 1) * width + x];
        if s.is_significant() {
            v_raw += if s.is_negative() { -1 } else { 1 };
        }
    }

    (h_raw.signum(), v_raw.signum())
}

// ── Test-only symbol trace (CUP differential debugging) ──────────────────────

#[cfg(test)]
thread_local! {
    /// (context index, bit) pairs recorded by the cleanup pass.
    pub(crate) static CUP_TRACE: std::cell::RefCell<Vec<(usize, u32)>> =
        std::cell::RefCell::default();
}

#[cfg(test)]
pub(crate) fn cup_trace_take() -> Vec<(usize, u32)> {
    CUP_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

#[inline(always)]
#[expect(unused_variables)]
pub(crate) fn trace(ctx: usize, bit: u32) {
    #[cfg(test)]
    CUP_TRACE.with(|t| t.borrow_mut().push((ctx, bit)));
}
