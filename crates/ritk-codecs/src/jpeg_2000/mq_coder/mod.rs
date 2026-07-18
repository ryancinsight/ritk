//! MQ arithmetic coder â€“ encoder and decoder (ISO 15444-1 Annex C).

pub mod decoder;
pub mod encoder;

#[allow(unused_imports)]
pub use decoder::MqDecoder;
#[allow(unused_imports)]
pub use encoder::MqEncoder;

/// One entry of the MQ probability state table.
///
/// Fields: `(Qe, NLPS, NMPS, SWITCH)`.
#[derive(Clone, Copy)]
pub(crate) struct QeEntry {
    pub(crate) qe: u16,
    pub(crate) nlps: u8,
    pub(crate) nmps: u8,
    pub(crate) switch_flag: bool }

#[rustfmt::skip]
pub(crate) const QE_TABLE: [QeEntry; 47] = [
    // idx  Qe       NMPS  NLPS  SWITCH (ISO 15444-1 Table C.2)
    QeEntry { qe: 0x5601, nmps:  1, nlps:  1, switch_flag: true  }, //  0
    QeEntry { qe: 0x3401, nmps:  2, nlps:  6, switch_flag: false }, //  1
    QeEntry { qe: 0x1801, nmps:  3, nlps:  9, switch_flag: false }, //  2
    QeEntry { qe: 0x0AC1, nmps:  4, nlps: 12, switch_flag: false }, //  3
    QeEntry { qe: 0x0521, nmps:  5, nlps: 29, switch_flag: false }, //  4
    QeEntry { qe: 0x0221, nmps: 38, nlps: 33, switch_flag: false }, //  5
    QeEntry { qe: 0x5601, nmps:  7, nlps:  6, switch_flag: true  }, //  6
    QeEntry { qe: 0x5401, nmps:  8, nlps: 14, switch_flag: false }, //  7
    QeEntry { qe: 0x4801, nmps:  9, nlps: 14, switch_flag: false }, //  8
    QeEntry { qe: 0x3801, nmps: 10, nlps: 14, switch_flag: false }, //  9
    QeEntry { qe: 0x3001, nmps: 11, nlps: 17, switch_flag: false }, // 10
    QeEntry { qe: 0x2401, nmps: 12, nlps: 18, switch_flag: false }, // 11
    QeEntry { qe: 0x1C01, nmps: 13, nlps: 20, switch_flag: false }, // 12
    QeEntry { qe: 0x1601, nmps: 29, nlps: 21, switch_flag: false }, // 13
    QeEntry { qe: 0x5601, nmps: 15, nlps: 14, switch_flag: true  }, // 14
    QeEntry { qe: 0x5401, nmps: 16, nlps: 14, switch_flag: false }, // 15
    QeEntry { qe: 0x5101, nmps: 17, nlps: 15, switch_flag: false }, // 16
    QeEntry { qe: 0x4801, nmps: 18, nlps: 16, switch_flag: false }, // 17
    QeEntry { qe: 0x3801, nmps: 19, nlps: 17, switch_flag: false }, // 18
    QeEntry { qe: 0x3401, nmps: 20, nlps: 18, switch_flag: false }, // 19
    QeEntry { qe: 0x3001, nmps: 21, nlps: 19, switch_flag: false }, // 20
    QeEntry { qe: 0x2801, nmps: 22, nlps: 19, switch_flag: false }, // 21
    QeEntry { qe: 0x2401, nmps: 23, nlps: 20, switch_flag: false }, // 22
    QeEntry { qe: 0x2201, nmps: 24, nlps: 21, switch_flag: false }, // 23
    QeEntry { qe: 0x1C01, nmps: 25, nlps: 22, switch_flag: false }, // 24
    QeEntry { qe: 0x1801, nmps: 26, nlps: 23, switch_flag: false }, // 25
    QeEntry { qe: 0x1601, nmps: 27, nlps: 24, switch_flag: false }, // 26
    QeEntry { qe: 0x1401, nmps: 28, nlps: 25, switch_flag: false }, // 27
    QeEntry { qe: 0x1201, nmps: 29, nlps: 26, switch_flag: false }, // 28
    QeEntry { qe: 0x1101, nmps: 30, nlps: 27, switch_flag: false }, // 29
    QeEntry { qe: 0x0AC1, nmps: 31, nlps: 28, switch_flag: false }, // 30
    QeEntry { qe: 0x09C1, nmps: 32, nlps: 29, switch_flag: false }, // 31
    QeEntry { qe: 0x08A1, nmps: 33, nlps: 30, switch_flag: false }, // 32
    QeEntry { qe: 0x0521, nmps: 34, nlps: 31, switch_flag: false }, // 33
    QeEntry { qe: 0x0441, nmps: 35, nlps: 32, switch_flag: false }, // 34
    QeEntry { qe: 0x02A1, nmps: 36, nlps: 33, switch_flag: false }, // 35
    QeEntry { qe: 0x0221, nmps: 37, nlps: 34, switch_flag: false }, // 36
    QeEntry { qe: 0x0141, nmps: 38, nlps: 35, switch_flag: false }, // 37
    QeEntry { qe: 0x0111, nmps: 39, nlps: 36, switch_flag: false }, // 38
    QeEntry { qe: 0x0085, nmps: 40, nlps: 37, switch_flag: false }, // 39
    QeEntry { qe: 0x0049, nmps: 41, nlps: 38, switch_flag: false }, // 40
    QeEntry { qe: 0x0025, nmps: 42, nlps: 39, switch_flag: false }, // 41
    QeEntry { qe: 0x0015, nmps: 43, nlps: 40, switch_flag: false }, // 42
    QeEntry { qe: 0x0009, nmps: 44, nlps: 41, switch_flag: false }, // 43
    QeEntry { qe: 0x0005, nmps: 45, nlps: 42, switch_flag: false }, // 44
    QeEntry { qe: 0x0001, nmps: 45, nlps: 43, switch_flag: false }, // 45
    QeEntry { qe: 0x5601, nmps: 46, nlps: 46, switch_flag: false }, // 46
];

/// Number of MQ contexts used by EBCOT.
pub const NUM_CONTEXTS: usize = 19;

/// Per-context probability state (state-table index + current MPS value).
#[derive(Clone, Copy, Debug, Default)]
pub struct CtxState {
    /// Index into `QE_TABLE` (0â€“46).
    pub state: u8,
    /// Current Most-Probable Symbol (0 or 1).
    pub mps: u8 }

/// Create the 19-context initial state array (ISO 15444-1 Table D.7).
///
/// All contexts start at state 0 except: ZC context 0 â†’ state 4,
/// run-length (AGG) context â†’ state 3, uniform (UNI) context â†’ state 46.
pub fn initial_contexts() -> [CtxState; NUM_CONTEXTS] {
    let mut ctx = [CtxState::default(); NUM_CONTEXTS];
    // ZC context 0 (all-non-significant neighbourhood).
    ctx[0] = CtxState { state: 4, mps: 0 };
    // UNI context (17): near-uniform probability.
    ctx[17] = CtxState { state: 46, mps: 0 };
    // AGG / RLC context (18).
    ctx[18] = CtxState { state: 3, mps: 0 };
    ctx
}

#[cfg(test)]
mod tests;
