//! Repair of encoding defects that stop conformant readers from using a
//! data set.
//!
//! The value rewrites cover date, time and UID values that are spelled
//! outside their value representation's grammar but still denote one value:
//! the ACR-NEMA date `YYYY.MM.DD`, the ACR-NEMA time `HH:MM:SS.frac`, and UIDs
//! with surrounding spaces or leading zeros in a component (PS3.5 Table 6.2-1
//! and section 9.1). A value with no known rewrite is left as it is; nothing
//! is guessed, removed or invented.

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "ratchet RITK-DICOM-REPAIR-OBJECT-001: the data-set repair is the first caller"
    )
)]
mod value;
