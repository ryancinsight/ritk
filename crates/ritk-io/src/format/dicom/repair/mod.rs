//! Repair of encoding defects that stop conformant readers from using a
//! data set.
//!
//! [`repair_values`] rewrites date, time and UID values that are spelled
//! outside their value representation's grammar but still denote one value:
//! the ACR-NEMA date `YYYY.MM.DD`, the ACR-NEMA time `HH:MM:SS.frac`, and UIDs
//! with surrounding spaces or leading zeros in a component (PS3.5 Table 6.2-1
//! and section 9.1). A value with no known rewrite is left byte for byte as it
//! is; nothing is guessed, removed or invented. Values the element holds as
//! parsed dates or times, rather than text, are conformant by construction
//! and are not read.

mod data_set;
mod study;
mod value;

pub use data_set::{repair_values, RepairError, RepairSummary};
pub use study::{repair_dicom_study, StudyRepairError, StudyRepairReport};
