//! RT Plan Storage (SOP Class 1.2.840.10008.5.1.4.1.1.481.5) reader/writer.
//!
//! # Specification
//!
//! An RT Plan file contains radiotherapy treatment planning metadata.
//! Key structures:
//! - (300A,0010) DoseReferenceSequence: prescribed dose constraints.
//! - (300A,0070) FractionGroupSequence: fractions and beam references.
//! - (300A,00B0) BeamSequence: treatment beams with geometry and MUs.

mod reader;
mod types;
mod writer;

pub use reader::read_rt_plan;
pub use types::{RtBeamInfo, RtFractionGroup, RtPlanInfo, RT_PLAN_SOP_CLASS_UID};
pub use writer::write_rt_plan;

#[cfg(test)]
mod tests;
