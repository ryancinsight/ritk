//! Domain types for RT Plan Storage.

/// SOP Class UID for RT Plan Storage.
pub const RT_PLAN_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.5";

/// Metadata for a single treatment beam from BeamSequence (300A,00B0).
#[derive(Debug, Clone)]
pub struct RtBeamInfo {
    /// BeamNumber (300A,00C0).
    pub beam_number: u32,
    /// BeamName (300A,00C2).
    pub beam_name: String,
    /// BeamDescription (300A,00C3).
    pub beam_description: String,
    /// RadiationType (300A,00C6): PHOTON, ELECTRON, NEUTRON, PROTON, etc.
    pub radiation_type: String,
    /// TreatmentDeliveryType (300A,00CE): TREATMENT, DRR, CONTINUATION, etc.
    pub treatment_delivery_type: String,
    /// NumberOfControlPoints (300A,0110).
    pub n_control_points: u32,
}

/// A fraction group from FractionGroupSequence (300A,0070).
#[derive(Debug, Clone)]
pub struct RtFractionGroup {
    /// FractionGroupNumber (300A,0071).
    pub fraction_group_number: u32,
    /// NumberOfFractionsPlanned (300A,0078).
    pub n_fractions_planned: u32,
    /// BeamNumbers referenced via ReferencedBeamSequence (300A,00B6) items.
    pub referenced_beam_numbers: Vec<u32>,
}

/// Parsed representation of a DICOM RT Plan file.
///
/// # Invariants
/// 1. SOP Class UID must equal `RT_PLAN_SOP_CLASS_UID`.
/// 2. `beams` order follows encounter order in BeamSequence (300A,00B0).
/// 3. `fraction_groups` order follows encounter order in FractionGroupSequence (300A,0070).
#[derive(Debug, Clone)]
pub struct RtPlanInfo {
    /// SOPInstanceUID (0008,0018).
    pub sop_instance_uid: String,
    /// RTPlanLabel (300A,0002).
    pub rt_plan_label: String,
    /// RTPlanName (300A,0003).
    pub rt_plan_name: String,
    /// RTPlanDescription (300A,0004).
    pub rt_plan_description: String,
    /// PlanIntent (300A,000A): CURATIVE, PALLIATIVE, PROPHYLACTIC, VERIFICATION, etc.
    pub plan_intent: String,
    /// Beams from BeamSequence (300A,00B0).
    pub beams: Vec<RtBeamInfo>,
    /// Fraction groups from FractionGroupSequence (300A,0070).
    pub fraction_groups: Vec<RtFractionGroup>,
}
