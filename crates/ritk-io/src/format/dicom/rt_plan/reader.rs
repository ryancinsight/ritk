//! RT Plan reader â€” parse a DICOM RT Plan Storage file into [`RtPlanInfo`].

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use dicom::core::value::Value;
use dicom::core::Tag;
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::path::Path;

use super::types::{RtBeamInfo, RtFractionGroup, RtPlanInfo, RT_PLAN_SOP_CLASS_UID};
use crate::format::dicom::reader::types::truncate_arraystring;

fn parse_u32_is(raw: &str, field: &str) -> Result<u32> {
    let token = raw.trim();
    if token.is_empty() {
        bail!("{field} is present but empty");
    }
    token
        .parse::<u32>()
        .with_context(|| format!("Invalid {field}: '{token}'"))
}

/// Read an RT Plan Storage DICOM file at `path` into an [`RtPlanInfo`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's `MediaStorageSOPClassUID` â‰  `1.2.840.10008.5.1.4.1.1.481.5`.
/// - Present RT Plan sequence integer fields are malformed.
/// - Present beam or fraction-group sequence elements are not DICOM sequences.
pub fn read_rt_plan<P: AsRef<Path>>(path: P) -> Result<RtPlanInfo> {
    let path = path.as_ref();
    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("open DICOM file: {}", path.display()))?;

    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != RT_PLAN_SOP_CLASS_UID {
        bail!(
            "SOP Class UID '{}' is not RT Plan Storage ({})",
            sop,
            RT_PLAN_SOP_CLASS_UID
        );
    }

    let sop_instance_uid = obj
        .element(Tag(0x0008, 0x0018))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .map(|s| match ArrayString::<64>::from(s.as_str()) {
            Ok(v) => v,
            Err(_) => {
                tracing::warn!("SOPInstanceUID exceeds 64 chars, truncating: {}", &s[..64]);
                truncate_arraystring::<64>(s.as_str())
            }
        })
        .unwrap_or_else(ArrayString::new);

    let rt_plan_label = obj
        .element(Tag(0x300A, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let rt_plan_name = obj
        .element(Tag(0x300A, 0x0003))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let rt_plan_description = obj
        .element(Tag(0x300A, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_default();

    let plan_intent = obj
        .element(Tag(0x300A, 0x000A))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .map(|s| match ArrayString::<16>::from(s.as_str()) {
            Ok(v) => v,
            Err(_) => {
                tracing::warn!("PlanIntent exceeds 16 chars, truncating: {}", &s[..16]);
                truncate_arraystring::<16>(s.as_str())
            }
        })
        .unwrap_or_else(ArrayString::new);

    let beams: Vec<RtBeamInfo> = match obj.element(Tag(0x300A, 0x00B0)) {
        Ok(elem) => match elem.value() {
            Value::Sequence(seq) => seq
                .items()
                .iter()
                .map(|item| -> Result<RtBeamInfo> {
                    let beam_number: u32 = item
                        .element(Tag(0x300A, 0x00C0))
                        .ok()
                        .map(|e| {
                            let raw = e.to_str().context("Read BeamNumber (300A,00C0)")?;
                            parse_u32_is(&raw, "BeamNumber (300A,00C0)")
                        })
                        .transpose()?
                        .unwrap_or(0);
                    let beam_name = item
                        .element(Tag(0x300A, 0x00C2))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .unwrap_or_default();
                    let beam_description = item
                        .element(Tag(0x300A, 0x00C3))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .unwrap_or_default();
                    let radiation_type = item
                        .element(Tag(0x300A, 0x00C6))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .map(|s| match ArrayString::<16>::from(s.as_str()) {
                            Ok(v) => v,
                            Err(_) => {
                                tracing::warn!(
                                    "RadiationType exceeds 16 chars, truncating: {}",
                                    &s[..16]
                                );
                                truncate_arraystring::<16>(s.as_str())
                            }
                        })
                        .unwrap_or_else(ArrayString::new);
                    let treatment_delivery_type = item
                        .element(Tag(0x300A, 0x00CE))
                        .ok()
                        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
                        .map(|s| match ArrayString::<16>::from(s.as_str()) {
                            Ok(v) => v,
                            Err(_) => {
                                tracing::warn!(
                                    "TreatmentDeliveryType exceeds 16 chars, truncating: {}",
                                    &s[..16]
                                );
                                truncate_arraystring::<16>(s.as_str())
                            }
                        })
                        .unwrap_or_else(ArrayString::new);
                    let n_control_points: u32 = item
                        .element(Tag(0x300A, 0x0110))
                        .ok()
                        .map(|e| {
                            let raw = e
                                .to_str()
                                .context("Read NumberOfControlPoints (300A,0110)")?;
                            parse_u32_is(&raw, "NumberOfControlPoints (300A,0110)")
                        })
                        .transpose()?
                        .unwrap_or(0);
                    Ok(RtBeamInfo {
                        beam_number,
                        beam_name,
                        beam_description,
                        radiation_type,
                        treatment_delivery_type,
                        n_control_points,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("BeamSequence (300A,00B0) is present but is not a sequence"),
        },
        Err(_) => Vec::new(),
    };

    let fraction_groups: Vec<RtFractionGroup> = match obj.element(Tag(0x300A, 0x0070)) {
        Ok(elem) => match elem.value() {
            Value::Sequence(seq) => seq
                .items()
                .iter()
                .map(|item| -> Result<RtFractionGroup> {
                    let fraction_group_number: u32 = item
                        .element(Tag(0x300A, 0x0071))
                        .ok()
                        .map(|e| {
                            let raw = e.to_str().context("Read FractionGroupNumber (300A,0071)")?;
                            parse_u32_is(&raw, "FractionGroupNumber (300A,0071)")
                        })
                        .transpose()?
                        .unwrap_or(0);
                    let n_fractions_planned: u32 = item
                        .element(Tag(0x300A, 0x0078))
                        .ok()
                        .map(|e| {
                            let raw = e
                                .to_str()
                                .context("Read NumberOfFractionsPlanned (300A,0078)")?;
                            parse_u32_is(&raw, "NumberOfFractionsPlanned (300A,0078)")
                        })
                        .transpose()?
                        .unwrap_or(0);

                    let referenced_beam_numbers: Vec<u32> =
                        match item.element(Tag(0x300A, 0x00B6)) {
                            Ok(e) => match e.value() {
                                Value::Sequence(s) => s
                                    .items()
                                    .iter()
                                    .map(|bi| -> Result<u32> {
                                        bi.element(Tag(0x300A, 0x00C0))
                                            .ok()
                                            .map(|be| {
                                                let raw = be.to_str().context(
                                                    "Read ReferencedBeamNumber (300A,00C0)",
                                                )?;
                                                parse_u32_is(
                                                    &raw,
                                                    "ReferencedBeamNumber (300A,00C0)",
                                                )
                                            })
                                            .transpose()
                                            .map(|value| value.unwrap_or(0))
                                    })
                                    .collect::<Result<Vec<_>>>()?,
                                _ => bail!(
                                    "ReferencedBeamSequence (300A,00B6) is present but is not a sequence"
                                ) },
                            Err(_) => Vec::new() };

                    Ok(RtFractionGroup {
                        fraction_group_number,
                        n_fractions_planned,
                        referenced_beam_numbers })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("FractionGroupSequence (300A,0070) is present but is not a sequence") },
        Err(_) => Vec::new() };

    tracing::debug!(
        "read_rt_plan: label='{}' n_beams={} n_fraction_groups={}",
        rt_plan_label,
        beams.len(),
        fraction_groups.len(),
    );

    Ok(RtPlanInfo {
        sop_instance_uid,
        rt_plan_label,
        rt_plan_name,
        rt_plan_description,
        plan_intent,
        beams,
        fraction_groups,
    })
}
