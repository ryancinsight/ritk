//! RT Plan writer — serialize an [`RtPlanInfo`] to a DICOM Part-10 file.

use anyhow::{Context, Result};
use dicom::core::header::Length;
use dicom::core::value::DataSetSequence;
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use super::types::{RtBeamInfo, RtFractionGroup, RtPlanInfo, RT_PLAN_SOP_CLASS_UID};

/// Write an [`RtPlanInfo`] to a DICOM RT Plan Storage file at `path`.
///
/// # Write/Read Invariant
///
/// All plan-level strings and all beam and fraction group fields are preserved
/// through the DICOM write-read cycle without loss.
///
/// # Errors
/// - File cannot be created or written at `path`.
pub fn write_rt_plan<P: AsRef<Path>>(path: P, plan: &RtPlanInfo) -> Result<()> {
    let path = path.as_ref();

    static RT_PLAN_UID_COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = RT_PLAN_UID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let generated_uid = format!("2.25.{}.{}", t, n);
    let sop_instance_uid = if plan.sop_instance_uid.trim().is_empty() {
        generated_uid.as_str()
    } else {
        plan.sop_instance_uid.trim()
    };

    let beam_items: Vec<InMemDicomObject> = plan.beams.iter().map(build_beam_item).collect();

    let fg_items: Vec<InMemDicomObject> = plan
        .fraction_groups
        .iter()
        .map(build_fraction_group_item)
        .collect();

    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(RT_PLAN_SOP_CLASS_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("RTPLAN"),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x0002),
        VR::LO,
        PrimitiveValue::from(plan.rt_plan_label.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x0003),
        VR::LO,
        PrimitiveValue::from(plan.rt_plan_name.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x0004),
        VR::ST,
        PrimitiveValue::from(plan.rt_plan_description.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x300A, 0x000A),
        VR::CS,
        PrimitiveValue::from(plan.plan_intent.as_str()),
    ));

    if !beam_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x300A, 0x00B0),
            VR::SQ,
            Value::from(DataSetSequence::new(beam_items, Length::UNDEFINED)),
        ));
    }
    if !fg_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x300A, 0x0070),
            VR::SQ,
            Value::from(DataSetSequence::new(fg_items, Length::UNDEFINED)),
        ));
    }

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_PLAN_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .with_context(|| "build RT Plan file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Plan to {}", path.display()))?;

    Ok(())
}

fn build_beam_item(beam: &RtBeamInfo) -> InMemDicomObject {
    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x300A, 0x00C0),
        VR::IS,
        PrimitiveValue::from(beam.beam_number.to_string().as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x00C2),
        VR::LO,
        PrimitiveValue::from(beam.beam_name.as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x00C3),
        VR::ST,
        PrimitiveValue::from(beam.beam_description.as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x00C6),
        VR::CS,
        PrimitiveValue::from(beam.radiation_type.as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x00CE),
        VR::CS,
        PrimitiveValue::from(beam.treatment_delivery_type.as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x0110),
        VR::IS,
        PrimitiveValue::from(beam.n_control_points.to_string().as_str()),
    ));
    item
}

fn build_fraction_group_item(fg: &RtFractionGroup) -> InMemDicomObject {
    let ref_beam_items: Vec<InMemDicomObject> = fg
        .referenced_beam_numbers
        .iter()
        .map(|&bn| {
            let mut ref_item = InMemDicomObject::new_empty();
            ref_item.put(DataElement::new(
                Tag(0x300A, 0x00C0),
                VR::IS,
                PrimitiveValue::from(bn.to_string().as_str()),
            ));
            ref_item
        })
        .collect();

    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x300A, 0x0071),
        VR::IS,
        PrimitiveValue::from(fg.fraction_group_number.to_string().as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x0078),
        VR::IS,
        PrimitiveValue::from(fg.n_fractions_planned.to_string().as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x300A, 0x00B6),
        VR::SQ,
        Value::from(DataSetSequence::new(ref_beam_items, Length::UNDEFINED)),
    ));
    item
}
