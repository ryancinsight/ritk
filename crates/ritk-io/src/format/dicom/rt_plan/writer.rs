//! RT Plan writer — serialize an [`RtPlanInfo`] to a DICOM Part-10 file.

use anyhow::{Context, Result};
use dicom::core::Tag;
use dicom::core::VR;
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;

use super::types::{RtBeamInfo, RtFractionGroup, RtPlanInfo, RT_PLAN_SOP_CLASS_UID};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::elements::{put_is, put_sequence, put_text};
use crate::format::dicom::writer::pixel_encoding::generate_series_uid;

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

    let generated_uid = generate_series_uid();
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

    put_text(&mut obj, Tag(0x0008, 0x0016), VR::UI, RT_PLAN_SOP_CLASS_UID);
    put_text(&mut obj, Tag(0x0008, 0x0018), VR::UI, sop_instance_uid);
    put_text(&mut obj, Tag(0x0008, 0x0060), VR::CS, "RTPLAN");
    put_text(
        &mut obj,
        Tag(0x300A, 0x0002),
        VR::LO,
        plan.rt_plan_label.as_str(),
    );
    put_text(
        &mut obj,
        Tag(0x300A, 0x0003),
        VR::LO,
        plan.rt_plan_name.as_str(),
    );
    put_text(
        &mut obj,
        Tag(0x300A, 0x0004),
        VR::ST,
        plan.rt_plan_description.as_str(),
    );
    put_text(
        &mut obj,
        Tag(0x300A, 0x000A),
        VR::CS,
        plan.plan_intent.as_str(),
    );

    if !beam_items.is_empty() {
        put_sequence(&mut obj, Tag(0x300A, 0x00B0), beam_items);
    }
    if !fg_items.is_empty() {
        put_sequence(&mut obj, Tag(0x300A, 0x0070), fg_items);
    }

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_PLAN_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid)
            .transfer_syntax(EXPLICIT_VR_LE),
    )
    .with_context(|| "build RT Plan file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Plan to {}", path.display()))?;

    Ok(())
}

fn build_beam_item(beam: &RtBeamInfo) -> InMemDicomObject {
    let mut item = InMemDicomObject::new_empty();
    put_is(&mut item, Tag(0x300A, 0x00C0), beam.beam_number);
    put_text(
        &mut item,
        Tag(0x300A, 0x00C2),
        VR::LO,
        beam.beam_name.as_str(),
    );
    put_text(
        &mut item,
        Tag(0x300A, 0x00C3),
        VR::ST,
        beam.beam_description.as_str(),
    );
    put_text(
        &mut item,
        Tag(0x300A, 0x00C6),
        VR::CS,
        beam.radiation_type.as_str(),
    );
    put_text(
        &mut item,
        Tag(0x300A, 0x00CE),
        VR::CS,
        beam.treatment_delivery_type.as_str(),
    );
    put_is(&mut item, Tag(0x300A, 0x0110), beam.n_control_points);
    item
}

fn build_fraction_group_item(fg: &RtFractionGroup) -> InMemDicomObject {
    let ref_beam_items: Vec<InMemDicomObject> = fg
        .referenced_beam_numbers
        .iter()
        .map(|&bn| {
            let mut ref_item = InMemDicomObject::new_empty();
            put_is(&mut ref_item, Tag(0x300A, 0x00C0), bn);
            ref_item
        })
        .collect();

    let mut item = InMemDicomObject::new_empty();
    put_is(&mut item, Tag(0x300A, 0x0071), fg.fraction_group_number);
    put_is(&mut item, Tag(0x300A, 0x0078), fg.n_fractions_planned);
    put_sequence(&mut item, Tag(0x300A, 0x00B6), ref_beam_items);
    item
}
