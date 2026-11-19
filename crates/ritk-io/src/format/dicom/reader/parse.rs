//! Metadata extraction after image SOP and series identity selection.

use super::types::Radioactivity;
use aequitas::systems::si::{
    quantities::{Activity, Angle, Length, Mass, Time},
    units::{Becquerel, Degree, Megabecquerel, Millimeter},
};
use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use dicom::core::Tag;
use dicom::dictionary_std::uids;
use dicom::object::DefaultDicomObject;
use ritk_dicom::PixelSignedness;

use super::preservation::preserve_unhandled_elements;
use super::types::{
    cs_to_arraystring, da_to_arraystring, parse_patient_position, tm_to_arraystring,
    uid_to_arraystring, DicomSliceMetadata, SeriesFirstSeen,
};
use crate::format::dicom::color_common::read_required_unsigned;
use crate::format::dicom::object_model::DicomPreservationSet;

/// Reject a PixelSpacing pair that cannot describe a physical grid.
///
/// `Spacing::new` asserts each component is finite and strictly positive, so a
/// value that fails here would abort the process rather than fail the read.
/// Derived and secondary-capture objects legitimately carry `PixelSpacing` of
/// `"0\0"`, and a malformed one can carry a negative or non-numeric pair, all
/// of which parse as `f64` and reach the constructor. Treating them as absent
/// keeps the documented fallback path and leaves the series readable.
fn usable_pixel_spacing(spacing: [f64; 2]) -> Option<[f64; 2]> {
    if spacing.iter().all(|&v| positive_length_mm(v).is_some()) {
        Some(spacing)
    } else {
        tracing::warn!(
            row_spacing = spacing[0],
            column_spacing = spacing[1],
            "DICOM PixelSpacing (0028,0030) has no finite positive SI representation"
        );
        None
    }
}

/// Parse a backslash-delimited DICOM string of floating-point values into a fixed-size array.
/// Returns `Some([v0..vN])` if at least `N` values parsed successfully, else `None`.
fn parse_ds_array<const N: usize>(s: &str) -> Option<[f64; N]> {
    let mut out = [0.0_f64; N];
    let mut count = 0usize;
    for part in s.split('\\') {
        if count >= N {
            break;
        }
        if let Ok(v) = part.trim().parse::<f64>() {
            out[count] = v;
            count += 1;
        }
    }
    if count >= N {
        Some(out)
    } else {
        None
    }
}

/// Shared tag-extraction logic for both file-based and in-memory DICOM parsing.
///
/// Populates `first` with series-level first-seen fields and returns the
/// per-slice metadata and per-file
/// image dimensions.
pub(super) fn extract_dicom_metadata(
    obj: &DefaultDicomObject,
    path_for_meta: PathBuf,
    first: &mut SeriesFirstSeen,
) -> Result<(DicomSliceMetadata, (u32, u32))> {
    let mut slice_meta = DicomSliceMetadata {
        path: path_for_meta,
        preservation: DicomPreservationSet::new(),
        sop_instance_uid: None,
        instance_number: None,
        slice_location: None,
        image_position_patient: None,
        image_orientation_patient: None,
        pixel_spacing: None,
        slice_thickness: None,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
        sop_class_uid: None,
        transfer_syntax_uid: None,
        private_tags: HashMap::new(),
        pixel_representation: PixelSignedness::Unsigned,
        bits_allocated: 16,
        bits_stored: 16,
        window_center: None,
        window_width: None,
        gantry_tilt: None,
        patient_position: None,
        part10_bytes: None,
    };

    // --- Per-slice fields ---
    if slice_meta.sop_instance_uid.is_none()
        && let Ok(elem) = obj.element(Tag(0x0008, 0x0018))
    {
        slice_meta.sop_instance_uid = elem
            .to_str()
            .ok()
            .as_ref()
            .and_then(|s| uid_to_arraystring(s));
    }
    if let Ok(elem) = obj.element(Tag(0x0020, 0x0013)) {
        slice_meta.instance_number = elem.to_str().ok().and_then(|s| s.parse().ok());
    }
    if let Ok(elem) = obj.element(Tag(0x0020, 0x1041)) {
        slice_meta.slice_location = elem
            .to_str()
            .ok()
            .and_then(|s| finite_decimal(&s))
            .map(Length::from_unit::<Millimeter>);
    }
    if let Ok(elem) = obj.element(Tag(0x0020, 0x0032))
        && let Ok(s) = elem.to_str()
    {
        slice_meta.image_position_patient = parse_ds_array::<3>(&s)
            .filter(|position| position.iter().all(|value| value.is_finite()))
            .map(|position| position.map(Length::from_unit::<Millimeter>));
    }
    if let Ok(elem) = obj.element(Tag(0x0020, 0x0037))
        && let Ok(s) = elem.to_str()
    {
        slice_meta.image_orientation_patient = parse_ds_array::<6>(&s);
    }
    if let Ok(elem) = obj.element(Tag(0x0028, 0x0030))
        && let Ok(s) = elem.to_str()
    {
        slice_meta.pixel_spacing = parse_ds_array::<2>(&s)
            .and_then(usable_pixel_spacing)
            .map(|spacing| spacing.map(Length::from_unit::<Millimeter>));
    }
    if let Ok(elem) = obj.element(Tag(0x0018, 0x0050)) {
        slice_meta.slice_thickness = elem
            .to_str()
            .ok()
            .and_then(|s| finite_decimal(&s))
            .and_then(positive_length_mm);
    }
    if let Ok(elem) = obj.element(Tag(0x0028, 0x1053)) {
        slice_meta.rescale_slope = elem
            .to_str()
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
    }
    if let Ok(elem) = obj.element(Tag(0x0028, 0x1052)) {
        slice_meta.rescale_intercept = elem
            .to_str()
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
    }
    if let Ok(elem) = obj.element(Tag(0x0008, 0x0016)) {
        slice_meta.sop_class_uid = elem
            .to_str()
            .ok()
            .as_ref()
            .and_then(|s| uid_to_arraystring(s));
    }
    // Transfer syntax from file meta (0002,0010), not main dataset.
    slice_meta.transfer_syntax_uid = uid_to_arraystring(obj.meta().transfer_syntax());

    if let Ok(elem) = obj.element(Tag(0x0028, 0x0103)) {
        slice_meta.pixel_representation = elem
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse().ok()) // parse u16
            .and_then(|v: u16| PixelSignedness::try_from(v).ok())
            .unwrap_or(PixelSignedness::Unsigned);
    }
    if let Ok(elem) = obj.element(Tag(0x0028, 0x0100)) {
        slice_meta.bits_allocated = elem
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(slice_meta.bits_allocated);
    }
    slice_meta.bits_stored = read_required_unsigned(obj, Tag(0x0028, 0x0101), "BitsStored")?;
    // WindowCenter (0028,1050) — first value of potentially multi-valued DS.
    if let Ok(elem) = obj.element(Tag(0x0028, 0x1050)) {
        slice_meta.window_center = elem.to_str().ok().and_then(|s| {
            s.trim()
                .split('\\')
                .next()
                .and_then(|v| v.trim().parse().ok())
        });
    }
    // WindowWidth (0028,1051) — first value of potentially multi-valued DS.
    if let Ok(elem) = obj.element(Tag(0x0028, 0x1051)) {
        slice_meta.window_width = elem.to_str().ok().and_then(|s| {
            s.trim()
                .split('\\')
                .next()
                .and_then(|v| v.trim().parse().ok())
        });
    }
    if let Ok(elem) = obj.element(Tag(0x0018, 0x1120)) {
        slice_meta.gantry_tilt = elem
            .to_str()
            .ok()
            .and_then(|s| finite_decimal(&s))
            .map(Angle::from_unit::<Degree>);
    }
    if let Ok(elem) = obj.element(Tag(0x0018, 0x5100)) {
        slice_meta.patient_position = elem
            .to_str()
            .ok()
            .as_deref()
            .and_then(parse_patient_position);
    }

    // Per-file dimension tracking (for canonical-dimension plurality selection).
    let this_rows: Option<u32> = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.parse().ok());
    let this_cols: Option<u32> = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.parse().ok());
    let file_dim = (this_rows.unwrap_or(0), this_cols.unwrap_or(0));

    // --- Series-level first-seen accumulation ---
    if first.rows.is_none() {
        first.rows = this_rows;
    }
    if first.cols.is_none() {
        first.cols = this_cols;
    }
    first.pixel_spacing = first.pixel_spacing.or(slice_meta.pixel_spacing);
    first.slice_thickness = first.slice_thickness.or(slice_meta.slice_thickness);
    if first.series_instance_uid.is_none()
        && let Ok(elem) = obj.element(Tag(0x0020, 0x000E))
    {
        first.series_instance_uid = elem
            .to_str()
            .ok()
            .as_ref()
            .and_then(|s| uid_to_arraystring(s));
    }
    if first.study_instance_uid.is_none()
        && let Ok(elem) = obj.element(Tag(0x0020, 0x000D))
    {
        first.study_instance_uid = elem
            .to_str()
            .ok()
            .as_ref()
            .and_then(|s| uid_to_arraystring(s));
    }
    if first.series_description.is_none()
        && let Ok(elem) = obj.element(Tag(0x0008, 0x103E))
    {
        first.series_description = elem.to_str().ok().map(String::from);
    }
    if first.modality.is_none()
        && let Ok(elem) = obj.element(Tag(0x0008, 0x0060))
    {
        first.modality = elem.to_str().ok().map(|s| cs_to_arraystring(s.trim()));
    }
    if first.patient_id.is_none()
        && let Ok(elem) = obj.element(Tag(0x0010, 0x0020))
    {
        first.patient_id = elem.to_str().ok().map(String::from);
    }
    if first.patient_name.is_none()
        && let Ok(elem) = obj.element(Tag(0x0010, 0x0010))
    {
        first.patient_name = elem.to_str().ok().map(String::from);
    }
    if first.study_date.is_none()
        && let Ok(elem) = obj.element(Tag(0x0008, 0x0020))
    {
        first.study_date = elem.to_str().ok().map(|s| da_to_arraystring(s.trim()));
    }
    if first.series_date.is_none()
        && let Ok(elem) = obj.element(Tag(0x0008, 0x0021))
    {
        first.series_date = elem.to_str().ok().map(|s| da_to_arraystring(s.trim()));
    }
    if first.series_time.is_none()
        && let Ok(elem) = obj.element(Tag(0x0008, 0x0031))
    {
        first.series_time = elem.to_str().ok().map(|s| tm_to_arraystring(s.trim()));
    }
    if first.frame_of_reference_uid.is_none()
        && let Ok(elem) = obj.element(Tag(0x0020, 0x0052))
    {
        first.frame_of_reference_uid = elem
            .to_str()
            .ok()
            .as_ref()
            .and_then(|s| uid_to_arraystring(s));
    }
    if first.bits_allocated.is_none()
        && let Ok(elem) = obj.element(Tag(0x0028, 0x0100))
    {
        first.bits_allocated = elem.to_str().ok().and_then(|s| s.parse().ok());
    }
    if first.bits_stored.is_none()
        && let Ok(elem) = obj.element(Tag(0x0028, 0x0101))
    {
        first.bits_stored = elem.to_str().ok().and_then(|s| s.parse().ok());
    }
    if first.high_bit.is_none()
        && let Ok(elem) = obj.element(Tag(0x0028, 0x0102))
    {
        first.high_bit = elem.to_str().ok().and_then(|s| s.parse().ok());
    }
    if first.photometric_interpretation.is_none()
        && let Ok(elem) = obj.element(Tag(0x0028, 0x0004))
    {
        first.photometric_interpretation = elem.to_str().ok().map(|s| cs_to_arraystring(s.trim()));
    }
    if first.transfer_syntax_uid.is_none() {
        first.transfer_syntax_uid = uid_to_arraystring(obj.meta().transfer_syntax());
    }
    if first.patient_weight_kg.is_none()
        && let Ok(elem) = obj.element(Tag(0x0010, 0x1030))
    {
        first.patient_weight_kg = elem
            .to_str()
            .ok()
            .and_then(|s| positive_decimal(&s))
            .map(Mass::from_base);
    }
    if first.decay_correction.is_none()
        && let Ok(elem) = obj.element(Tag(0x0054, 0x1102))
    {
        first.decay_correction = elem.to_str().ok().map(|s| cs_to_arraystring(s.trim()));
    }
    let has_pet_concentration = slice_meta.sop_class_uid.as_deref()
        == Some(uids::POSITRON_EMISSION_TOMOGRAPHY_IMAGE_STORAGE)
        && obj
            .element(Tag(0x0054, 0x1001))
            .is_ok_and(|units| units.to_str().is_ok_and(|value| value.trim() == "BQML"));
    first.pet_pixels_in_bq_per_ml =
        Some(first.pet_pixels_in_bq_per_ml.unwrap_or(true) && has_pet_concentration);
    // A series-level activity is unambiguous only for a single isotope item.
    if (first.radionuclide_total_dose_bq.is_none()
        || first.radionuclide_half_life_s.is_none()
        || first.radiopharmaceutical_start_time.is_none())
        && let Ok(seq_elem) = obj.element(Tag(0x0054, 0x0016))
        && let Some(items) = seq_elem.value().items()
        && let [first_item] = items
    {
        if first.radionuclide_total_dose_bq.is_none()
            && let Ok(e) = first_item.element(Tag(0x0018, 0x1074))
        {
            first.radionuclide_total_dose_bq = slice_meta
                .sop_class_uid
                .as_deref()
                .zip(e.to_str().ok())
                .and_then(|(sop_class, dose)| radionuclide_activity(sop_class, &dose));
        }
        if first.radionuclide_half_life_s.is_none()
            && let Ok(e) = first_item.element(Tag(0x0018, 0x1075))
        {
            first.radionuclide_half_life_s = e
                .to_str()
                .ok()
                .and_then(|s| positive_decimal(&s))
                .map(Time::from_base);
        }
        if first.radiopharmaceutical_start_time.is_none()
            && let Ok(e) = first_item.element(Tag(0x0018, 0x1072))
        {
            first.radiopharmaceutical_start_time =
                e.to_str().ok().map(|s| tm_to_arraystring(s.trim()));
        }
    }

    preserve_unhandled_elements(obj, &mut slice_meta.preservation);
    Ok((slice_meta, file_dim))
}

fn positive_length_mm(value: f64) -> Option<Length> {
    let length = Length::from_unit::<Millimeter>(value);
    (length.as_base().is_finite() && *length.as_base() > 0.0).then_some(length)
}

fn finite_decimal(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

fn positive_decimal(value: &str) -> Option<f64> {
    finite_decimal(value).filter(|v| *v > 0.0)
}

fn radionuclide_activity(sop_class: &str, value: &str) -> Option<Radioactivity> {
    // PS3.3 C.8.9.2 uses Bq, and C.8.4.10 and C.8.22.4 use MBq.
    let value = positive_decimal(value)?;
    let activity = match sop_class {
        uids::POSITRON_EMISSION_TOMOGRAPHY_IMAGE_STORAGE => Activity::from_unit::<Becquerel>(value),
        uids::ENHANCED_PET_IMAGE_STORAGE | uids::NUCLEAR_MEDICINE_IMAGE_STORAGE => {
            Activity::from_unit::<Megabecquerel>(value)
        }
        _ => return None,
    };
    Radioactivity::from_activity(activity)
}

#[cfg(test)]
mod tests_pixel_spacing {
    use super::usable_pixel_spacing;

    #[test]
    fn spacing_rejects_values_that_round_to_zero_in_metres() {
        let least_positive_millimetre = f64::from_bits(1);
        assert_eq!(usable_pixel_spacing([least_positive_millimetre, 1.0]), None);
        assert_eq!(usable_pixel_spacing([1.0, least_positive_millimetre]), None);
    }

    /// A conformant pair passes through byte-for-byte; the guard must not
    /// perturb legitimate geometry.
    #[test]
    fn conformant_spacing_is_preserved_exactly() {
        assert_eq!(usable_pixel_spacing([0.7, 0.65]), Some([0.7, 0.65]));
    }

    /// `PixelSpacing = "0\0"` occurs in derived and secondary-capture objects.
    /// It parses as a valid pair of f64 and previously reached `Spacing::new`,
    /// whose assert aborts the process.
    #[test]
    fn zero_spacing_is_rejected_rather_than_reaching_the_constructor() {
        assert_eq!(usable_pixel_spacing([0.0, 0.0]), None);
        assert_eq!(usable_pixel_spacing([0.0, 1.0]), None);
        assert_eq!(usable_pixel_spacing([1.0, 0.0]), None);
    }

    #[test]
    fn negative_spacing_is_rejected() {
        assert_eq!(usable_pixel_spacing([-1.0, 1.0]), None);
        assert_eq!(usable_pixel_spacing([1.0, -0.5]), None);
    }

    #[test]
    fn non_finite_spacing_is_rejected() {
        assert_eq!(usable_pixel_spacing([f64::NAN, 1.0]), None);
        assert_eq!(usable_pixel_spacing([1.0, f64::INFINITY]), None);
        assert_eq!(usable_pixel_spacing([f64::NEG_INFINITY, 1.0]), None);
    }

    /// Every value this guard admits must satisfy `Spacing::try_new`, which is
    /// the invariant the panicking `Spacing::new` asserts. Without this the two
    /// validity definitions could drift apart silently.
    #[test]
    fn admitted_values_satisfy_the_spacing_invariant() {
        for candidate in [
            [0.7, 0.65],
            [1.0, 1.0],
            [f64::MIN_POSITIVE, 1e6],
            [0.0, 1.0],
            [-1.0, 1.0],
            [f64::NAN, 1.0],
            [f64::INFINITY, 1.0],
        ] {
            if let Some(accepted) = usable_pixel_spacing(candidate) {
                assert!(
                    ritk_spatial::Spacing::<2>::try_new(accepted).is_ok(),
                    "admitted {accepted:?} that Spacing rejects"
                );
            }
        }
    }
}
