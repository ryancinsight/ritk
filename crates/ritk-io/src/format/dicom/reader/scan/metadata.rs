//! Assemble reader metadata for the selected DICOM series.

use std::collections::HashMap;

use super::super::types::{DicomReadMetadata, DicomSliceMetadata, SeriesFirstSeen, SeriesGeometry};
use crate::format::dicom::object_model::{DicomObjectModel, DicomPreservationSet};

/// Assemble a [`DicomReadMetadata`] from a [`SeriesFirstSeen`] accumulator, sorted slices,
/// and computed geometry fields.
pub(super) fn assemble_metadata(
    first: SeriesFirstSeen,
    slices: Vec<DicomSliceMetadata>,
    geometry: SeriesGeometry,
    series_object: DicomObjectModel,
) -> DicomReadMetadata {
    let SeriesGeometry {
        rows,
        cols,
        spacing,
        origin,
        direction,
    } = geometry;
    DicomReadMetadata {
        series_instance_uid: first.series_instance_uid,
        study_instance_uid: first.study_instance_uid,
        frame_of_reference_uid: first.frame_of_reference_uid,
        series_description: first.series_description,
        modality: first.modality,
        patient_id: first.patient_id,
        patient_name: first.patient_name,
        study_date: first.study_date,
        series_date: first.series_date,
        series_time: first.series_time,
        dimensions: [rows, cols, slices.len()],
        spacing,
        origin,
        direction,
        bits_allocated: first.bits_allocated,
        bits_stored: first.bits_stored,
        high_bit: first.high_bit,
        photometric_interpretation: first.photometric_interpretation,
        slices,
        private_tags: HashMap::new(),
        preservation: DicomPreservationSet {
            object: series_object,
            preserved: Vec::new(),
            // Per-slice retention losses stay on their own slice so each one
            // keeps an exact frame scope; see `inventory::dicom_metadata_losses`.
            losses: Vec::new(),
        },
        enhanced_pet_suv: None,
        enhanced_pet_selection: None,
        classic_pet_calibration: first.classic_pet_calibration.common(),
        patient_weight_kg: first.patient_weight_kg,
        decay_correction: first.decay_correction,
        pet_pixels_in_bq_per_ml: first.pet_pixels_in_bq_per_ml.unwrap_or(false),
        radionuclide_total_dose_bq: first.radionuclide_total_dose_bq,
        radiopharmaceutical_start_time: first.radiopharmaceutical_start_time,
        radionuclide_half_life_s: first.radionuclide_half_life_s,
    }
}
