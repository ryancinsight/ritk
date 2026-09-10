//! Tests for the loader module.

pub(crate) mod fixtures;

mod multiframe;
mod presentation;
mod selection;
mod volumes;

fn assert_study(volume: &crate::LoadedVolume, modality: &str) {
    // Integer samples and powers-of-two rescale coefficients are exact in f32.
    let expected: Vec<f32> = fixtures::SAMPLES
        .iter()
        .map(|&raw| 2.0 * f32::from(raw) - 20.0)
        .collect();
    assert_eq!(volume.data.as_slice(), expected);
    assert_eq!(volume.shape, fixtures::SHAPE);
    assert_eq!(volume.channels, 1);
    assert_eq!(volume.spacing, fixtures::SPACING);
    assert_eq!(volume.origin, fixtures::ORIGIN);
    assert_eq!(volume.direction, fixtures::DIRECTION);
    assert_eq!(volume.modality.as_deref(), Some(modality));
    let metadata = volume.metadata.as_ref().expect("DICOM metadata retained");
    assert_eq!(
        metadata.series_instance_uid.as_deref(),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        metadata.study_instance_uid.as_deref(),
        Some("2.25.20260905")
    );
    // P(d,r,c) = (10+2d, 20+0.5c, 30+1.5r), independently from the encoded IOP/IPP.
    assert_eq!(
        crate::ui::voxel_to_lps([2, 1, 3], volume.origin, volume.direction, volume.spacing),
        [14.0, 21.5, 31.5]
    );
}
