use super::{ProjectionStatistic, SlabProjection, SlabProjectionError};
use crate::LoadedVolume;
use std::sync::Arc;

fn scalar_volume(shape: [usize; 3]) -> LoadedVolume {
    let [depth, rows, cols] = shape;
    let data = (0..depth)
        .flat_map(|depth_index| {
            (0..rows).flat_map(move |row| {
                (0..cols).map(move |column| (100 * depth_index + 10 * row + column) as f32)
            })
        })
        .collect();
    LoadedVolume {
        data: Arc::new(data),
        shape,
        channels: 1,
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: None,
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    }
}

#[test]
fn projection_statistics_match_the_manufactured_volume_on_all_axes() {
    let volume = scalar_volume([3, 3, 3]);

    let axial = SlabProjection::try_new(&volume, 0, 1, 1).expect("valid axial slab");
    assert_eq!(axial.dimensions(), [3, 3]);
    assert_eq!(axial.sample_count(), 3);
    assert_eq!(
        axial
            .compute(&volume, ProjectionStatistic::Minimum)
            .expect("valid minimum projection")
            .pixels(),
        &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    );
    assert_eq!(
        axial
            .compute(&volume, ProjectionStatistic::Maximum)
            .expect("valid maximum projection")
            .pixels(),
        &[200.0, 201.0, 202.0, 210.0, 211.0, 212.0, 220.0, 221.0, 222.0]
    );
    assert_eq!(
        axial
            .compute(&volume, ProjectionStatistic::Average)
            .expect("valid average projection")
            .pixels(),
        &[100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 120.0, 121.0, 122.0]
    );

    let coronal = SlabProjection::try_new(&volume, 1, 1, 1).expect("valid coronal slab");
    assert_eq!(coronal.dimensions(), [3, 3]);
    assert_eq!(
        coronal
            .compute(&volume, ProjectionStatistic::Minimum)
            .expect("valid coronal minimum projection")
            .pixels(),
        &[0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 200.0, 201.0, 202.0]
    );
    assert_eq!(
        coronal
            .compute(&volume, ProjectionStatistic::Maximum)
            .expect("valid coronal maximum projection")
            .pixels(),
        &[20.0, 21.0, 22.0, 120.0, 121.0, 122.0, 220.0, 221.0, 222.0]
    );
    assert_eq!(
        coronal
            .compute(&volume, ProjectionStatistic::Average)
            .expect("valid coronal average projection")
            .pixels(),
        &[10.0, 11.0, 12.0, 110.0, 111.0, 112.0, 210.0, 211.0, 212.0]
    );

    let sagittal = SlabProjection::try_new(&volume, 2, 1, 1).expect("valid sagittal slab");
    assert_eq!(sagittal.dimensions(), [3, 3]);
    assert_eq!(
        sagittal
            .compute(&volume, ProjectionStatistic::Minimum)
            .expect("valid sagittal minimum projection")
            .pixels(),
        &[0.0, 10.0, 20.0, 100.0, 110.0, 120.0, 200.0, 210.0, 220.0]
    );
    assert_eq!(
        sagittal
            .compute(&volume, ProjectionStatistic::Maximum)
            .expect("valid sagittal maximum projection")
            .pixels(),
        &[2.0, 12.0, 22.0, 102.0, 112.0, 122.0, 202.0, 212.0, 222.0]
    );
    assert_eq!(
        sagittal
            .compute(&volume, ProjectionStatistic::Average)
            .expect("valid sagittal average projection")
            .pixels(),
        &[1.0, 11.0, 21.0, 101.0, 111.0, 121.0, 201.0, 211.0, 221.0]
    );
}

#[test]
fn one_sample_maximum_matches_the_existing_slice_contract() {
    let volume = scalar_volume([3, 3, 3]);
    for axis in 0..=2 {
        let extent = volume.shape[axis];
        for index in 0..extent {
            let request =
                SlabProjection::try_new(&volume, axis, index, 0).expect("one-sample slab is valid");
            let projection = request
                .compute(&volume, ProjectionStatistic::Maximum)
                .expect("one-sample projection is valid");
            let (slice, width, height) = volume.extract_slice(axis, index);
            assert_eq!(projection.dimensions(), [width, height]);
            assert_eq!(projection.pixels(), slice.as_slice());
        }
    }
}

#[test]
fn compute_into_reuses_capacity_and_rejects_shape_changes() {
    let volume = scalar_volume([3, 3, 3]);
    let request = SlabProjection::try_new(&volume, 0, 1, 1).expect("valid scratch slab");
    let mut pixels = Vec::new();
    request
        .compute_into(&volume, ProjectionStatistic::Average, &mut pixels)
        .expect("average scratch projection is valid");
    let capacity = pixels.capacity();
    request
        .compute_into(&volume, ProjectionStatistic::Minimum, &mut pixels)
        .expect("minimum scratch projection is valid");
    assert_eq!(pixels.capacity(), capacity);
    assert_eq!(pixels, [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0]);

    let changed = scalar_volume([4, 3, 3]);
    assert!(matches!(
        request.compute(&changed, ProjectionStatistic::Maximum),
        Err(SlabProjectionError::ShapeChanged { .. })
    ));
}

#[test]
fn invalid_requests_and_payloads_return_typed_errors() {
    let volume = scalar_volume([3, 2, 2]);
    assert!(matches!(
        SlabProjection::try_new(&volume, 3, 0, 0),
        Err(SlabProjectionError::InvalidAxis { axis: 3 })
    ));
    assert!(matches!(
        SlabProjection::try_new(&volume, 0, 3, 0),
        Err(SlabProjectionError::CenterOutOfBounds { .. })
    ));
    assert!(matches!(
        SlabProjection::try_new(&volume, 0, 0, 1),
        Err(SlabProjectionError::RangeOutOfBounds { .. })
    ));

    let mut rgb = volume.clone();
    rgb.channels = 3;
    assert!(matches!(
        SlabProjection::try_new(&rgb, 0, 1, 0),
        Err(SlabProjectionError::UnsupportedChannels { channels: 3 })
    ));

    let mut malformed = volume;
    Arc::make_mut(&mut malformed.data).pop();
    assert!(matches!(
        SlabProjection::try_new(&malformed, 0, 1, 0),
        Err(SlabProjectionError::MalformedPayload { .. })
    ));
}
