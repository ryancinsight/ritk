use super::{
    PixelMappingError, ProjectionStatistic, ResliceError, ResliceInterpolation, ReslicePlane,
    SlabProjection,
};
use crate::LoadedVolume;
use std::sync::Arc;

fn scalar_volume(shape: [usize; 3]) -> LoadedVolume {
    let [depth, rows, columns] = shape;
    let data = (0..depth)
        .flat_map(|depth_index| {
            (0..rows).flat_map(move |row| {
                (0..columns).map(move |column| (100 * depth_index + 10 * row + column) as f32)
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
fn axis_aligned_planes_match_existing_slice_values() {
    let volume = scalar_volume([3, 3, 3]);
    for axis in 0..=2 {
        for index in 0..volume.shape[axis] {
            let plane =
                ReslicePlane::axis_aligned(&volume, axis, index, ResliceInterpolation::Linear)
                    .expect("axis-aligned request is valid");
            let output = plane
                .compute(&volume, ProjectionStatistic::Maximum)
                .expect("axis-aligned reslice is valid");
            let (expected, width, height) = volume.extract_slice(axis, index);
            assert_eq!(output.dimensions(), [width, height]);
            assert_eq!(output.pixels(), expected.as_slice());
        }
    }
}

#[test]
fn rotated_anisotropic_plane_preserves_voxel_coordinates() {
    let mut volume = scalar_volume([2, 2, 2]);
    volume.spacing = [2.0, 3.0, 4.0];
    volume.direction = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let plane = ReslicePlane::try_new(
        &volume,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0],
        [-3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2, 2],
        2,
        ResliceInterpolation::Nearest,
    )
    .expect("rotated physical plane is valid");
    let output = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("rotated plane computes");
    assert_eq!(output.pixels(), &[100.0, 101.0, 110.0, 111.0]);
}

fn rotated_anisotropic_patient_plane() -> ReslicePlane {
    let mut volume = scalar_volume([8, 8, 8]);
    volume.spacing = [2.0, 3.0, 4.0];
    volume.origin = [10.0, 20.0, 30.0];
    volume.direction = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    ReslicePlane::try_new(
        &volume,
        [7.0, 21.0, 35.0],
        [0.0, 0.5, 2.0],
        [-1.5, 0.0, 1.0],
        [0.0; 3],
        [4, 3],
        1,
        ResliceInterpolation::Linear,
    )
    .expect("rotated anisotropic plane is valid")
}

#[test]
fn continuous_pixel_mapping_preserves_rotated_physical_geometry() {
    let plane = rotated_anisotropic_patient_plane();
    assert_eq!(
        plane
            .patient_at_pixel([1.5, 0.5])
            .expect("interior coordinate maps to patient space")
            .coordinates(),
        [6.25, 21.75, 38.5]
    );

    for pixel in [[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [3.0, 2.0]] {
        let expected = [
            7.0 - 1.5 * pixel[1],
            21.0 + 0.5 * pixel[0],
            35.0 + 2.0 * pixel[0] + pixel[1],
        ];
        assert_eq!(
            plane
                .patient_at_pixel(pixel)
                .expect("boundary coordinate maps to patient space")
                .coordinates(),
            expected
        );
    }
}

#[test]
fn continuous_pixel_mapping_rejects_invalid_and_out_of_bounds_coordinates() {
    let plane = rotated_anisotropic_patient_plane();
    for axis in 0..2 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let pixel =
                std::array::from_fn(|component| if component == axis { invalid } else { 0.5 });
            assert!(matches!(
                plane.patient_at_pixel(pixel),
                Err(PixelMappingError::InvalidCoordinate { .. })
            ));
        }
    }

    for pixel in [
        [-f64::EPSILON, 0.5],
        [4.0, 0.5],
        [1.5, -f64::EPSILON],
        [1.5, 3.0],
    ] {
        assert!(matches!(
            plane.patient_at_pixel(pixel),
            Err(PixelMappingError::OutOfBounds { .. })
        ));
    }
}

#[test]
fn trilinear_resampling_reproduces_a_linear_field() {
    let mut volume = scalar_volume([2, 2, 2]);
    volume.data = Arc::new(
        (0..2)
            .flat_map(|depth| {
                (0..2).flat_map(move |row| {
                    (0..2).map(move |column| (2 * depth + 3 * row + 5 * column) as f32)
                })
            })
            .collect(),
    );
    let plane = ReslicePlane::try_new(
        &volume,
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [1, 1],
        1,
        ResliceInterpolation::Linear,
    )
    .expect("fractional plane is valid");
    let output = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("fractional plane computes");
    let expected = 5.0_f32;
    assert!((output.pixels()[0] - expected).abs() <= 16.0 * f32::EPSILON);
}

#[test]
fn slab_statistics_share_the_physical_plane_contract() {
    let volume = scalar_volume([3, 3, 3]);
    let slab = SlabProjection::try_new(&volume, 0, 1, 1).expect("valid slab");
    let plane = ReslicePlane::from_slab(&volume, slab, ResliceInterpolation::Nearest)
        .expect("slab plane is valid");
    assert_eq!(
        plane
            .compute(&volume, ProjectionStatistic::Average)
            .expect("average slab computes")
            .pixels(),
        &[100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 120.0, 121.0, 122.0]
    );
}

#[test]
fn compute_into_reuses_storage_and_rejects_changed_geometry() {
    let volume = scalar_volume([2, 2, 2]);
    let plane = ReslicePlane::axis_aligned(&volume, 0, 0, ResliceInterpolation::Nearest)
        .expect("valid plane");
    let mut pixels = Vec::new();
    plane
        .compute_into(&volume, ProjectionStatistic::Maximum, &mut pixels)
        .expect("first render computes");
    let capacity = pixels.capacity();
    plane
        .compute_into(&volume, ProjectionStatistic::Minimum, &mut pixels)
        .expect("second render computes");
    assert_eq!(pixels.capacity(), capacity);

    let mut changed = volume.clone();
    changed.spacing[0] = 2.0;
    assert!(matches!(
        plane.compute(&changed, ProjectionStatistic::Maximum),
        Err(ResliceError::GeometryChanged)
    ));
}

#[test]
fn invalid_planes_and_source_layouts_return_typed_errors() {
    let volume = scalar_volume([2, 2, 2]);
    assert!(matches!(
        ReslicePlane::try_new(
            &volume,
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0; 3],
            [2, 2],
            1,
            ResliceInterpolation::Nearest,
        ),
        Err(ResliceError::InvalidPlaneBasis)
    ));
    assert!(matches!(
        ReslicePlane::try_new(
            &volume,
            [5.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1, 1],
            2,
            ResliceInterpolation::Nearest,
        ),
        Err(ResliceError::OutOfVolume { .. })
    ));

    let mut rgb = volume.clone();
    rgb.channels = 3;
    assert!(matches!(
        ReslicePlane::axis_aligned(&rgb, 0, 0, ResliceInterpolation::Nearest),
        Err(ResliceError::UnsupportedChannels { channels: 3 })
    ));
}
