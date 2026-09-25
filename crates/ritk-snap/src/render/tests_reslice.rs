use super::{
    PixelMappingError, ProjectionStatistic, ResliceError, ResliceInterpolation, ReslicePlane,
    SlabProjection,
};
use crate::LoadedVolume;
use std::sync::Arc;

pub(super) fn scalar_volume(shape: [usize; 3]) -> LoadedVolume {
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

#[path = "tests_reslice/orientation.rs"]
mod orientation;

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

#[test]
fn patient_projection_round_trips_skewed_anisotropic_pixels() {
    let plane = rotated_anisotropic_patient_plane();
    for pixel in [[0.0, 0.5], [3.0, 0.75], [1.5, 0.0], [1.25, 2.0]] {
        let point = plane
            .patient_at_pixel(pixel)
            .expect("in-bounds pixel maps to patient space");
        let projection = plane
            .project_patient(point.coordinates())
            .expect("round-tripped patient point projects into the plane");
        let enclosure = plane
            .patient_pixel_enclosure(point.coordinates())
            .expect("finite point has a bounded pixel projection");
        // The rounded forward and inverse maps need not reproduce an edge
        // coordinate bit-for-bit; the enclosure is the round-trip oracle.
        let maximum = [3.0, 2.0];
        for (((bound, expected), projected), limit) in enclosure
            .into_iter()
            .zip(pixel)
            .zip(projection.pixel())
            .zip(maximum)
        {
            assert!(bound.contains(expected));
            assert!(bound.contains(projected));
            assert!((0.0..=limit).contains(&projected));
        }
    }
}

#[test]
fn patient_projection_preserves_componentwise_bounds_at_large_normal_offsets() {
    let volume = scalar_volume([5, 4, 3]);
    let plane = ReslicePlane::try_new(
        &volume,
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [5, 4],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("unit plane fits the source volume");

    let interior = plane
        .project_patient([0.5, 0.5, 1.0e13])
        .expect("normal displacement does not widen the in-plane axes");
    assert_eq!(interior.pixel(), [0.5, 0.5]);
    assert_eq!(interior.distance_mm(), 1.0e13);
    let negative = plane
        .project_patient([0.5, 0.5, -18.0])
        .expect("point on the negative normal side projects into the plane");
    assert_eq!(negative.pixel(), [0.5, 0.5]);
    assert_eq!(negative.distance_mm(), -18.0);

    let tiny_in_plane = plane
        .project_patient([1.0e-20, 0.5, 1.0e308])
        .expect("normal displacement preserves representable in-plane coordinates");
    assert_eq!(tiny_in_plane.pixel(), [1.0e-20, 0.5]);
    assert_eq!(tiny_in_plane.distance_mm(), 1.0e308);

    assert!(matches!(
        plane.project_patient([-0.125, 0.5, 1.0e13]),
        Err(PixelMappingError::OutOfBounds { .. })
    ));
    assert!(matches!(
        plane.project_patient([-f64::EPSILON, 0.5, 0.0]),
        Err(PixelMappingError::OutOfBounds { .. })
    ));
}

#[test]
fn patient_projection_rejects_indistinguishable_adjacent_pixel_centres() {
    let mut volume = scalar_volume([2, 2, 2]);
    volume.origin = [1.0e16, 0.0, 0.0];
    let plane = ReslicePlane::try_new(
        &volume,
        [1.0e16, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2, 2],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("finite plane geometry is valid");

    let first = plane
        .patient_at_pixel([0.0, 0.0])
        .expect("first pixel centre maps to patient space");
    let adjacent = plane
        .patient_at_pixel([1.0, 0.0])
        .expect("adjacent pixel centre maps to patient space");
    assert_eq!(first, adjacent);
    assert!(matches!(
        plane.project_patient(first.coordinates()),
        Err(PixelMappingError::ProjectionUnresolved { .. })
    ));
}

#[test]
fn patient_projection_reports_signed_distance_for_a_rotated_plane() {
    let volume = scalar_volume([2, 2, 2]);
    let plane = ReslicePlane::try_new(
        &volume,
        [0.0; 3],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [2, 2],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("rotated plane geometry is valid");

    let positive = plane
        .project_patient([2.5, 0.25, 0.75])
        .expect("point projects into the rotated plane");
    assert_eq!(positive.pixel(), [0.25, 0.75]);
    assert_eq!(positive.distance_mm(), 2.5);

    let negative = plane
        .project_patient([-1.25, 0.25, 0.75])
        .expect("point projects from the negative normal side");
    assert_eq!(negative.pixel(), [0.25, 0.75]);
    assert_eq!(negative.distance_mm(), -1.25);
}

#[test]
fn patient_projection_returns_typed_errors_for_invalid_unresolved_and_overflowing_points() {
    let plane = rotated_anisotropic_patient_plane();
    assert!(matches!(
        plane.project_patient([f64::NAN, 21.0, 35.0]),
        Err(PixelMappingError::InvalidPatientPoint { .. })
    ));

    let mut volume = scalar_volume([2, 2, 2]);
    volume.origin = [f64::MAX, 0.0, 0.0];
    let large_origin_plane = ReslicePlane::try_new(
        &volume,
        [f64::MAX, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2, 2],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("finite translated plane is valid");
    assert!(matches!(
        large_origin_plane.project_patient([f64::MAX, 0.0, 0.0]),
        Err(PixelMappingError::ProjectionUnresolved { .. })
    ));

    let volume = scalar_volume([32, 32, 32]);
    let diagonal_plane = ReslicePlane::try_new(
        &volume,
        [4.0, 4.0, 8.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, -2.0],
        [0.0; 3],
        [1, 1],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("diagonal plane is valid");
    assert!(matches!(
        diagonal_plane.project_patient([1.7e308; 3]),
        Err(PixelMappingError::ProjectionOverflow { .. })
    ));
}
