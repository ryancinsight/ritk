use super::{
    ProjectionStatistic, ResliceError, ResliceInterpolation, ResliceOrientation, ReslicePlane,
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

#[test]
fn continuous_pixel_mapping_preserves_rotated_physical_geometry() {
    let mut volume = scalar_volume([4, 5, 6]);
    volume.spacing = [2.0, 3.0, 4.0];
    volume.origin = [10.0, 20.0, 30.0];
    volume.direction = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    volume.data = Arc::new(
        (0..4)
            .flat_map(|depth| {
                (0..5).flat_map(move |row| {
                    (0..6).map(move |column| (100 * depth + 10 * row + column) as f32)
                })
            })
            .collect(),
    );
    let plane = ReslicePlane::try_new(
        &volume,
        [7.0, 21.0, 35.0],
        [0.0, 0.5, 2.0],
        [-1.5, 0.0, 1.0],
        [0.0; 3],
        [4, 3],
        1,
        ResliceInterpolation::Linear,
    )
    .expect("rotated anisotropic plane is valid");

    let sample = plane
        .sample_pixel(&volume, [1.5, 0.5])
        .expect("interior output coordinate maps and samples");
    assert_eq!(sample.pixel(), [1.5, 0.5]);
    assert_eq!(sample.patient(), [6.25, 21.75, 38.5]);
    let patient = plane
        .patient_at_pixel([1.5, 0.5])
        .expect("interior output coordinate maps to patient space");
    assert_eq!(patient.coordinates(), [6.25, 21.75, 38.5]);
    let projection = plane
        .project_patient([6.25, 21.75, 38.5])
        .expect("analytical patient point projects into the output plane");
    assert_eq!(projection.distance_mm(), 0.0);
    for coordinate in [
        [f64::NAN, 21.0, 35.0],
        [7.0, f64::NAN, 35.0],
        [7.0, 21.0, f64::NAN],
        [f64::INFINITY, 21.0, 35.0],
        [7.0, f64::INFINITY, 35.0],
        [7.0, 21.0, f64::INFINITY],
        [f64::NEG_INFINITY, 21.0, 35.0],
        [7.0, f64::NEG_INFINITY, 35.0],
        [7.0, 21.0, f64::NEG_INFINITY],
    ] {
        assert!(matches!(
            plane.project_patient(coordinate),
            Err(ResliceError::InvalidPatientPoint { .. })
        ));
    }
    for coordinate in [
        [6.125, 21.0, 35.0],
        [6.25, 23.0, 43.5],
        [7.375, 21.5, 36.75],
        [2.5, 21.5, 40.0],
    ] {
        assert!(matches!(
            plane.project_patient(coordinate),
            Err(ResliceError::PixelOutOfBounds { .. })
        ));
    }
    assert_eq!(sample.voxel(), [0.875, 1.25, 2.125]);
    assert_eq!(sample.nearest_voxel(), [1, 1, 2]);
    assert_eq!(sample.value(), 102.125);
}

#[test]
fn continuous_pixel_mapping_rejects_invalid_coordinates_and_changed_sources() {
    let volume = scalar_volume([3, 4, 5]);
    let plane = ReslicePlane::axis_aligned(&volume, 0, 1, ResliceInterpolation::Linear)
        .expect("axis-aligned plane is valid");

    assert!(matches!(
        plane.sample_pixel(&volume, [f64::NAN, 0.0]),
        Err(ResliceError::InvalidPixelCoordinate { .. })
    ));
    for coordinate in [
        [f64::NAN, 0.0],
        [0.0, f64::NAN],
        [f64::INFINITY, 0.0],
        [0.0, f64::NEG_INFINITY],
    ] {
        assert!(matches!(
            plane.patient_at_pixel(coordinate),
            Err(ResliceError::InvalidPixelCoordinate { .. })
        ));
    }
    for coordinate in [[-f64::EPSILON, 0.0], [5.0, 0.0], [0.0, 4.0]] {
        assert!(matches!(
            plane.sample_pixel(&volume, coordinate),
            Err(ResliceError::PixelOutOfBounds { .. })
        ));
        assert!(matches!(
            plane.patient_at_pixel(coordinate),
            Err(ResliceError::PixelOutOfBounds { .. })
        ));
    }
    let mut changed_shape = volume.clone();
    changed_shape.shape = [2, 4, 5];
    changed_shape.data = Arc::new(changed_shape.data[..40].to_vec());
    assert!(matches!(
        plane.sample_pixel(&changed_shape, [0.0, 0.0]),
        Err(ResliceError::ShapeChanged { .. })
    ));

    let mut changed_geometry = volume.clone();
    changed_geometry.spacing[0] = 2.0;
    assert!(matches!(
        plane.sample_pixel(&changed_geometry, [0.0, 0.0]),
        Err(ResliceError::GeometryChanged)
    ));
}

#[test]
fn patient_projection_uses_componentwise_forward_rounding_bounds() {
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
    .expect("unit plane matches the source volume");

    for pixel in [
        [0.0, 0.0],
        [4.0, 0.0],
        [0.0, 3.0],
        [4.0, 3.0],
        [0.0, 2.5],
        [3.75, 3.0],
        [2.5, 0.0],
        [1.25, 3.0],
    ] {
        let patient = plane
            .patient_at_pixel(pixel)
            .expect("in-bounds pixel maps to patient space");
        let projection = plane
            .project_patient(patient.coordinates())
            .expect("forward-mapped edge and fractional pixels project back");
        assert_eq!(projection.pixel(), pixel);
    }

    let distant_normal = plane
        .project_patient([0.5, 0.5, 1.0e13])
        .expect("normal distance does not widen independent pixel axes");
    assert_eq!(distant_normal.pixel(), [0.5, 0.5]);

    assert!(matches!(
        plane.project_patient([-0.125, 0.5, 1.0e13]),
        Err(ResliceError::PixelOutOfBounds { .. })
    ));
    assert!(matches!(
        plane.project_patient([-f64::EPSILON, 0.5, 0.0]),
        Err(ResliceError::PixelOutOfBounds { .. })
    ));
}

#[test]
fn patient_projection_round_trips_fractional_edges_on_oblique_anisotropic_plane() {
    let volume = scalar_volume([16, 24, 24]);
    let plane = ReslicePlane::try_new(
        &volume,
        [1.0, 8.0, 1.0],
        [2.0, 4.0, 4.0],
        [2.0, -2.0, 1.0],
        [0.0; 3],
        [4, 3],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("oblique anisotropic plane lies inside the source");

    for pixel in [[0.0, 0.5], [3.0, 0.75], [1.5, 0.0], [1.25, 2.0]] {
        let patient = plane
            .patient_at_pixel(pixel)
            .expect("fractional edge pixel maps to patient space");
        let projection = plane
            .project_patient(patient.coordinates())
            .expect("forward-mapped fractional edge remains in bounds");
        let enclosure = plane
            .patient_pixel_enclosure(patient.coordinates())
            .expect("finite patient point has a bounded pixel projection");
        for ((reference, projected), bound) in
            pixel.into_iter().zip(projection.pixel()).zip(enclosure)
        {
            assert!(bound.contains(reference));
            assert!(bound.contains(projected));
        }
    }

    let edge = plane
        .patient_at_pixel([0.0, 0.5])
        .expect("edge pixel maps to patient space")
        .coordinates();
    let off_plane = [edge[0] + 12.0, edge[1] + 6.0, edge[2] - 12.0];
    let projection = plane
        .project_patient(off_plane)
        .expect("finite off-plane point retains its physical projection");
    assert_eq!(projection.pixel(), [0.0, 0.5]);
    assert_eq!(projection.distance_mm(), 18.0);
}

#[test]
fn patient_projection_handles_basis_with_subnormal_squared_length() {
    let volume = scalar_volume([2, 2, 2]);
    let horizontal_step = 2.0_f64.powi(-537);
    let plane = ReslicePlane::try_new(
        &volume,
        [0.0; 3],
        [horizontal_step, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0; 3],
        [2, 2],
        1,
        ResliceInterpolation::Nearest,
    )
    .expect("nonzero minimum representable plane basis is valid");
    let patient = plane
        .patient_at_pixel([1.0, 0.5])
        .expect("tiny physical step maps to patient space");
    let projection = plane
        .project_patient(patient.coordinates())
        .expect("scaled norm keeps the nonzero basis invertible");
    assert_eq!(projection.pixel(), [1.0, 0.5]);
}

#[test]
fn centered_oblique_plane_preserves_aspect_and_samples_linear_field() {
    let mut volume = scalar_volume([7, 9, 11]);
    volume.spacing = [4.0, 2.0, 1.0];
    volume.origin = [13.0, -7.0, 21.0];
    volume.direction = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    volume.data = Arc::new(
        (0..7)
            .flat_map(|depth| {
                (0..9).flat_map(move |row| {
                    (0..11).map(move |column| (2 * depth + 3 * row + 5 * column) as f32)
                })
            })
            .collect(),
    );
    let orientation =
        ResliceOrientation::try_new(23.0, -17.0).expect("bounded orientation is valid");
    let plane = ReslicePlane::centered_oblique(
        &volume,
        [3.0, 4.0, 5.0],
        orientation,
        ResliceInterpolation::Linear,
    )
    .expect("centered oblique plane fits the volume");

    let horizontal_spacing = plane
        .horizontal_step()
        .map(|value| value * value)
        .into_iter()
        .sum::<f64>()
        .sqrt();
    let vertical_spacing = plane
        .vertical_step()
        .map(|value| value * value)
        .into_iter()
        .sum::<f64>()
        .sqrt();
    let physical_width = (plane.dimensions()[0] - 1) as f64 * horizontal_spacing;
    let physical_height = (plane.dimensions()[1] - 1) as f64 * vertical_spacing;
    let spacing_aspect = horizontal_spacing / vertical_spacing;
    assert!((spacing_aspect - 0.5).abs() <= 64.0 * f64::EPSILON);
    assert!(physical_width > 0.0);
    assert!(physical_height > 0.0);

    let pixels = plane
        .compute(&volume, ProjectionStatistic::Maximum)
        .expect("oblique plane computes");
    let [width, height] = pixels.dimensions();
    for row in [0, height / 2, height - 1] {
        for column in [0, width / 2, width - 1] {
            #[expect(
                clippy::cast_precision_loss,
                reason = "test coordinates are bounded by the committed small fixture"
            )]
            let pixel = [column as f64, row as f64];
            let sample = plane
                .sample_pixel(&volume, pixel)
                .expect("rendered pixel maps to a source sample");
            let expected = (2.0 * sample.voxel()[0]
                + 3.0 * sample.voxel()[1]
                + 5.0 * sample.voxel()[2]) as f32;
            assert!((sample.value() - expected).abs() <= 64.0 * f32::EPSILON);
            assert_eq!(pixels.pixels()[row * width + column], sample.value());
        }
    }
}

#[test]
fn oriented_oblique_plane_keeps_its_exact_patient_centre() {
    let mut volume = scalar_volume([7, 9, 11]);
    volume.spacing = [4.0, 2.0, 1.0];
    volume.origin = [13.0, -7.0, 21.0];
    volume.direction = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let initial = ReslicePlane::centered_oblique(
        &volume,
        [3.0, 4.0, 5.0],
        ResliceOrientation::default(),
        ResliceInterpolation::Linear,
    )
    .expect("initial oblique plane is valid");
    let initial_center = initial
        .sample_pixel(
            &volume,
            [
                (initial.dimensions()[0] - 1) as f64 * 0.5,
                (initial.dimensions()[1] - 1) as f64 * 0.5,
            ],
        )
        .expect("plane centre samples the source")
        .patient();
    let rotated = ReslicePlane::oblique_at_patient(
        &volume,
        initial_center,
        ResliceOrientation::try_new(25.0, -13.0).expect("bounded orientation"),
        ResliceInterpolation::Linear,
    )
    .expect("rotated plane around the retained patient centre");
    let rotated_center = rotated
        .sample_pixel(
            &volume,
            [
                (rotated.dimensions()[0] - 1) as f64 * 0.5,
                (rotated.dimensions()[1] - 1) as f64 * 0.5,
            ],
        )
        .expect("rotated plane centre samples the source")
        .patient();
    for (actual, expected) in rotated_center.into_iter().zip(initial_center) {
        let bound = 32.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!((actual - expected).abs() <= bound);
    }
}

#[test]
fn centered_oblique_plane_rejects_bad_center_orientation_and_depth_shift() {
    let volume = scalar_volume([5, 7, 9]);
    let orientation = ResliceOrientation::default();
    assert!(matches!(
        ResliceOrientation::try_new(f64::NAN, 0.0),
        Err(ResliceError::InvalidOrientation { .. })
    ));
    assert!(matches!(
        ReslicePlane::centered_oblique(
            &volume,
            [2.0, 3.0, 9.0],
            orientation,
            ResliceInterpolation::Linear,
        ),
        Err(ResliceError::InvalidCenter { .. })
    ));
    let plane = ReslicePlane::centered_oblique(
        &volume,
        [2.0, 3.0, 4.0],
        orientation,
        ResliceInterpolation::Linear,
    )
    .expect("centered plane is valid");
    assert!(matches!(
        plane.shifted_along_depth(&volume, f64::NAN),
        Err(ResliceError::InvalidDepthOffset { .. })
    ));
    assert!(matches!(
        plane.shifted_along_depth(&volume, 20.0),
        Err(ResliceError::OutOfVolume { .. })
    ));
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
