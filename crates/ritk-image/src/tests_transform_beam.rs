//! Transform contracts for beam-space acquisitions.
//!
//! Curvilinear and phased-array images index beams and samples, not a
//! Cartesian raster, so `origin + D S index` is meaningless for them. These
//! cover the paths where the coordinate map is load-bearing; the Cartesian
//! contracts live in `tests_transform.rs`.

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};

use crate::test_support::{curvilinear_image, metadata_only_image, phased_array_image};
/// The single-point transform must honour the attached coordinate map.
///
/// A curvilinear image indexes beams and samples, not a Cartesian raster,
/// so `origin + D S index` places its points in no physical space at all —
/// index (32, 63) would land at 32 m by 63 m rather than 66 mm by 9 mm.
/// The batch form is the independent oracle: it has always routed through
/// the map, so agreement pins the single-point path to it.
///
/// Tolerance: `f32` storage carries ~1.2e-7 relative precision and the
/// radii here reach 0.07 m, so the two paths may differ by ~1e-8 through
/// the polar trig. 1e-6 stays well above that and far below the ~30 m
/// error the Cartesian formula produces.
#[test]
fn single_point_transform_honours_the_curvilinear_map() {
    let img = curvilinear_image();
    // Far sample on the centre beam: deep in the fan, where the polar
    // mapping is furthest from the raw index pair.
    let index = Point::<2>::new([32.0, 63.0]);
    let point = img.continuous_index_to_physical_point(&index);

    // Batch column c corresponds to axis D-1-c on the index side.
    let batch = img.index_to_world_native(&Tensor::<f32, SequentialBackend>::from_slice(
        [1, 2],
        &[index[1] as f32, index[0] as f32],
    ));

    for axis in 0..2 {
        assert!(
            (f64::from(batch.as_slice()[axis]) - point[axis]).abs() <= 1e-6,
            "axis {axis}: batch={}, single-point={}",
            batch.as_slice()[axis],
            point[axis]
        );
    }
}

/// Beam index -> physical point -> beam index, through the real `Image`
/// transforms rather than the geometry helper, so the column conventions
/// are covered too.
///
/// Tolerance: `f32` storage carries ~1e-7 relative precision, and the
/// sample index divides the radius error by `radius_sample_size = 1e-4`,
/// so an absolute index error up to ~1e-1 sample is expected at the far
/// field. Asserting 0.05 keeps that meaningful while remaining above the
/// narrowing floor.
#[test]
fn curvilinear_index_round_trips_through_the_image_transforms() {
    let img = curvilinear_image();
    let indices: Vec<f32> = vec![
        0.0, 0.0, // sample 0, beam 0
        63.0, 32.0, // far sample, centre beam
        10.0, 16.0, 40.0, 5.0,
    ];
    let idx_t = Tensor::<f32, SequentialBackend>::from_slice([4, 2], &indices);

    let world = img.index_to_world_native(&idx_t);
    let back = img.world_to_index_native(&world);

    for (row, chunk) in back.as_slice().chunks_exact(2).enumerate() {
        for col in 0..2 {
            let want = indices[row * 2 + col];
            assert!(
                (chunk[col] - want).abs() < 0.05,
                "row {row} col {col}: {} != {want}",
                chunk[col]
            );
        }
    }
}

/// The fan must actually curve: beams either side of centre map to mirrored
/// lateral positions at equal axial depth, which a Cartesian map could not
/// produce from a rectangular index grid.
#[test]
fn curvilinear_fan_is_symmetric_and_curved() {
    let img = curvilinear_image();
    // Same sample on the outermost beams (0 and 32), centre beam 16.
    let indices: Vec<f32> = vec![63.0, 0.0, 63.0, 32.0, 63.0, 16.0];
    let idx_t = Tensor::<f32, SequentialBackend>::from_slice([3, 2], &indices);
    let world = img.index_to_world_native(&idx_t);
    let w = world.as_slice();
    // Columns are axis-major: axis 1 = lateral (r sin), axis 0 = axial (r cos).
    let (left_lat, left_ax) = (w[1], w[0]);
    let (right_lat, right_ax) = (w[3], w[2]);
    let (centre_lat, centre_ax) = (w[5], w[4]);

    assert!(
        (left_lat + right_lat).abs() < 1.0e-6,
        "outer beams must mirror: {left_lat} vs {right_lat}"
    );
    assert!(
        (left_ax - right_ax).abs() < 1.0e-6,
        "outer beams share a depth: {left_ax} vs {right_ax}"
    );
    assert!(
        centre_lat.abs() < 1.0e-6,
        "centre beam must be axial, got {centre_lat}"
    );
    // Curvature: the centre beam reaches deeper than the outer ones.
    assert!(
        centre_ax > left_ax,
        "fan must curve: centre {centre_ax} should exceed outer {left_ax}"
    );
}

/// A point behind the apex plane has no beam; the batch form marks it NaN
/// rather than aliasing it onto a real index.
#[test]
fn points_outside_the_fan_become_nan_indices() {
    let img = curvilinear_image();
    // axis-major (axial, lateral): axial <= 0 is outside the acquisition.
    let pts = [-0.05_f32, 0.01];
    let pts_t = Tensor::<f32, SequentialBackend>::from_slice([1, 2], &pts);
    let idx = img.world_to_index_native(&pts_t);
    assert!(idx.as_slice()[0].is_nan(), "sample index must be NaN");
    assert!(idx.as_slice()[1].is_nan(), "beam index must be NaN");
}

/// Boresight through the real `Image` transform: the centre azimuth and
/// elevation beams must produce a point with no lateral or elevation
/// offset, confirming the column-to-axis wiring.
#[test]
fn phased_array_boresight_through_the_image_transform() {
    let img = phased_array_image();
    // shape [8, 5, 9] -> azimuth_count = shape[2] = 9 (centre 4),
    // elevation_count = shape[1] = 5 (centre 2).
    let indices = [4.0_f32, 2.0, 50.0];
    let idx_t = Tensor::<f32, SequentialBackend>::from_slice([1, 3], &indices);
    let world = img.index_to_world_native(&idx_t);
    let w = world.as_slice();
    // axis-major: axis 2 = azimuth, axis 1 = elevation, axis 0 = depth.
    assert!(w[2].abs() < 1.0e-6, "azimuth offset {}", w[2]);
    assert!(w[1].abs() < 1.0e-6, "elevation offset {}", w[1]);
    let expected = 0.01 + 50.0 * 1.0e-4;
    assert!((w[0] - expected).abs() < 1.0e-6, "depth {}", w[0]);
}

/// Index -> point -> index through the real transforms, covering steered
/// rays in both angles. Tolerance follows the geometry-level reasoning,
/// loosened for `f32` storage.
#[test]
fn phased_array_round_trips_through_the_image_transforms() {
    let img = phased_array_image();
    let indices: Vec<f32> = vec![
        4.0, 2.0, 0.0, // boresight, first sample
        0.0, 0.0, 60.0, // both angles steered to a corner
        8.0, 4.0, 30.0, // opposite corner
        6.0, 1.0, 75.0,
    ];
    let idx_t = Tensor::<f32, SequentialBackend>::from_slice([4, 3], &indices);
    let world = img.index_to_world_native(&idx_t);
    let back = img.world_to_index_native(&world);

    for (row, chunk) in back.as_slice().chunks_exact(3).enumerate() {
        for col in 0..3 {
            let want = indices[row * 3 + col];
            assert!(
                (chunk[col] - want).abs() < 0.05,
                "row {row} col {col}: {} != {want}",
                chunk[col]
            );
        }
    }
}

/// Steering must be independent per angle: moving only the azimuth beam
/// must leave the elevation offset at zero, which a single spherical polar
/// angle would not do.
#[test]
fn phased_array_angles_steer_independently() {
    let img = phased_array_image();
    let indices = [0.0_f32, 2.0, 50.0, 8.0, 2.0, 50.0];
    let idx_t = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &indices);
    let w = img.index_to_world_native(&idx_t);
    let w = w.as_slice();
    // Elevation stays on boresight for both azimuth-steered rays.
    assert!(w[1].abs() < 1.0e-6, "elevation leaked: {}", w[1]);
    assert!(w[4].abs() < 1.0e-6, "elevation leaked: {}", w[4]);
    // Azimuth offsets mirror, depths match.
    assert!((w[2] + w[5]).abs() < 1.0e-6, "azimuth must mirror");
    assert!((w[0] - w[3]).abs() < 1.0e-6, "depth must match");
}

#[test]
fn phased_array_points_behind_the_array_become_nan() {
    let img = phased_array_image();
    // axis-major (depth, elevation, azimuth): depth <= 0 is behind the array.
    let pts = [-0.02_f32, 0.001, 0.001];
    let pts_t = Tensor::<f32, SequentialBackend>::from_slice([1, 3], &pts);
    let idx = img.world_to_index_native(&pts_t);
    assert!(idx.as_slice().iter().all(|v| v.is_nan()));
}

#[test]
fn phased_array_map_is_rejected_outside_three_dimensions() {
    let geometry = ritk_spatial::PhasedArray3D::centred(
        1.0e-4,
        0.01,
        0.75_f64.to_radians(),
        1.5_f64.to_radians(),
        9,
        5,
    )
    .expect("valid geometry");
    let img = metadata_only_image::<f64, SequentialBackend, 2>(
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    );
    assert!(img
        .with_coordinate_map(CoordinateMap::PhasedArray3D(geometry))
        .is_err());
}

#[test]
fn curvilinear_map_is_rejected_on_a_one_dimensional_image() {
    let geometry = ritk_spatial::CurvilinearArray::centred(1.0e-4, 0.06, 0.5_f64.to_radians(), 33)
        .expect("valid geometry");
    let img = metadata_only_image::<f64, SequentialBackend, 1>(
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    );
    assert!(img
        .with_coordinate_map(CoordinateMap::CurvilinearArray(geometry))
        .is_err());
}
