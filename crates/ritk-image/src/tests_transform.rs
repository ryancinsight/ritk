use crate::types::Image;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;
type Point3 = Point<3>;
type Spacing3 = Spacing<3>;
type Direction3 = Direction<3>;

#[test]
fn test_physical_to_index_transform() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let point = Point3::new([5.0, 5.0, 5.0]);
    let index = image
        .physical_point_to_continuous_index(&point)
        .expect("Cartesian image with an invertible direction has an index for every point");

    assert!((index[0] - 5.0).abs() < 1e-6);
    assert!((index[1] - 5.0).abs() < 1e-6);
    assert!((index[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_index_to_physical_transform() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let index = Point3::new([5.0, 5.0, 5.0]);
    let point = image.continuous_index_to_physical_point(&index);

    assert!((point[0] - 5.0).abs() < 1e-6);
    assert!((point[1] - 5.0).abs() < 1e-6);
    assert!((point[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_transform_roundtrip() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let original_point = Point3::new([3.5, 4.5, 5.5]);
    let index = image
        .physical_point_to_continuous_index(&original_point)
        .expect("Cartesian image with an invertible direction has an index for every point");
    let transformed_point = image.continuous_index_to_physical_point(&index);

    assert!((original_point[0] - transformed_point[0]).abs() < 1e-6);
    assert!((original_point[1] - transformed_point[1]).abs() < 1e-6);
    assert!((original_point[2] - transformed_point[2]).abs() < 1e-6);
}

#[test]
fn test_non_unit_spacing() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([2.0, 2.0, 2.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let point = Point3::new([10.0, 10.0, 10.0]);
    let index = image
        .physical_point_to_continuous_index(&point)
        .expect("Cartesian image with an invertible direction has an index for every point");

    assert!((index[0] - 5.0).abs() < 1e-6);
    assert!((index[1] - 5.0).abs() < 1e-6);
    assert!((index[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_non_zero_origin() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([10.0, 20.0, 30.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let point = Point3::new([15.0, 25.0, 35.0]);
    let index = image
        .physical_point_to_continuous_index(&point)
        .expect("Cartesian image with an invertible direction has an index for every point");

    assert!((index[0] - 5.0).abs() < 1e-6);
    assert!((index[1] - 5.0).abs() < 1e-6);
    assert!((index[2] - 5.0).abs() < 1e-6);
}

use crate::test_support::{metadata_only_image, pseudo_points, rotated_metadata_3d};
use ritk_spatial::CoordinateMap;

type TensorImage<const D: usize> = Image<f32, SequentialBackend, D>;

#[test]
fn physical_index_mapping_obeys_anisotropic_rotated_reference() {
    let image = TensorImage::<3>::from_flat(
        vec![0.0; 6],
        [2, 3, 1],
        Point::new([10.0, 20.0, 0.0]),
        Spacing::new([2.0, 4.0, 1.0]),
        Direction::from_row_major([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    .expect("fixture shape and data length agree");
    let index = Point::new([3.0, -2.0, 0.0]);
    let physical = image.continuous_index_to_physical_point(&index);
    assert_eq!(physical, Point::new([18.0, 26.0, 0.0]));
    assert_eq!(
        image
            .physical_point_to_continuous_index(&physical)
            .expect("rotation matrix is invertible"),
        index
    );
}

#[test]
fn physical_index_mapping_rejects_singular_direction() {
    let image = TensorImage::<3>::from_flat(
        vec![0.0],
        [1, 1, 1],
        Point::origin(),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::from_row_major([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    .expect("fixture shape and data length agree");
    let error = image
        .physical_point_to_continuous_index(&Point::origin())
        .unwrap_err();
    assert_eq!(error.to_string(), "image direction matrix is singular");
}

#[test]
fn physical_point_tensor_matches_scalar_mapping() {
    let image = TensorImage::<3>::from_flat(
        vec![0.0; 8],
        [2, 2, 2],
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([2.0, 4.0, 5.0]),
        Direction::identity(),
    )
    .expect("fixture shape and data length agree");
    let backend = SequentialBackend;
    let points = Tensor::from_slice_on([2, 3], &[12.0_f32, 28.0, 40.0, 8.0, 16.0, 25.0], &backend);
    let indices = image
        .physical_points_to_continuous_indices(&points, &backend)
        .expect("identity direction and point tensor are valid");
    assert_eq!(indices.as_slice(), &[1.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
}

#[test]
fn physical_point_tensor_rejects_wrong_coordinate_width() {
    let image = TensorImage::<3>::from_flat(
        vec![0.0],
        [1, 1, 1],
        Point::origin(),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("fixture shape and data length agree");
    let backend = SequentialBackend;
    let points = Tensor::zeros_on([2, 2], &backend);
    let error = match image.physical_points_to_continuous_indices(&points, &backend) {
        Ok(_) => panic!("wrong coordinate width must be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "physical point tensor shape must be [point_count, 3], got [2, 2]"
    );
}

/// Attaching a Cartesian map must not perturb a single bit of either batch
/// transform: `CoordinateMap::Cartesian` is the default, so this pins that
/// the dispatch introduced no arithmetic change on the existing path.
#[test]
fn cartesian_map_is_bit_identical_to_the_default_path() {
    let make = || {
        metadata_only_image::<f64, SequentialBackend, 3>(
            Point::new([-1.5, 0.25, 3.0]),
            Spacing::new([0.75, 1.25, 2.5]),
            Direction::identity(),
        )
    };
    let plain = make();
    let tagged = make()
        .with_coordinate_map(CoordinateMap::Cartesian)
        .expect("cartesian map is valid at any rank");

    let world = [1.0_f64, 2.0, 3.0, -4.0, 5.5, -6.25];
    let world_t = Tensor::<f64, SequentialBackend>::from_slice([2, 3], &world);
    assert_eq!(
        plain.world_to_index_native(&world_t).as_slice(),
        tagged.world_to_index_native(&world_t).as_slice()
    );

    let idx = [0.5_f64, 1.5, 2.5, -3.5, 4.5, 5.5];
    let idx_t = Tensor::<f64, SequentialBackend>::from_slice([2, 3], &idx);
    assert_eq!(
        plain.index_to_world_native(&idx_t).as_slice(),
        tagged.index_to_world_native(&idx_t).as_slice()
    );
}

/// Analytical oracle — identity geometry. `world_to_index` maps a physical
/// point (axis-major columns) to its index (innermost-first columns), which
/// under identity origin/spacing/direction equals the point with reversed
/// column order; `index_to_world` is the exact inverse. Exact in f64 (×1,
/// +0 only).
#[test]
fn native_batch_identity_reverses_columns_exactly() {
    let img = metadata_only_image::<f64, SequentialBackend, 3>(
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    let world = [1.0_f64, 2.0, 3.0, -4.0, 5.5, -6.25]; // 2 axis-major points
    let world_t = Tensor::<f64, SequentialBackend>::from_slice([2, 3], &world);

    let idx = img.world_to_index_native(&world_t);
    // innermost-first == axis-major reversed per row.
    assert_eq!(idx.as_slice(), &[3.0, 2.0, 1.0, -6.25, 5.5, -4.0]);

    let back = img.index_to_world_native(&idx);
    assert_eq!(back.as_slice(), &world);
}

/// Independent oracle — the batch transforms agree with the single-point
/// `transform_*` methods (mathematically independent code path) under
/// non-trivial anisotropic, rotated geometry. Accounts for the batch
/// innermost-first index column order vs the single-point axis-major
/// `Point`. f64 throughout; tolerance is f64 machine slack.
#[test]
fn native_batch_agrees_with_single_point_methods() {
    let (origin, spacing, direction) = rotated_metadata_3d();
    let img = metadata_only_image::<f64, SequentialBackend, 3>(origin, spacing, direction);

    let pts = pseudo_points(12, 40.0);
    let world_t = Tensor::<f64, SequentialBackend>::from_slice([12, 3], &pts);
    let idx_batch = img.world_to_index_native(&world_t);

    for row in 0..12 {
        let p = Point::<3>::new([pts[row * 3], pts[row * 3 + 1], pts[row * 3 + 2]]);
        // Single-point index (axis-major).
        let idx_axis = img
            .physical_point_to_continuous_index(&p)
            .expect("rotated Cartesian metadata is invertible");
        let batch_row = &idx_batch.as_slice()[row * 3..row * 3 + 3];
        // batch column c ↔ axis D-1-c.
        for c in 0..3 {
            assert!(
                (batch_row[c] - idx_axis[2 - c]).abs() <= 1e-9,
                "row {row} col {c}: batch={}, single={}",
                batch_row[c],
                idx_axis[2 - c]
            );
        }

        // index → world: feed the batch (innermost-first) index back.
        let world_batch = img.index_to_world_native(&Tensor::<f64, SequentialBackend>::from_slice(
            [1, 3],
            batch_row,
        ));
        let idx_pt = Point::<3>::new([idx_axis[0], idx_axis[1], idx_axis[2]]);
        let world_single = img.continuous_index_to_physical_point(&idx_pt);
        for a in 0..3 {
            assert!(
                (world_batch.as_slice()[a] - world_single[a]).abs() <= 1e-9,
                "row {row} axis {a}: batch={}, single={}",
                world_batch.as_slice()[a],
                world_single[a]
            );
        }
    }
}

/// Round-trip: index → world → index recovers the original index within f64
/// eps under non-trivial geometry (composition consistency of the pair).
#[test]
fn native_batch_index_world_roundtrip_identity() {
    let (origin, spacing, direction) = rotated_metadata_3d();
    let img = metadata_only_image::<f64, SequentialBackend, 3>(origin, spacing, direction);

    let idx = pseudo_points(20, 30.0); // innermost-first index rows
    let idx_t = Tensor::<f64, SequentialBackend>::from_slice([20, 3], &idx);

    let world = img.index_to_world_native(&idx_t);
    let world_t = Tensor::<f64, SequentialBackend>::from_slice([20, 3], world.as_slice());
    let idx_rt = img.world_to_index_native(&world_t);

    for (a, b) in idx.iter().zip(idx_rt.as_slice()) {
        assert!((a - b).abs() <= 1e-9, "round-trip drift: {a} vs {b}");
    }
}
