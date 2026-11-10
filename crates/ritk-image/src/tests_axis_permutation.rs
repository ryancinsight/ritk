//! Value-semantic tests for [`SignedAxisPermutation`].
//!
//! Every fixture uses power-of-two spacings and integer origins, so world
//! points are exact in `f64` and the oracle needs no tolerance.

use std::borrow::Cow;

use ritk_spatial::{Direction, Point, Spacing};

use super::SignedAxisPermutation;
use crate::file_axes_direction;

/// Distinct extents per tensor axis, so a relabeling changes the shape.
const DIMS: [usize; 3] = [2, 3, 4];
/// Distinct power-of-two spacings per tensor axis.
const SPACING: [f64; 3] = [0.5, 2.0, 4.0];
/// Nonzero integer origin.
const ORIGIN: [f64; 3] = [3.0, -5.0, 7.0];

/// All 48 signed 3-D permutation matrices: 6 column orders × 8 sign masks.
fn signed_permutations() -> Vec<Direction<3>> {
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut directions = Vec::with_capacity(48);
    for order in orders {
        for mask in 0..8u8 {
            let mut rows = [[0.0; 3]; 3];
            for (column, &row) in order.iter().enumerate() {
                rows[row][column] = if mask & (1 << column) == 0 { 1.0 } else { -1.0 };
            }
            directions.push(Direction::from_rows(rows));
        }
    }
    directions
}

/// `origin + D · diag(spacing) · index` for an integer index.
fn world_point(
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: &Direction<3>,
    index: [usize; 3],
) -> [f64; 3] {
    let mut point = origin;
    for axis in 0..3 {
        let step = spacing[axis] * <f64 as eunomia::FloatElement>::from_count(index[axis]);
        for (row, coordinate) in point.iter_mut().enumerate() {
            *coordinate += direction[(row, axis)] * step;
        }
    }
    point
}

/// Every index of a `[z, y, x]` grid in buffer order.
fn indices(dims: [usize; 3]) -> impl Iterator<Item = [usize; 3]> {
    (0..dims[0])
        .flat_map(move |z| (0..dims[1]).flat_map(move |y| (0..dims[2]).map(move |x| [z, y, x])))
}

/// World point → voxel value for a grid, keyed by the exact coordinate bits.
fn world_map(
    values: &[u32],
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: &Direction<3>,
) -> Vec<([u64; 3], u32)> {
    let mut map: Vec<([u64; 3], u32)> = indices(dims)
        .zip(values)
        .map(|(index, &value)| {
            let point = world_point(origin, spacing, direction, index);
            (point.map(f64::to_bits), value)
        })
        .collect();
    map.sort_unstable();
    map
}

/// Each of the 48 signed permutations re-expresses to the file axes with
/// every voxel at its original world point.
#[test]
fn every_signed_permutation_keeps_each_voxel_at_its_world_point() {
    let values: Vec<u32> = (0..24).collect();
    let target = file_axes_direction();
    for direction in signed_permutations() {
        let permutation = SignedAxisPermutation::between(&direction, &target)
            .expect("a signed permutation reaches the file axes");
        let grid = permutation.reexpress(
            &values,
            DIMS,
            &Point::new(ORIGIN),
            &Spacing::new(SPACING),
            &direction,
        );

        assert_eq!(
            world_map(
                &grid.values,
                grid.dims,
                grid.origin.to_array(),
                grid.spacing.to_array(),
                &target
            ),
            world_map(&values, DIMS, ORIGIN, SPACING, &direction),
            "{direction:?}: world point → value map changed"
        );
    }
}

/// The file axes reach themselves without copying the buffer.
#[test]
fn the_file_axes_permutation_borrows_the_buffer() {
    let values: Vec<u32> = (0..24).collect();
    let target = file_axes_direction();
    let permutation =
        SignedAxisPermutation::between(&target, &target).expect("file axes reach themselves");

    assert!(permutation.is_identity());
    let grid = permutation.reexpress(
        &values,
        DIMS,
        &Point::new(ORIGIN),
        &Spacing::new(SPACING),
        &target,
    );
    assert!(matches!(grid.values, Cow::Borrowed(_)));
    assert_eq!(grid.dims, DIMS);
    assert_eq!(grid.origin, Point::new(ORIGIN));
    assert_eq!(grid.spacing, Spacing::new(SPACING));
}

/// An identity-tagged grid swaps its depth and column axes on the way to the
/// file axes: tensor axis 0 advanced world X, so it becomes the column axis.
#[test]
fn an_identity_grid_swaps_depth_and_column_for_the_file_axes() {
    let values: Vec<u32> = (0..24).collect();
    let permutation =
        SignedAxisPermutation::between(&Direction::identity(), &file_axes_direction())
            .expect("identity is a signed permutation");
    let grid = permutation.reexpress(
        &values,
        DIMS,
        &Point::new(ORIGIN),
        &Spacing::new(SPACING),
        &Direction::identity(),
    );

    assert_eq!(grid.dims, [4, 3, 2]);
    assert_eq!(grid.spacing, Spacing::new([4.0, 2.0, 0.5]));
    assert_eq!(grid.origin, Point::new(ORIGIN));
    // Output (z, y, x) reads input (x, y, z): input strides are [12, 4, 1].
    let expected: Vec<u32> = indices([4, 3, 2])
        .map(|[z, y, x]| u32::try_from(x * 12 + y * 4 + z).expect("index fits u32"))
        .collect();
    assert_eq!(grid.values.as_ref(), expected.as_slice());
}

/// A reversed axis moves the origin to the far corner of that axis.
#[test]
fn a_reversed_axis_moves_the_origin_to_its_last_voxel() {
    let values: Vec<u32> = (0..24).collect();
    let target = file_axes_direction();
    // File axes with world Y reversed: tensor axis 1 advances −Y.
    let flipped = Direction::from_rows([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]);
    let permutation =
        SignedAxisPermutation::between(&flipped, &target).expect("signed permutation");
    let grid = permutation.reexpress(
        &values,
        DIMS,
        &Point::new(ORIGIN),
        &Spacing::new(SPACING),
        &flipped,
    );

    assert_eq!(grid.dims, DIMS);
    // Row index 2 of the input sits at y = −5 − 2·2 = −9.
    assert_eq!(grid.origin, Point::new([3.0, -9.0, 7.0]));
    assert_eq!(grid.values[0], 8, "output (0,0,0) is input (0,2,0)");
}

/// An oblique or degenerate direction has no exact relabeling.
#[test]
fn non_permutation_directions_have_no_relabeling() {
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let rotated = Direction::from_rows([[half, -half, 0.0], [half, half, 0.0], [0.0, 0.0, 1.0]]);
    let nearly = Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, f64::EPSILON], [0.0, 0.0, 1.0]]);
    let repeated = Direction::from_rows([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let scaled = Direction::from_rows([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let target = file_axes_direction();
    for direction in [rotated, nearly, repeated, scaled, Direction::zeros()] {
        assert_eq!(
            SignedAxisPermutation::between(&direction, &target),
            None,
            "{direction:?}"
        );
        assert_eq!(
            SignedAxisPermutation::between(&target, &direction),
            None,
            "{direction:?} as target"
        );
    }
}
