//! Value-semantic tests for [`SignedAxisPermutation`]. Power-of-two spacings
//! and an integer origin keep world points exact, so no tolerance is needed.

use super::SignedAxisPermutation;
use crate::file_axes_direction;
use ritk_spatial::{Direction, Point, Spacing};
use std::borrow::Cow;

/// Distinct extents, spacings and a nonzero origin per tensor axis.
const DIMS: [usize; 3] = [2, 3, 4];
const SPACING: [f64; 3] = [0.5, 2.0, 4.0];
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

/// Every pair of the 48 signed permutations relabels with each voxel at its
/// world point, including extents of 0 and 1, which must not underflow.
#[test]
fn every_pair_of_signed_permutations_keeps_world_points() {
    let shapes = [
        DIMS,
        [0, 3, 4],
        [1, 3, 4],
        [1, 1, 1],
        [2, 0, 4],
        [2, 3, 1],
        [0, 0, 0],
        [2, 1, 4],
    ];
    let directions = signed_permutations();
    for dims in shapes {
        let count = dims.iter().product::<usize>();
        let values: Vec<u32> = (0..u32::try_from(count).expect("small fixture")).collect();
        for from in &directions {
            for to in &directions {
                let permutation = SignedAxisPermutation::between(from, to).expect("signed");
                let grid = permutation.reexpress(
                    &values,
                    dims,
                    &Point::new(ORIGIN),
                    &Spacing::new(SPACING),
                    from,
                );
                assert_eq!(grid.values.len(), count, "{dims:?}");
                let (origin, spacing) = (grid.origin.to_array(), grid.spacing.to_array());
                assert_eq!(
                    world_map(&grid.values, grid.dims, origin, spacing, to),
                    world_map(&values, dims, ORIGIN, SPACING, from),
                    "{dims:?} {from:?} -> {to:?}"
                );
            }
        }
    }
}

/// A buffer whose length disagrees with the extents is a caller defect.
#[test]
#[should_panic(expected = "voxel count must match the grid extents")]
fn a_short_buffer_panics() {
    let target = file_axes_direction();
    let permutation =
        SignedAxisPermutation::between(&Direction::identity(), &target).expect("signed");
    let values = [0u32; 23];
    permutation.reexpress(
        &values,
        DIMS,
        &Point::new(ORIGIN),
        &Spacing::new(SPACING),
        &Direction::identity(),
    );
}

/// Extents whose product overflows `usize` cannot describe any buffer. These
/// wrap to exactly 0, so a wrapping product would accept the empty buffer.
#[test]
#[should_panic(expected = "voxel count must match the grid extents")]
fn overflowing_extents_panic() {
    let target = file_axes_direction();
    let permutation =
        SignedAxisPermutation::between(&Direction::identity(), &target).expect("signed");
    let values: [u32; 0] = [];
    permutation.reexpress(
        &values,
        [1 << (usize::BITS - 1), 2, 1],
        &Point::new(ORIGIN),
        &Spacing::new(SPACING),
        &Direction::identity(),
    );
}

/// A zero extent makes the grid empty wherever it sits, including behind
/// extents whose running product would overflow before reaching it.
#[test]
fn an_empty_grid_is_accepted_in_any_extent_order() {
    let target = file_axes_direction();
    let permutation =
        SignedAxisPermutation::between(&Direction::identity(), &target).expect("signed");
    let values: [u32; 0] = [];
    for dims in [[usize::MAX, 2, 0], [0, usize::MAX, 2], [2, 0, usize::MAX]] {
        let grid = permutation.reexpress(
            &values,
            dims,
            &Point::new(ORIGIN),
            &Spacing::new(SPACING),
            &Direction::identity(),
        );
        assert_eq!(grid.dims, [dims[2], dims[1], dims[0]], "extents {dims:?}");
        assert!(grid.values.is_empty(), "extents {dims:?}");
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

/// An oblique or degenerate direction has no exact relabeling.
#[test]
fn non_permutation_directions_have_no_relabeling() {
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let rotated = Direction::from_rows([[half, -half, 0.0], [half, half, 0.0], [0.0, 0.0, 1.0]]);
    let repeated = Direction::from_rows([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let scaled = Direction::from_rows([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    // One ULP from ±1 on a unit entry, and the smallest magnitudes in a zero slot.
    let up = f64::from_bits(1.0f64.to_bits() + 1);
    let down = f64::from_bits(1.0f64.to_bits() - 1);
    let tiny = f64::from_bits(1);
    let near_misses = [up, down, -up, -down]
        .map(|unit| Direction::from_rows([[unit, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
    let near_zeros = [tiny, -tiny, f64::MIN_POSITIVE, f64::EPSILON]
        .map(|small| Direction::from_rows([[1.0, small, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
    let target = file_axes_direction();
    let degenerate = [rotated, repeated, scaled, Direction::zeros()];
    for direction in degenerate.into_iter().chain(near_misses).chain(near_zeros) {
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

/// `from_axes` accepts exactly the permutations of `[0, 1, 2]`.
#[test]
fn from_axes_accepts_only_axis_permutations() {
    let flips = [true, false, true];
    let built = SignedAxisPermutation::from_axes([2, 0, 1], flips).expect("a permutation");
    assert_eq!((built.axis, built.reversed), ([2, 0, 1], flips));
    for axis in [[0, 0, 1], [1, 2, 2], [0, 1, 3], [usize::MAX, 0, 1]] {
        assert_eq!(
            SignedAxisPermutation::from_axes(axis, flips),
            None,
            "{axis:?}"
        );
    }
}

/// For an oblique input the origin moves exactly as the grid geometry maps
/// the corner voxel, so a caller that snaps to the nearest axes keeps the
/// world point the geometry assigns that voxel, bit for bit.
#[test]
fn an_oblique_corner_moves_the_origin_like_the_grid_geometry() {
    let (sin, cos) = 0.3f64.sin_cos();
    let direction = Direction::from_rows([[cos, -sin, 0.0], [sin, cos, 0.1], [0.0, 0.2, 1.0]]);
    let (origin, spacing) = (Point::new([3.1, -5.3, 7.7]), Spacing::new([0.7, 1.3, 2.9]));
    let permutation =
        SignedAxisPermutation::from_axes([2, 1, 0], [true, false, true]).expect("a permutation");
    let values: Vec<u32> = (0..24).collect();
    let grid = permutation.reexpress(&values, DIMS, &origin, &spacing, &direction);
    let geometry = ritk_spatial::CartesianGridGeometry::new(
        &origin,
        &spacing,
        &direction,
        &ritk_spatial::CoordinateMap::Cartesian,
    )
    .expect("an invertible Cartesian grid");
    let corner = geometry.point([1.0, 0.0, 3.0]);
    assert_eq!(
        grid.origin.to_array().map(f64::to_bits),
        corner.map(f64::to_bits)
    );
}
