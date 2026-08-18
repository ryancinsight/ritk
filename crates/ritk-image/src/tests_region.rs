//! Region-view contracts: what a borrow sees, and what it costs.
//!
//! The oracle for every value assertion here is the host row-major layout of
//! the fixture, computed by hand in the test rather than by re-running the
//! implementation's own index arithmetic.

use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};

use crate::region::RowWalker;
use crate::test_support::{make_image, make_image_with};
use crate::types::Image;

type B = SequentialBackend;

/// 2x3 image, values 1..=6 row-major:
///   [ 1 2 3 ]
///   [ 4 5 6 ]
fn image_2x3() -> Image<f32, B, 2> {
    make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
}

/// 2x3x4 image, values 0..=23 row-major.
fn image_2x3x4() -> Image<f32, B, 3> {
    make_image((0..24).map(|v| v as f32).collect(), [2, 3, 4])
}

#[test]
fn full_region_borrows_whole_image_in_row_major_order() {
    let image = image_2x3();
    let region = image.region().expect("contiguous image yields a region");

    assert_eq!(region.shape(), [2, 3]);
    assert_eq!(region.strides(), [3, 1]);
    assert_eq!(region.len(), 6);
    assert!(!region.is_empty());
    assert!(region.is_contiguous());

    let values: Vec<f32> = region.iter().copied().collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // A contiguous region hands back the source slice itself, not a copy.
    let slice = region
        .as_slice()
        .expect("contiguous region exposes a slice");
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(
        std::ptr::eq(slice.as_ptr(), image.data_slice().unwrap().as_ptr()),
        "region slice must alias the image buffer, not copy it"
    );
}

#[test]
fn subregion_yields_only_its_own_voxels() {
    let image = image_2x3();
    let region = image.region().unwrap();

    // Columns 1..3 of both rows: [[2,3],[5,6]].
    let sub = region
        .subregion([(0, 2), (1, 3)])
        .expect("in-bounds bounds");

    assert_eq!(sub.shape(), [2, 2]);
    // Strides are inherited: the sub-region is strided, not compacted.
    assert_eq!(sub.strides(), [3, 1]);
    assert!(!sub.is_contiguous());
    assert!(
        sub.as_slice().is_none(),
        "strided region must refuse a flat slice rather than materialise one"
    );

    let values: Vec<f32> = sub.iter().copied().collect();
    assert_eq!(values, vec![2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn subregion_indexed_access_matches_the_parent() {
    let image = image_2x3x4();
    let region = image.region().unwrap();
    let sub = region.subregion([(1, 2), (1, 3), (2, 4)]).unwrap();

    assert_eq!(sub.shape(), [1, 2, 2]);
    // Parent index [1+z, 1+y, 2+x] flattens to ((1+z)*3 + (1+y))*4 + (2+x).
    for z in 0..1 {
        for y in 0..2 {
            for x in 0..2 {
                let expected = (((1 + z) * 3 + (1 + y)) * 4 + (2 + x)) as f32;
                assert_eq!(*sub.get([z, y, x]).expect("in bounds"), expected);
            }
        }
    }
    assert!(sub.get([1, 0, 0]).is_none(), "out-of-bounds must be None");
}

#[test]
fn subregion_rejects_out_of_range_bounds() {
    let image = image_2x3();
    let region = image.region().unwrap();

    let err = region.subregion([(0, 2), (1, 4)]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "region bounds [1..4) exceed extent 3 on axis 1"
    );

    let inverted = region.subregion([(2, 1), (0, 3)]).unwrap_err();
    assert_eq!(
        inverted.to_string(),
        "region bounds [2..1) exceed extent 2 on axis 0"
    );
}

#[test]
fn subregion_shifts_origin_by_direction_times_spacing() {
    // Non-identity direction and anisotropic spacing, so a forgotten direction
    // term or a swapped axis order changes the expected numbers.
    // Row-major shape axis d maps to spatial axis D-1-d.
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let image: Image<f32, B, 3> = make_image_with(
        (0..24).map(|v| v as f32).collect(),
        [2, 3, 4],
        Some(Point::new([10.0, 20.0, 30.0])),
        Some(Spacing::new([0.5, 1.25, 2.0])),
        Some(direction),
    );

    let region = image.region().unwrap();
    // Start at row-major [1, 2, 3] => spatial start (innermost-first) [3, 2, 1].
    let sub = region.subregion([(1, 2), (2, 3), (3, 4)]).unwrap();

    // origin'[row] = origin[row] + Σ_axis direction[(row,axis)] * spacing[axis] * start[axis]
    // start = [3, 2, 1], spacing = [0.5, 1.25, 2.0]
    //   scaled = [1.5, 2.5, 2.0]
    //   row 0: 10 + (0*1.5 + -1*2.5 + 0*2.0) =  7.5
    //   row 1: 20 + (1*1.5 +  0*2.5 + 0*2.0) = 21.5
    //   row 2: 30 + (0*1.5 +  0*2.5 + 1*2.0) = 32.0
    let origin = sub.origin();
    assert!((origin[0] - 7.5).abs() < 1e-12, "got {}", origin[0]);
    assert!((origin[1] - 21.5).abs() < 1e-12, "got {}", origin[1]);
    assert!((origin[2] - 32.0).abs() < 1e-12, "got {}", origin[2]);

    // Spacing and direction are inherited unchanged.
    assert_eq!(sub.spacing().to_array(), [0.5, 1.25, 2.0]);
    assert_eq!(*sub.direction(), direction);
}

#[test]
fn subregion_origin_agrees_with_the_canonical_forward_transform() {
    // Cross-check against the crate's own index->physical map: a region's
    // origin must equal the physical point of its first voxel. Two independent
    // routes to one number.
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let image: Image<f32, B, 3> = make_image_with(
        (0..24).map(|v| v as f32).collect(),
        [2, 3, 4],
        Some(Point::new([10.5, -3.25, 7.0])),
        Some(Spacing::new([0.5, 1.25, 2.0])),
        Some(direction),
    );

    let region = image.region().unwrap();
    let sub = region.subregion([(1, 2), (2, 3), (3, 4)]).unwrap();

    // The canonical transform takes an innermost-first continuous index.
    let expected = image.continuous_index_to_physical_point(&Point::new([3.0, 2.0, 1.0]));
    for row in 0..3 {
        assert!(
            (sub.origin()[row] - expected[row]).abs() < 1e-12,
            "axis {row}: region origin {} vs transform {}",
            sub.origin()[row],
            expected[row]
        );
    }
}

#[test]
fn clipped_window_shrinks_at_the_boundary() {
    let image = image_2x3();
    let region = image.region().unwrap();

    // Centre at the corner [0,0] with radius 1: clipped to [[1,2],[4,5]].
    let corner = region.clipped_window([0, 0], [1, 1]).unwrap();
    assert_eq!(corner.shape(), [2, 2]);
    assert_eq!(
        corner.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 4.0, 5.0]
    );

    // Centre at [1,1] with radius 1: the whole image, since 2x3 fits inside.
    let middle = region.clipped_window([1, 1], [1, 1]).unwrap();
    assert_eq!(middle.shape(), [2, 3]);
    assert_eq!(middle.len(), 6);

    // Radius zero is the single voxel.
    let single = region.clipped_window([1, 2], [0, 0]).unwrap();
    assert_eq!(single.shape(), [1, 1]);
    assert_eq!(single.iter().copied().collect::<Vec<_>>(), vec![6.0]);

    let err = region.clipped_window([2, 0], [1, 1]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "window centre 2 exceeds extent 2 on axis 0"
    );
}

#[test]
fn rows_lend_direct_borrows_when_the_inner_axis_is_unit_stride() {
    let image = image_2x3();
    let region = image.region().unwrap();
    let sub = region.subregion([(0, 2), (1, 3)]).unwrap();

    let mut walker = sub.rows();
    assert!(
        walker.is_zero_copy(),
        "unit inner stride must lend source borrows"
    );
    assert_eq!(walker.remaining(), 2);

    let source = image.data_slice().unwrap();
    let mut seen = Vec::new();
    while let Some(row) = walker.next_row() {
        // A lent row must alias the image buffer, not a gathered copy.
        let offset =
            (row.as_ptr() as usize - source.as_ptr() as usize) / std::mem::size_of::<f32>();
        assert!(
            offset < source.len(),
            "row must point into the source buffer"
        );
        seen.push(row.to_vec());
    }
    assert_eq!(seen, vec![vec![2.0, 3.0], vec![5.0, 6.0]]);
}

/// A permuted (transposed) 2x3 image: shape [3, 2], strides [1, 3].
fn permuted_image_3x2() -> Image<f32, B, 2> {
    use coeus_tensor::Tensor;

    let base = Tensor::<f32, B>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Image::<f32, B, 2>::new(
        base.permute(&[1, 0]),
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
    .unwrap()
}

#[test]
fn rows_gather_into_reused_scratch_when_the_inner_axis_is_strided() {
    let image = permuted_image_3x2();
    let region = image
        .region()
        .expect("a strided image still yields a region");

    assert_eq!(region.shape(), [3, 2]);
    assert_eq!(region.strides(), [1, 3]);
    assert!(!region.is_contiguous());

    let mut walker = region.rows();
    assert!(
        !walker.is_zero_copy(),
        "non-unit inner stride must gather into scratch"
    );

    let mut seen = Vec::new();
    while let Some(row) = walker.next_row() {
        seen.push(row.to_vec());
    }
    // Host transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]].
    assert_eq!(seen, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
}

#[test]
fn subregions_tile_the_region_and_drop_the_ragged_tail() {
    let image = image_2x3x4();
    let region = image.region().unwrap();

    // 2x3x4 tiled by 1x2x2 => counts 2 x 1 x 2 = 4 tiles; the y tail (1 of 3)
    // is dropped.
    let tiles: Vec<_> = region.subregions([1, 2, 2]).unwrap().collect();
    assert_eq!(tiles.len(), 4);
    for tile in &tiles {
        assert_eq!(tile.shape(), [1, 2, 2]);
        assert_eq!(tile.len(), 4);
    }

    // First tile is [0,0..2,0..2] => flat 0,1, 4,5.
    assert_eq!(
        tiles[0].iter().copied().collect::<Vec<_>>(),
        vec![0.0, 1.0, 4.0, 5.0]
    );
    // Last tile is z=1, y=0..2, x=2..4 => flat 12+2,12+3, 16+2,16+3.
    assert_eq!(
        tiles[3].iter().copied().collect::<Vec<_>>(),
        vec![14.0, 15.0, 18.0, 19.0]
    );

    let err = region.subregions([1, 0, 2]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "tile extent must be non-zero on every axis, got 0 on axis 1"
    );
}

#[test]
fn iter_reports_an_exact_size_and_fuses() {
    let image = image_2x3();
    let region = image.region().unwrap();
    let sub = region.subregion([(0, 2), (1, 3)]).unwrap();

    let mut iter = sub.iter();
    assert_eq!(iter.len(), 4);
    assert_eq!(iter.size_hint(), (4, Some(4)));
    iter.next();
    assert_eq!(iter.len(), 3);
    let _: Vec<_> = iter.by_ref().collect();
    assert!(iter.next().is_none());
    assert!(iter.next().is_none(), "iterator must stay fused");
}

#[test]
fn empty_region_yields_nothing() {
    let image = image_2x3();
    let region = image.region().unwrap();
    let empty = region.subregion([(1, 1), (0, 3)]).unwrap();

    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.iter().count(), 0);
    let mut walker = empty.rows();
    assert!(walker.next_row().is_none());
}

#[test]
fn strided_image_is_readable_in_place_rather_than_materialised() {
    // The capability the whole module exists for: `data_slice` refuses this
    // image and `data_cow` copies the whole volume; a region reads it in place.
    let image = permuted_image_3x2();

    assert!(
        image.data_slice().is_err(),
        "the flat-slice accessor still refuses a strided image"
    );
    assert!(
        matches!(image.data_cow_on(&B::default()), std::borrow::Cow::Owned(_)),
        "the cow accessor still materialises a strided image"
    );

    let region = image.region().unwrap();
    assert!(
        region.as_slice().is_none(),
        "a strided region must not pretend to be flat"
    );
    // Logical row-major order of the transposed view.
    assert_eq!(
        region.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}
