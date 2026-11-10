use super::OrientImageFilter;
use ritk_core::rejection::assert_rejects;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, NonCartesianGrid, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// A `[2, 3, 4]` image of distinct voxels with the given direction.
fn with_direction(direction: Direction<3>) -> Image<f32, B, 3> {
    Image::from_flat_on(
        (0..24u8).map(f32::from).collect(),
        [2, 3, 4],
        Point::new([3.0, -5.0, 7.0]),
        Spacing::new([0.5, 2.0, 4.0]),
        direction,
        &B::default(),
    )
    .expect("fixture image")
}

/// Orient `image` through both application paths, which must agree on the
/// output's world map and direction, or on the refusal.
fn orient(code: &str, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
    let filter = OrientImageFilter::from_code(code).expect("a valid code");
    let legacy = filter.apply(image);
    let native = filter.apply_native(image, &B::default());
    match (&legacy, &native) {
        (Ok(legacy), Ok(native)) => {
            assert_eq!(legacy.shape(), native.shape(), "{code}");
            assert_eq!(legacy.direction(), native.direction(), "{code}");
            assert_eq!(world_map(legacy), world_map(native), "{code}");
        }
        (Err(legacy), Err(native)) => {
            assert_eq!(legacy.to_string(), native.to_string(), "{code}");
        }
        _ => panic!("{code}: one path refused and the other did not"),
    }
    native
}

/// With an identity direction, the image's own orientation code is "SPL"
/// (image axes x, y, z = tensor axes 2, 1, 0 point world z+, y+, x+). Orienting
/// to the self-code is a no-op.
#[test]
fn orient_to_self_code_is_identity() {
    let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let img = make(data.clone(), [1, 2, 3]);
    let out = OrientImageFilter::from_code("SPL")
        .expect("infallible: validated precondition")
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), [1, 2, 3]);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, data, "self-code orientation must not change data");
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
}

/// "IAR" reverses each image axis from "SPL" (S→I, P→A, L→R), a full reversal of
/// all three axes with no permutation: out[z][y][x] = in[z][Y-1-y][X-1-x].
#[test]
fn orient_full_reversal() {
    let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let img = make(data, [1, 2, 3]); // y0=[0,1,2], y1=[3,4,5]
    let out = OrientImageFilter::from_code("IAR")
        .expect("infallible: validated precondition")
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), [1, 2, 3]);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
}

/// A permuting code reorders the axes: it preserves the voxel multiset and
/// permutes the shape, leaving the data a rearrangement of the input.
#[test]
fn orient_permutation_preserves_multiset() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let img = make(data.clone(), [2, 3, 4]);
    // "LPS" maps image axes to world x+, y+, z+; for identity direction that
    // permutes the tensor axes (z↔x) relative to the "SPL" self-code.
    let out = OrientImageFilter::from_code("LPS")
        .expect("infallible: validated precondition")
        .apply(&img)
        .expect("infallible: validated precondition");
    let (mut ov, _) = extract_vec_infallible(&out);
    let mut sorted = data;
    ov.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("infallible: validated precondition")
    });
    sorted.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("infallible: validated precondition")
    });
    assert_eq!(ov, sorted, "orientation must preserve the voxel multiset");
    assert_eq!(out.shape(), [4, 3, 2], "tensor axes z and x swap");
}

/// Malformed codes are rejected, not silently accepted.
#[test]
fn orient_invalid_codes_error() {
    let img = make(vec![0.0; 6], [1, 2, 3]);
    let f = OrientImageFilter::from_code("XYZ");
    assert_rejects(f, "invalid orientation letter 'X' in");
    assert!(
        OrientImageFilter::from_code("LL").is_err(),
        "wrong length must error"
    );
    // Repeated anatomical axis (two letters on the L/R axis).
    let dup = OrientImageFilter::from_code("LRP");
    assert_rejects(dup, "repeats an anatomical axis");
    // A valid filter still applies cleanly.
    assert!(OrientImageFilter::from_code("SPL")
        .expect("infallible: validated precondition")
        .apply(&img)
        .is_ok());
}

/// World point bits → voxel value, sorted, for every voxel of `image`.
fn world_map(image: &Image<f32, B, 3>) -> Vec<([u64; 3], u32)> {
    let [nz, ny, nx] = image.shape();
    let (values, _) = extract_vec_infallible(image);
    let mut map = Vec::with_capacity(values.len());
    let mut flat = 0;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = [z, y, x].map(|i| {
                    f64::from(u32::try_from(i).expect("invariant: fixture index fits u32"))
                });
                let point = image.continuous_index_to_physical_point(&Point::new(index));
                map.push((point.to_array().map(f64::to_bits), values[flat].to_bits()));
                flat += 1;
            }
        }
    }
    map.sort_unstable();
    map
}

/// Every code keeps each voxel of an axis-aligned image at its world point
/// and sets the output direction to the code's axes.
///
/// Power-of-two spacings and an integer origin keep world points exact.
#[test]
fn axis_aligned_inputs_keep_world_points_under_every_code() {
    let inputs = [
        Direction::identity(),
        Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        Direction::from_rows([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
        Direction::from_rows([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
    ];
    // Code letter → world column of the output direction, image axis x first.
    let codes = [
        ("LPS", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        (
            "RAI",
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        ),
        ("SPL", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ("PIR", [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
    ];
    for direction in inputs {
        let image = with_direction(direction);
        for (code, image_axes) in codes {
            let out = orient(code, &image).expect("an axis-aligned input orients");
            assert_eq!(
                world_map(&out),
                world_map(&image),
                "{code} from {direction:?}"
            );
            // Image axis j is tensor axis 2 − j, so its column is reversed.
            for (j, column) in image_axes.iter().enumerate() {
                for (row, &entry) in column.iter().enumerate() {
                    assert_eq!(out.direction()[(row, 2 - j)], entry, "{code} axis {j}");
                }
            }
        }
    }
}

/// Only a Cartesian grid with an invertible direction has a corner voxel at
/// an affine world point; both paths refuse any other grid under every code.
#[test]
fn grids_without_an_affine_corner_are_refused() {
    let singular = with_direction(Direction::from_rows([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]));
    for code in ["LPS", "RAI", "SPL", "PIR"] {
        let Err(beam) = orient(code, &ts::phased_array_image()) else {
            panic!("{code}: a beam-space image must not orient");
        };
        assert!(
            matches!(
                beam.downcast_ref(),
                Some(NonCartesianGrid::CoordinateMap(_))
            ),
            "{code}: {beam:#}"
        );
        let Err(flat) = orient(code, &singular) else {
            panic!("{code}: a singular direction must not orient");
        };
        assert!(
            matches!(
                flat.downcast_ref(),
                Some(NonCartesianGrid::SingularDirection)
            ),
            "{code}: {flat:#}"
        );
    }
}

/// A 45-degree in-plane rotation is equally close to two world axes, so two
/// output axes snap to one input axis and no relabeling exists.
#[test]
fn a_direction_whose_axes_collide_is_refused() {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let image = with_direction(Direction::from_rows([
        [h, -h, 0.0],
        [h, h, 0.0],
        [0.0, 0.0, 1.0],
    ]));
    let Err(error) = orient("LPS", &image) else {
        panic!("colliding axes must not orient");
    };
    assert_eq!(
        error.to_string(),
        "input direction is not axis-aligned; cannot DICOM-orient"
    );
}
