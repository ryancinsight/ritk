//! Tests for binary_dilate
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn flat(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// T1: Radius-0 dilation is identity.
#[test]
fn radius_zero_is_identity() {
    let vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let out = BinaryDilateFilter::new(0).apply(&img).unwrap();
    assert_eq!(flat(&out), vals);
}

/// T2: Single foreground voxel dilates to (2r+1)³ cube.
///
/// 1×1×5 image with fg at centre (index 2), r=1:
/// Expected output: [0, fg, fg, fg, 0] — centre ± 1.
#[test]
fn single_voxel_dilates_to_cube() {
    let img = make_image(vec![0.0, 0.0, 1.0, 0.0, 0.0], [1, 1, 5]);
    let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
    assert_eq!(flat(&out), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
}

/// T3: r=1 on a 1×1×5 image, fg at index 0 — cannot dilate left (border).
/// Expected: [fg, fg, 0, 0, 0].
#[test]
fn border_dilation_bounded_by_image_edge() {
    let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0], [1, 1, 5]);
    let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
    assert_eq!(flat(&out), vec![1.0, 1.0, 0.0, 0.0, 0.0]);
}

/// T4: All-background image → all background after dilation.
#[test]
fn all_background_unchanged() {
    let img = make_image(vec![0.0; 8], [2, 2, 2]);
    let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 0.0));
}

/// T5: All-foreground image → all foreground after dilation.
#[test]
fn all_foreground_unchanged() {
    let img = make_image(vec![1.0; 8], [2, 2, 2]);
    let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 1.0));
}

/// T6: Custom foreground value 255.0.
#[test]
fn custom_foreground_value() {
    let img = make_image(vec![0.0, 0.0, 255.0, 0.0, 0.0], [1, 1, 5]);
    let out = BinaryDilateFilter::new(1)
        .with_foreground(255.0)
        .apply(&img)
        .unwrap();
    assert_eq!(flat(&out), vec![0.0, 255.0, 255.0, 255.0, 0.0]);
}

/// T7: Dilation produces a known analytically correct output.
///
/// Input `f = [0, 0, 1, 0, 0]` in 1×1×5.  With r=1, each voxel is fg if
/// EXISTS an in-bounds X-neighbour that is fg (Z/Y neighbours are OOB at
/// nz=ny=1, but dilation only requires EXISTS — OOB does not contribute fg).
///
/// Voxel (0,0,0): in-bounds X-neighbours = {(0,0,0)=0,(0,0,1)=0} → no fg → bg.
/// Voxel (0,0,1): in-bounds X-neighbour (0,0,2) = 1 → fg.
/// Voxel (0,0,2): self = 1 → fg.
/// Voxel (0,0,3): in-bounds X-neighbour (0,0,2) = 1 → fg.
/// Voxel (0,0,4): in-bounds X-neighbours = {(0,0,3)=0,(0,0,4)=0} → no fg → bg.
///
/// Expected: [0, 1, 1, 1, 0].
#[test]
fn dilation_known_output() {
    let f: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let img = make_image(f, [1, 1, 5]);
    let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
    assert_eq!(flat(&out), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
}

/// Brute-force `(2r+1)³` cubic dilation — the direct definition, used as the
/// differential oracle for the separable implementation.
fn cubic_reference(data: &[f32], dims: [usize; 3], radius: usize, fg: f32) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut out = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut any = false;
                'o: for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (zz, yy, xx) =
                                (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                            if zz < 0
                                || yy < 0
                                || xx < 0
                                || zz >= nz as isize
                                || yy >= ny as isize
                                || xx >= nx as isize
                            {
                                continue;
                            }
                            if data[zz as usize * ny * nx + yy as usize * nx + xx as usize]
                                == fg
                            {
                                any = true;
                                break 'o;
                            }
                        }
                    }
                }
                if any {
                    out[iz * ny * nx + iy * nx + ix] = fg;
                }
            }
        }
    }
    out
}

/// T9: separable 3-D dilation is bitwise-identical to the brute-force cubic
/// definition across radii on a non-trivial asymmetric volume (5×6×7) with
/// scattered foreground seeds, exercising interior, edge, and corner voxels.
#[test]
fn separable_matches_cubic_3d() {
    let dims = [5, 6, 7];
    let n = dims.iter().product::<usize>();
    // Deterministic scattered seeds (no rng dependency).
    let mut data = vec![0.0_f32; n];
    for (i, v) in data.iter_mut().enumerate() {
        if i % 11 == 0 || i % 17 == 3 {
            *v = 1.0;
        }
    }
    for r in 0..=3 {
        let got = dilate_binary_3d(&data, dims, r, ForegroundValue::ONE);
        let want = cubic_reference(&data, dims, r, 1.0);
        assert_eq!(got, want, "separable != cubic at radius {r}");
    }
}

/// T8: Spatial metadata is preserved unchanged.
#[test]
fn spatial_metadata_preserved() {
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 1.0]);
    let direction = Direction::identity();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let img = Image::new(t, origin, spacing, direction);
    let out = BinaryDilateFilter::new(0).apply(&img).unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}
