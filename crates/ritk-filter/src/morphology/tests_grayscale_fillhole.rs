//! Tests for grayscale_fillhole
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data().to_vec()
}

fn flat(iz: usize, iy: usize, ix: usize, ny: usize, nx: usize) -> usize {
    iz * ny * nx + iy * nx + ix
}

/// Constant image unchanged: no holes in a uniform field.
///
/// **Proof**: every voxel has minimax path level = c. âˆŽ
#[test]
fn constant_image_unchanged() {
    let c = 7.0_f32;
    let dims = [5, 5, 5];
    let img = make_image(vec![c; 125], dims);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    for &v in extract_vals(&out).iter() {
        assert!((v - c).abs() < 1e-6, "constant unchanged: got {v}");
    }
}

/// Output satisfies h[x] â‰¥ I[x] for all x (holes only raised, not lowered).
#[test]
fn output_ge_input_everywhere() {
    let dims = [6, 6, 6];
    let n = 216;
    let vals: Vec<f32> = (0..n as u32).map(|i| (i * 7919 % 128) as f32).collect();
    let img = make_image(vals.clone(), dims);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    let out_vals = extract_vals(&out);
    for (i, (&before, &after)) in vals.iter().zip(out_vals.iter()).enumerate() {
        assert!(
            after >= before - 1e-5,
            "output must be â‰¥ input at voxel {i}: before={before} after={after}"
        );
    }
}

/// Border voxels are never modified.
///
/// **Proof**: border voxels are seeded with I[b]; they can only be updated
/// by paths from other borders, which can never produce a lower level. âˆŽ
#[test]
fn border_voxels_unchanged() {
    let [nz, ny, nx] = [5usize, 5, 5];
    let n = nz * ny * nx;
    let mut vals: Vec<f32> = (0..n as u32).map(|i| (i * 2017 % 64) as f32).collect();
    // Put a dark pit in the interior at (2,2,2)
    vals[flat(2, 2, 2, ny, nx)] = 0.0;
    let img = make_image(vals.clone(), [nz, ny, nx]);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    let out_vals = extract_vals(&out);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let on_border =
                    iz == 0 || iz == nz - 1 || iy == 0 || iy == ny - 1 || ix == 0 || ix == nx - 1;
                if on_border {
                    let f = flat(iz, iy, ix, ny, nx);
                    assert!(
                        (out_vals[f] - vals[f]).abs() < 1e-6,
                        "border voxel [{iz},{iy},{ix}] changed"
                    );
                }
            }
        }
    }
}

/// Enclosed dark pit filled to surrounding wall level.
///
/// Volume: 3Ã—3Ã—3 (27 voxels). Only interior voxel is flat[13] at (1,1,1).
/// I = 5 everywhere on border; I[1,1,1] = 0 (dark pit).
///
/// Expected: h[1,1,1] = 5.0 (raised to border level).
///
/// **Proof**: minimax path from (1,1,1) to any border passes through exactly
/// one border voxel at level 5. max along path = max(0, 5) = 5. âˆŽ
#[test]
fn enclosed_pit_filled_to_border_level() {
    let dims = [3usize, 3, 3];
    let n = 27;
    let mut vals = vec![5.0_f32; n];
    vals[flat(1, 1, 1, 3, 3)] = 0.0;
    let img = make_image(vals, dims);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    let out_vals = extract_vals(&out);
    let pit_out = out_vals[flat(1, 1, 1, 3, 3)];
    assert!(
        (pit_out - 5.0).abs() < 1e-5,
        "pit filled to 5.0; got {pit_out}"
    );
}

/// Enclosed pit filled to WALL level (not border level) when wall < border.
///
/// Volume: 5Ã—5Ã—5. Outer shell (border) = 1.0. Inner shell at iz/iy/ix âˆˆ {1..3}
/// = 8.0. Innermost voxel (2,2,2) = 2.0.
///
/// The minimum-barrier path from (2,2,2) to the border must pass through
/// the inner shell at level 8. Therefore h[2,2,2] = 8.0.
///
/// **Proof**: any path from (2,2,2) to the border with |dx|+|dy|+|dz|=1 steps
/// must pass through a voxel in the inner shell with value 8. The minimax
/// path level is therefore min(8) = 8, not the border level 1. âˆŽ
#[test]
fn pit_filled_to_wall_level_not_border_level() {
    let [nz, ny, nx] = [5usize, 5, 5];
    let n = nz * ny * nx;
    let mut vals = vec![0.0_f32; n]; // all outer = 0 (overwritten below)
                                     // Outer shell: value 1
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let on_border =
                    iz == 0 || iz == nz - 1 || iy == 0 || iy == ny - 1 || ix == 0 || ix == nx - 1;
                if on_border {
                    vals[flat(iz, iy, ix, ny, nx)] = 1.0;
                }
            }
        }
    }
    // Inner shell iz/iy/ix âˆˆ {1..3}: value 8
    for iz in 1..4 {
        for iy in 1..4 {
            for ix in 1..4 {
                vals[flat(iz, iy, ix, ny, nx)] = 8.0;
            }
        }
    }
    // Innermost pit at (2,2,2): value 2
    vals[flat(2, 2, 2, ny, nx)] = 2.0;

    let img = make_image(vals, [nz, ny, nx]);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    let out_vals = extract_vals(&out);
    let pit_out = out_vals[flat(2, 2, 2, ny, nx)];
    assert!(
        (pit_out - 8.0).abs() < 1e-5,
        "pit filled to wall level 8.0; got {pit_out}"
    );
}

/// Border-connected dark region NOT filled.
///
/// Volume: 3Ã—3Ã—3 with value 0 everywhere. All voxels are on the border
/// or connect to it through 0-valued paths. Fill should not increase anything.
#[test]
fn border_connected_dark_not_filled() {
    let dims = [3usize, 3, 3];
    let img = make_image(vec![0.0_f32; 27], dims);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    for &v in extract_vals(&out).iter() {
        assert!(v.abs() < 1e-6, "border-connected dark must stay 0, got {v}");
    }
}

/// Spatial metadata (origin, spacing, direction) is preserved.
#[test]
fn spatial_metadata_preserved() {
    let origin = Point::new([2.0, 3.0, 4.0]);
    let spacing = Spacing::new([0.75, 0.75, 1.5]);
    let direction = Direction::identity();
    let tensor = Tensor::<f32, B>::from_slice([3, 3, 3], &[1.0_f32; 27]);
    let img = Image::new(tensor, origin, spacing, direction)
        .expect("invariant: fixture tensor has the declared rank");
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
}

/// A volume in which every voxel lies on the border has no interior to fill, so
/// fill-hole is the identity. A 2Ã—2Ã—2 volume is all corners (every voxel is an
/// extremum of all three axes), so output = input exactly.
#[test]
fn all_border_volume_unchanged() {
    let vals = vec![5.0_f32, 1.0, 2.0, 8.0, 3.0, 6.0, 4.0, 7.0];
    let dims = [2, 2, 2];
    let img = make_image(vals.clone(), dims);
    let out = GrayscaleFillholeFilter::new().apply(&img).unwrap();
    let out_vals = extract_vals(&out);
    for (i, (&a, &b)) in vals.iter().zip(out_vals.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "all-border voxel {i}: {a} â‰  {b}");
    }
}

/// A `z = 1` slab is a genuine 2-D image, not an all-border volume: its interior
/// dark pits must be filled (regression for the degenerate-axis border bug,
/// where `iz == 0` wrongly flagged every voxel as border, making fill-hole a
/// no-op on 2-D images). The 1-D signal `[5,5,1,5,1,5,5]` (as `1Ã—1Ã—7`) has its
/// interior `1`-pits raised to the connecting wall level `5`.
#[test]
fn degenerate_axis_interior_pits_are_filled() {
    let vals = vec![5.0_f32, 5.0, 1.0, 5.0, 1.0, 5.0, 5.0];
    let out = extract_vals(
        &GrayscaleFillholeFilter::new()
            .apply(&make_image(vals, [1, 1, 7]))
            .unwrap(),
    );
    let expected = [5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    for (i, (&got, exp)) in out.iter().zip(expected).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "fill-hole 1-D pit {i}: got {got}, expected {exp}"
        );
    }
}
