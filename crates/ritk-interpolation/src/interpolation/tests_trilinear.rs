//! Atlas-typed tests for the trilinear-interpolation sister adapter.
//!
//! Strict subtractive on test surface (ADR 0012 §Decision §Sub-batch #3.f):
//! every assertion exercises the
//! `crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate`
//! sister API over `Image<f32, MoiraiBackend, 5>` carriers instead of the
//! legacy tensor-backed `super::trilinear_interpolation::<B>(image, grid)`.

use ritk_image::native::Image as AtlasImage;

type B = coeus_core::MoiraiBackend;

/// `[1, 1, 2, 2, 2]` volume; voxel value = z*4 + y*2 + x (row-major, values 0..7).
fn unit_cube() -> AtlasImage<f32, coeus_core::MoiraiBackend, 5> {
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let dims = [1, 1, 2, 2, 2];
    AtlasImage::<f32, coeus_core::MoiraiBackend, 5>::from_flat(
        values,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas unit cube")
}

/// Constant grid `[1, 3, 2, 2, 2]` — every output voxel samples the same (z, y, x).
fn constant_grid(z: f32, y: f32, x: f32) -> AtlasImage<f32, coeus_core::MoiraiBackend, 5> {
    let n = 8usize; // 2 × 2 × 2 spatial
    let mut values = Vec::with_capacity(3 * n);
    values.extend(std::iter::repeat_n(z, n));
    values.extend(std::iter::repeat_n(y, n));
    values.extend(std::iter::repeat_n(x, n));
    let dims = [1, 3, 2, 2, 2];
    AtlasImage::<f32, coeus_core::MoiraiBackend, 5>::from_flat(
        values,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas constant grid")
}

fn assert_all_close(out: &AtlasImage<f32, coeus_core::MoiraiBackend, 5>, expected: f32, eps: f32) {
    let slice = out
        .data_slice()
        .expect("trilinear output atlas carrier host-slice access");
    for (i, &v) in slice.iter().enumerate() {
        assert!(
            (v - expected).abs() < eps,
            "voxel[{i}]: expected {expected}, got {v}"
        );
    }
}

#[test]
fn test_atlas_trilinear_sampling_at_corner_000_returns_value_0() {
    let image = unit_cube();
    let out = crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate::<B>(
        &image,
        &constant_grid(0.0, 0.0, 0.0),
    )
    .expect("atlas trilinear corner(0,0,0)");
    assert_all_close(&out, 0.0, 1e-5);
}

#[test]
fn test_atlas_trilinear_sampling_at_corner_111_returns_value_7() {
    let image = unit_cube();
    let out = crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate::<B>(
        &image,
        &constant_grid(1.0, 1.0, 1.0),
    )
    .expect("atlas trilinear corner(1,1,1)");
    assert_all_close(&out, 7.0, 1e-5);
}

#[test]
fn test_atlas_trilinear_center_sample_returns_arithmetic_mean_3_5() {
    let image = unit_cube();
    let out = crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate::<B>(
        &image,
        &constant_grid(0.5, 0.5, 0.5),
    )
    .expect("atlas trilinear center");
    assert_all_close(&out, 3.5, 1e-5);
}

#[test]
fn test_atlas_trilinear_out_of_bounds_low_clamps_to_corner_000() {
    let image = unit_cube();
    let out = crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate::<B>(
        &image,
        &constant_grid(-1.0, -1.0, -1.0),
    )
    .expect("atlas trilinear oob_low");
    assert_all_close(&out, 0.0, 1e-5);
}

#[test]
fn test_atlas_trilinear_out_of_bounds_high_clamps_to_corner_111() {
    let image = unit_cube();
    let out = crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate::<B>(
        &image,
        &constant_grid(2.5, 2.5, 2.5),
    )
    .expect("atlas trilinear oob_high");
    assert_all_close(&out, 7.0, 1e-5);
}

#[test]
fn test_atlas_trilinear_multichannel_channels_interpolated_independently() {
    let ch0 = [1.0f32; 8];
    let ch1 = [2.0f32; 8];
    let values: Vec<f32> = ch0.into_iter().chain(ch1).collect();
    let dims = [1, 2, 2, 2, 2];
    let image = AtlasImage::<f32, coeus_core::MoiraiBackend, 5>::from_flat(
        values,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas two-channel image");
    let out = crate::interpolation::atlas_trilinear::atlas_trilinear_interpolate::<B>(
        &image,
        &constant_grid(0.5, 0.5, 0.5),
    )
    .expect("atlas trilinear multichannel");
    let slice = out
        .data_slice()
        .expect("atlas multichannel output host-slice access");
    assert_eq!(slice.len(), 16);
    for (i, &v) in slice[0..8].iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-5, "ch0[{i}]: expected 1.0, got {v}");
    }
    for (i, &v) in slice[8..16].iter().enumerate() {
        assert!((v - 2.0).abs() < 1e-5, "ch1[{i}]: expected 2.0, got {v}");
    }
}
