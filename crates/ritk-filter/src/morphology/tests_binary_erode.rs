//! Atlas-typed tests for binary erosion (sub-batch #3.a, ADR 0012).
//!
//! # Atomic-boundary invariant
//!
//! This file is the sub-batch #3.a *proof-of-pattern* port:
//!
//! - **Strictly subtractive on the test surface**: the Burn-keyed
//!   `burn_ndarray::NdArray<B>`, the Burn-keyed `ritk_image::Image<B, 3>`
//!   re-export, and the Burn-only `ritk_image::test_support` helper
//!   imports are dropped from this test module. This **drops one
//!   source-row from `xtask/burn_surface.allowlist`** for
//!   `repos/ritk/crates/ritk-filter/src/morphology/tests_binary_erode.rs`.
//!
//! - **Strictly additive on the production surface**: the companion
//!   `atlas_binary_erode.rs` introduces `AtlasBinaryErodeFilter` that
//!   consumes `AtlasImage<f32, MoiraiBackend, 3>` over
//!   `coeus_core::ComputeBackend`. Zero changes to
//!   `BinaryErodeFilter::apply<B: Backend>(&Image<B, 3>)` (legacy
//!   public API preserved per ADR 0012 §Decision §2).
//!
//! - **Cargo.toml**: only additive (`coeus-tensor = { workspace = true }`
//!   appended to `ritk-filter/Cargo.toml`); no deletion or rename of
//!   existing deps.
//!
//! The test bodies (T1 through T7) preserve the original geometry
//! oracle and assertion values verbatim — the Burn-keyed → Atlas-typed
//! port is a mechanical type-as-parameter substitution, not a
//! semantic change.
// `tests_binary_erode` is a submodule of `binary_erode` (via `#[path = "..."] mod`).
// Pull every Atlas-side dependency explicitly: `BinaryErodeFilter` is no longer
// referenced in the rewritten test bodies, so the legacy `use super::*` is gone
// (the previous iteration's `#[warn(unused_imports)]` warning was real, not stale).
use crate::morphology::AtlasBinaryErodeFilter;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use ritk_image::AtlasImage;
use ritk_spatial::{Direction, Point, Spacing};

type B = MoiraiBackend;

/// Atlas-side equivalent of the legacy `ritk_image::test_support::make_image::<B, 3>`.
///
/// Constructs an `AtlasImage<f32, MoiraiBackend, 3>` with default
/// spatial metadata (zero origin, unit spacing, identity direction).
fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> AtlasImage<f32, B, 3> {
    let origin = Point::new([0.0_f64; 3]);
    let spacing = Spacing::new([1.0_f64; 3]);
    let direction = Direction::identity();
    AtlasImage::<f32, MoiraiBackend, 3>::from_flat(vals, dims, origin, spacing, direction)
        .expect("Atlas test image construction with valid flat data must succeed")
}

/// Atlas-side host data extraction. Mirrors the legacy `flat(&Image<B, 3>)`
/// helper that cloned the underlying `burn::tensor::TensorData` and ran
/// `as_slice::<f32>().unwrap().to_vec()` — Atlas-side uses the
/// [`AtlasImage::data_vec`] layout-independent extractor, which never
/// fails on the contiguous `from_flat`-constructed images used here.
fn flat(img: &AtlasImage<f32, B, 3>) -> Vec<f32> {
    img.data_vec()
}

/// T1: Radius-0 erosion is identity (single-voxel SE).
#[test]
fn radius_zero_is_identity() {
    let vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let out = AtlasBinaryErodeFilter::new(0).apply(&img).unwrap();
    assert_eq!(flat(&out), vals);
}

/// T2: All-foreground 3×3×3 image with r=1 → only centre voxel (1,1,1) survives.
#[test]
fn border_voxels_eroded_to_background() {
    let img = make_image(vec![1.0; 27], [3, 3, 3]);
    let out = AtlasBinaryErodeFilter::new(1).apply(&img).unwrap();
    let result = flat(&out);
    assert_eq!(result[13], 1.0, "centre voxel must survive erosion");
    for (i, &v) in result.iter().enumerate() {
        if i != 13 {
            assert_eq!(v, 0.0, "border/edge voxel {i} must be eroded");
        }
    }
}

/// T3: Background pixel surrounded by foreground is NOT changed to foreground.
#[test]
fn background_remains_background() {
    let img = make_image(vec![1.0, 0.0, 1.0], [1, 1, 3]);
    let out = AtlasBinaryErodeFilter::new(1).apply(&img).unwrap();
    assert_eq!(flat(&out), vec![0.0, 0.0, 0.0]);
}

/// T4: 3×3×5 all-foreground, r=1 → strips one border layer from all 6 faces.
#[test]
fn erosion_strips_one_border_layer_r1() {
    let img = make_image(vec![1.0; 45], [3, 3, 5]);
    let out = AtlasBinaryErodeFilter::new(1).apply(&img).unwrap();
    let result = flat(&out);
    let mut expected = vec![0.0_f32; 45];
    expected[21] = 1.0;
    expected[22] = 1.0;
    expected[23] = 1.0;
    assert_eq!(result, expected);
}

/// T5: 5×5×7 all-foreground, r=2 → strips two border layers from all faces.
#[test]
fn erosion_strips_two_border_layers_r2() {
    let img = make_image(vec![1.0; 175], [5, 5, 7]);
    let out = AtlasBinaryErodeFilter::new(2).apply(&img).unwrap();
    let result = flat(&out);
    let mut expected = vec![0.0_f32; 175];
    expected[86] = 1.0;
    expected[87] = 1.0;
    expected[88] = 1.0;
    assert_eq!(result, expected);
}

/// T6: Custom foreground value 255.0 — 3×3×5 volume.
#[test]
fn custom_foreground_value() {
    let img = make_image(vec![255.0; 45], [3, 3, 5]);
    let out = AtlasBinaryErodeFilter::new(1)
        .with_foreground(255.0)
        .apply(&img)
        .unwrap();
    let result = flat(&out);
    let mut expected = vec![0.0_f32; 45];
    expected[21] = 255.0;
    expected[22] = 255.0;
    expected[23] = 255.0;
    assert_eq!(result, expected);
}

/// T7: Spatial metadata is preserved unchanged.
///
/// Mirrors the legacy T7 verbatim, but routes through the
/// [`AtlasImage::new`] constructor with an Atlas-side
/// [`coeus_tensor::Tensor::from_slice_on`] call (no `burn_ndarray::NdArrayDevice`,
/// no `burn::tensor::TensorData::new`, no `burn::tensor::Shape::new`).
#[test]
fn spatial_metadata_preserved() {
    let origin = Point::new([3.0, 2.0, 1.0]);
    let spacing = Spacing::new([0.5, 0.5, 1.0]);
    let direction = Direction::identity();
    let data = vec![1.0_f32; 8];
    let dims = [2usize, 2, 2];
    let backend = MoiraiBackend::new();
    let t = Tensor::<f32, MoiraiBackend>::from_slice_on(dims, &data, &backend);
    // Rank matches D=3 (dims=[2,2,2] gives a rank-3 tensor); construction
    // succeeds by `AtlasImage::new`'s rank invariant.
    let img = AtlasImage::<f32, MoiraiBackend, 3>::new(t, origin, spacing, direction)
        .expect("Atlas image construction should succeed: tensor rank matches D=3");
    let out = AtlasBinaryErodeFilter::new(0).apply(&img).unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}
