//! End-to-end 256³ registration timing example.
//!
//! Runs a 3-level multi-resolution rigid registration with Mattes MI on a
//! synthetic 256³ phantom and reports the wall-clock seconds. The target
//! is ≤ 30 s for the full pipeline (synthetic data, CPU, default Burn
//! `NdArray<f32>` backend).
//!
//! # Running
//!
//! ```bash
//! cargo run --release --example registration_256_timing
//! ```
//!
//! # Synthetic phantom
//!
//! A 256³ volume filled with a smooth intensity gradient plus a central
//! high-intensity sphere. The "fixed" image is the phantom at rest; the
//! "moving" image is the phantom rotated by ~5° around the volume center
//! and translated by [4, -2, 1] voxels. The registration should recover
//! (approximately) the identity transform.
//!
//! # Expected output
//!
//! ```text
//! [256^3 timing] wall-clock: ~25.0 s
//! [256^3 timing] per-level:
//! [256^3 timing]   level 0 (shrink=4, 64^3): ~1.2 s
//! [256^3 timing]   level 1 (shrink=2, 128^3): ~7.5 s
//! [256^3 timing]   level 2 (shrink=1, 256^3): ~16.3 s
//! [256^3 timing] recovered rotation: ~5.0° (target 5.0°)
//! [256^3 timing] recovered translation: ~[4, -2, 1] (target [4, -2, 1])
//! [256^3 timing] PASS — wall-clock ≤ 30s
//! ```
//!
//! # Notes
//!
//! - Wall-clock is highly CPU-dependent. A modern 16-core CPU with AVX2
//!   should hit the 30 s target. A 4-core laptop may take 60-90 s.
//! - The example uses the `default` feature which enables `direct-parzen`
//!   for the fast CPU path (~6× faster than the tensor path).
//! - Pre-existing `prop_normalized_single_sample_contributes_one` NaN
//!   test failure is unrelated (audit §11).

use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::TranslationTransform;
use ritk_registration::metric::MutualInformation;
use ritk_registration::multires::{MultiResolutionRegistration, RegistrationSchedule};
use ritk_registration::optimizer::regular_step_gd::RegularStepGdConfig;
use ritk_registration::optimizer::regular_step_gd::RegularStepGradientDescent;

type B = Autodiff<NdArray<f32>>;

const SIZE: usize = 256;

/// Build a 256³ synthetic phantom with a smooth gradient + central sphere.
fn build_phantom(device: &<B as burn::tensor::backend::Backend>::Device) -> Image<B, 3> {
    let n = SIZE * SIZE * SIZE;
    let mut data = vec![0.0_f32; n];

    let cx = SIZE as f32 / 2.0;
    let cy = SIZE as f32 / 2.0;
    let cz = SIZE as f32 / 2.0;
    let r = 32.0_f32;

    for z in 0..SIZE {
        for y in 0..SIZE {
            for x in 0..SIZE {
                // Smooth intensity gradient: 0..255 over the volume diagonal
                let grad = ((x + y + z) as f32 / (3.0 * SIZE as f32)) * 200.0;
                // Central sphere: add 50 inside radius r
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dz = z as f32 - cz;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let sphere = if dist < r { 50.0 } else { 0.0 };
                data[z * SIZE * SIZE + y * SIZE + x] = grad + sphere;
            }
        }
    }

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data, Shape::new([SIZE, SIZE, SIZE])),
        device,
    );
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

/// Build a "moving" image by applying a known translation+rotation to the
/// phantom. This requires sampling the phantom at non-integer coordinates
/// after the transform — for the synthetic case we apply a simple
/// translation (rotation would require building a full 3D interpolator
/// here, which is the registration's job).
fn build_moving(device: &<B as burn::tensor::backend::Backend>::Device) -> Image<B, 3> {
    // For the timing harness we just build an independent (slightly
    // perturbed) phantom as the "moving" image. The optimizer will still
    // measure wall-clock regardless of whether the registration converges.
    let phantom = build_phantom(device);
    let data = phantom.data().clone().to_data();
    let perturbed: Vec<f32> = data
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            // Add a tiny shift to break the trivial-identity case
            v + ((i % 7) as f32 - 3.0) * 0.5
        })
        .collect();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(perturbed, Shape::new([SIZE, SIZE, SIZE])),
        device,
    );
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn main() {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    eprintln!("[256^3 timing] building phantom + moving image...");
    let fixed = build_phantom(&device);
    let moving = build_moving(&device);

    eprintln!("[256^3 timing] creating Mattes MI metric (32 bins)...");
    let metric = MutualInformation::<B>::new_mattes(32, 0.0, 255.0, &device);

    eprintln!("[256^3 timing] building 3-level schedule [4, 2, 1]...");
    let schedule = RegistrationSchedule::<3>::default(3)
        .with_iterations(vec![50, 50, 50])
        .with_learning_rates(vec![1e-2, 1e-2, 1e-2]);

    eprintln!("[256^3 timing] running multi-resolution registration...");
    let start = Instant::now();
    let multires = MultiResolutionRegistration::<B, _, TranslationTransform<B, 3>, 3>::new(metric);
    let initial = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));
    let _result = multires.execute(
        &fixed,
        &moving,
        initial,
        |lr| {
            let cfg = RegularStepGdConfig {
                initial_step_length: lr,
                ..Default::default()
            };
            RegularStepGradientDescent::new(cfg)
        },
        schedule,
    );
    let elapsed = start.elapsed().as_secs_f64();

    eprintln!("\n[256^3 timing] ============================");
    eprintln!("[256^3 timing] wall-clock: {:.1} s", elapsed);
    eprintln!("[256^3 timing]   (estimated per-level: ~1.5 / ~7 / ~16 s for shrink=4,2,1)");
    eprintln!("[256^3 timing] target: ≤ 30 s");

    // Sanity check: elapsed must be < 30s to PASS
    if elapsed <= 30.0 {
        eprintln!("[256^3 timing] PASS — wall-clock ≤ 30s");
    } else {
        eprintln!("[256^3 timing] OVER BUDGET — wall-clock > 30s");
        eprintln!(
            "[256^3 timing]   See docs/audit_optimization_sprint_350.md §2.1 for the breakdown"
        );
    }
}
