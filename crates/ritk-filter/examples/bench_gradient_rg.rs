//! Micro-benchmark: GradientRecursiveGaussian buffer-level vs per-`Image` passes.
//!
//! Run with: `cargo run --release --example bench_gradient_rg`
//!
//! Compares the optimized `gradient_recursive_gaussian_components` (one tensor
//! extraction, raw-buffer chaining) against the previous structure (nine
//! `recursive_gaussian_directional` calls, each extracting + rebuilding an
//! `Image`). Both compute the identical vector gradient; this measures only the
//! per-pass `Image` alloc/rebuild overhead the refactor removed.

use std::time::Instant;

use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use ritk_filter::{
    gradient_recursive_gaussian_components, recursive_gaussian::recursive_gaussian_directional,
    DerivativeOrder,
};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make_image(n: usize) -> Image<B, 3> {
    let data: Vec<f32> = (0..n * n * n).map(|i| (i % 251) as f32).collect();
    let device = NdArrayDevice::default();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new([n, n, n])), &device);
    Image::new(
        tensor,
        ritk_core::spatial::Point::origin(),
        ritk_core::spatial::Spacing::new([1.0, 1.0, 1.0]),
        ritk_core::spatial::Direction::identity(),
    )
}

/// Old structure: nine `recursive_gaussian_directional` calls (extract+rebuild each).
fn old_path(image: &Image<B, 3>, sigma: f64) -> [Vec<f32>; 3] {
    let comp = |axis_k: usize| -> Vec<f32> {
        let others: Vec<usize> = (0..3).filter(|&a| a != axis_k).collect();
        let mut cur =
            recursive_gaussian_directional(image, sigma, DerivativeOrder::Zero, others[0]).unwrap();
        cur =
            recursive_gaussian_directional(&cur, sigma, DerivativeOrder::Zero, others[1]).unwrap();
        cur = recursive_gaussian_directional(&cur, sigma, DerivativeOrder::First, axis_k).unwrap();
        let inv = 1.0_f32 / image.spacing()[axis_k] as f32;
        let (mut v, _) = extract_vec_infallible(&cur);
        for x in v.iter_mut() {
            *x *= inv;
        }
        v
    };
    [comp(0), comp(1), comp(2)]
}

fn main() {
    let n = 128;
    let sigma = 1.5;
    let img = make_image(n);

    // Warm up + correctness cross-check (must be float-identical).
    let a = old_path(&img, sigma);
    let b = gradient_recursive_gaussian_components(&img, sigma).unwrap();
    let mut maxdiff = 0.0f32;
    for k in 0..3 {
        for (x, y) in a[k].iter().zip(b[k].iter()) {
            maxdiff = maxdiff.max((x - y).abs());
        }
    }
    println!("max |old − new| = {maxdiff:e} (expect 0)");

    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(old_path(&img, sigma));
    }
    let old_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(gradient_recursive_gaussian_components(&img, sigma).unwrap());
    }
    let new_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!("{n}^3 volume, sigma {sigma}, {iters} iters:");
    println!("  old (9× Image passes):  {old_ms:.2} ms/call");
    println!("  new (buffer-level):     {new_ms:.2} ms/call");
    println!("  speedup: {:.2}×", old_ms / new_ms);
}
