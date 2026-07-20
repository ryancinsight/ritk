# Example: Gradient Recursive Gaussian Benchmark

Micro-benchmark: GradientRecursiveGaussian buffer-level vs per-`Image` passes.

## Source

`crates/ritk-filter/examples/bench_gradient_rg.rs`

## Description

Compares the optimized `gradient_recursive_gaussian_components` (one tensor
extraction, component-wise smoothing) against the per-`Image` filter
path. Both exercise the identical substrate-agnostic host core via
`ritk-image::Image` boundary. No Burn tensor is constructed.

## Usage

```bash
cargo run --release --example bench_gradient_rg
```

## Verification

- Reports timing for both buffer-level and per-Image paths
- Uses `assert_coeus_matches_coeus` to verify bitwise-identical results
- Demonstrates the native host core performance vs wrapper overhead
