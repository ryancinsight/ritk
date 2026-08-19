# ritk-block-matching

Dependency-light block-matching displacement estimation for speckle tracking
and elastography.

The crate compares a fixed block with candidate blocks in a moving image using
zero-mean normalized cross-correlation and returns the displacement of the
best peak. Parabolic and cosine subpixel refinements are available. Inputs are
caller-owned flat row-major `[z, y, x]` sample buffers; the crate does not
depend on an image, tensor, or device backend.

```rust
use ritk_block_matching::{
    match_block, BlockMatchingConfig, SubpixelRefinement,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixed: Vec<f32> = (0..81)
        .map(|index| (index % 11) as f32 + index as f32 * 0.001)
        .collect();
    let moving = fixed.clone();
    let config = BlockMatchingConfig {
        block_radius: [0, 1, 1],
        search_radius: [0, 1, 1],
    };
    let displacement = match_block(
        &fixed,
        &moving,
        [1, 9, 9],
        [0, 4, 4],
        config,
        SubpixelRefinement::Parabolic,
    )?;
    assert!(displacement.peak_similarity > 0.99);
    Ok(())
}
```

Use [`MultiResolutionSearch`](https://docs.rs/ritk-block-matching/latest/ritk_block_matching/struct.MultiResolutionSearch.html)
when the caller owns a coarse-to-fine image pyramid. The optional `fft`
feature provides a finite, zero-padded FFT-backed metric through the Apollo
provider.

The algorithm follows the metric-image and displacement-calculator split from
ITKUltrasound and the sub-sample estimators described by Céspedes et al.
(1995). Boundary candidates are left unevaluated rather than padded or
silently clamped.
