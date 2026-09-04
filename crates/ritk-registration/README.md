# ritk-registration

Medical image registration algorithms for [RITK](https://github.com/ryancinsight/ritk).

Classical and deformable registration under one API: Coeus autograd operations
for differentiable paths, deterministic CPU algorithms for classical paths.

| Algorithm | Category |
|---|---|
| Kabsch SVD | Classical rigid alignment |
| MI-based rigid / affine | Classical iterative |
| Packed MIND-SSC | Multimodal rigid similarity |
| Thirion / Diffeomorphic / Symmetric Demons | Deformable |
| Greedy SyN, Multi-Resolution SyN, BSpline SyN | Diffeomorphic |
| BSpline FFD | Deformable |
| LDDMM | Diffeomorphic |
| Groupwise Atlas | Template building (iterative SyN) |
| Joint Label Fusion (Wang 2013), Majority Voting | Multi-atlas label fusion |

**Metrics** — MSE, NCC, LNCC, NGF, packed fixed-domain MIND-SSC, and
deep-learning losses here; histogram
Mutual Information (Standard / Mattes / NMI) lives in
`ritk-statistics::information` and is consumed by the classical engine.

For repeated CT/MR rigid-pose scoring, prepare the fixed MIND-SSC state once:

```rust
use ritk_registration::metric::mind::{MindSscConfig, MindSscFixedPrep};
use ritk_registration::AffineTransform;

# fn score<B>(fixed: &ritk_image::Image<f32, B, 3>, moving: &ritk_image::Image<f32, B, 3>) -> Result<f32, Box<dyn std::error::Error>>
# where B: coeus_core::ComputeBackend, B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32> {
let prepared = MindSscFixedPrep::try_new(fixed, MindSscConfig::default(), None, None)?;
let similarity = prepared.eval(moving, &AffineTransform::IDENTITY)?;
# Ok(similarity)
# }
```

The default retains at most 8,192 deterministic, physically stratified fixed
centers. Moving descriptors are evaluated at those centers on demand; no dense
moving descriptor volume is built per pose.

**Optimizers** — the autodiff gradient-descent driver, plus the Coeus
optimizers (SGD with momentum, Adam, AdamW, AdaGrad, RMSProp).

**Regularization** — Bending energy, curvature, diffusion, elastic, total
variation.

## Usage

```toml
[dependencies]
ritk-registration = "0.54.0"
```
