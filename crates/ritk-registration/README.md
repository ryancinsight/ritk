# ritk-registration

Medical image registration algorithms for [RITK](https://github.com/ryancinsight/ritk).

Classical and deformable registration under one API: Coeus autograd operations
for differentiable paths, deterministic CPU algorithms for classical paths.

| Algorithm | Category |
|---|---|
| Kabsch SVD | Classical rigid alignment |
| MI-based rigid / affine | Classical iterative |
| Thirion / Diffeomorphic / Symmetric Demons | Deformable |
| Greedy SyN, Multi-Resolution SyN, BSpline SyN | Diffeomorphic |
| BSpline FFD | Deformable |
| LDDMM | Diffeomorphic |
| Groupwise Atlas | Template building (iterative SyN) |
| Joint Label Fusion (Wang 2013), Majority Voting | Multi-atlas label fusion |

**Metrics** — MSE, NCC, LNCC, NGF, and deep-learning losses here; histogram
Mutual Information (Standard / Mattes / NMI) lives in
`ritk-statistics::information` and is consumed by the classical engine.

**Optimizers** — the autodiff gradient-descent driver, plus the Coeus
optimizers (SGD with momentum, Adam, AdamW, AdaGrad, RMSProp).

**Regularization** — Bending energy, curvature, diffusion, elastic, total
variation.

## Usage

```toml
[dependencies]
ritk-registration = "0.54.0"
```
