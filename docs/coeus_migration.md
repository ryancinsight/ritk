# RITK Burn to Coeus Migration

## Status

RITK still uses Burn as its tensor, autodiff, and model backend. Coeus is the
target replacement, but it is not yet a drop-in dependency for RITK because the
RITK surface spans image containers, I/O construction, registration metrics,
autodiff transforms, neural models, CLI boundaries, Python bindings, and GPU
execution.

Evidence tier: manifest audit and source audit. No migration proof is claimed.

## Current Burn Surface

Run the repeatable audit from the RITK workspace root:

```sh
cargo run -p xtask -- burn-migration-audit
```

The audit reports:

- crate manifests that depend on `burn` or `burn-ndarray`;
- Rust source files containing Burn-surface tokens such as `burn::`,
  `Tensor<`, `TensorData`, `Shape::new`, `Autodiff`, `GradientsParams`,
  `Conv3d`, and `Param<`;
- a crate-level summary of manifest and source references;
- the Coeus capabilities required before dependency replacement.

The audit is intentionally lexical. It is a synchronization gate for planning,
not a type-level compatibility proof.

Baseline result on 2026-06-08:

- 18 manifest dependency files include `burn` or `burn-ndarray`;
- 490 Rust source files contain Burn-surface tokens;
- largest surfaces by token count: `ritk-core`, `ritk-registration`,
  `ritk-model`, `ritk-io`, and `ritk-cli`.

## Required Coeus Capabilities

The migration requires Coeus to provide these RITK-facing capabilities before
Burn can be removed:

| Capability | Required for |
| --- | --- |
| Tensor construction from shape plus `f32` slices | image readers, tests, CLI input assembly |
| Host extraction with explicit synchronization | image writers, CLI output, Python conversions |
| Rank-generic reshape, slice, transpose, broadcast, and permute | image layout transforms, registration points, format adapters |
| Elementwise arithmetic and reductions | filtering, statistics, metrics |
| Matmul and batched matrix operations | transforms, affine and registration code |
| Interpolation and sampling primitives | resampling, registration metrics, deformation fields |
| Reverse-mode autodiff | MI/MSE/NCC/CR metric optimization and transform parameters |
| Parameter/module API | `ritk-model` affine and TransMorph-style model code |
| 3-D convolution and pooling forward/backward | neural registration models |
| WGPU backend parity | GPU registration and model execution |
| Sparse/scatter or fused histogram GPU kernel | Parzen histogram direct/sparse paths |
| CPU reference backend | deterministic differential tests |
| PyO3 conversion boundary | Python remains a thin binding layer over Rust logic |

## GPU Migration Gates

Coeus WGPU work exists, including tensor transfers, elementwise operations,
matmul, convolution, pooling, activation, optimizer, and backward kernels. RITK
must not switch GPU registration to Coeus until these RITK-specific gates pass:

1. CPU Coeus backend matches current Burn NdArray results for representative
   RITK tensor operations.
2. WGPU Coeus backend matches the CPU Coeus backend for the same operations.
3. Registration metrics preserve autodiff tape connectivity; no host extraction
   is allowed on differentiable paths.
4. Parzen histogram GPU path has either a Coeus scatter-compatible
   implementation or a fused WGPU kernel with CPU differential tests.
5. RITK model 3-D convolution and pooling forward/backward tests pass on CPU and
   WGPU.
6. Python binding tests verify value-semantic equivalence through the Rust core.

## Development Sequence

1. Keep Burn as the production backend.
2. Maintain `burn-migration-audit` as the authoritative migration surface.
3. Introduce a RITK-owned tensor contract only after Coeus exposes all required
   CPU operations.
4. Add CPU Coeus differential tests behind an explicit feature.
5. Add WGPU Coeus differential tests after CPU parity is green.
6. Replace Burn call sites by crate boundary, starting with I/O construction and
   host extraction, then filters/statistics, then registration metrics, then
   `ritk-model`.
7. Remove Burn dependencies only after all call sites are migrated and the audit
   reports zero manifest dependencies and zero source references.

## Non-goals

- No Burn compatibility shim.
- No placeholder Coeus backend in RITK.
- No fallback branch that silently swaps Coeus failures back to Burn.
- No Python domain logic during the migration.
