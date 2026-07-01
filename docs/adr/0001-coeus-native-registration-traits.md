# ADR 0001 — Coeus-native registration `Transform`/`Metric` trait surface

- Status: Accepted
- Change class: [arch]
- Date: 2026-06-30
- Supersedes: none
- Related: `docs/coeus_migration.md` (dev-sequence step 6), backlog MIG-471…478

## Context

The burn→coeus migration (`docs/coeus_migration.md`) has landed a complete,
individually-verified set of Coeus-autograd registration primitives as free
functions in `ritk_registration::metric::coeus_autograd`:

- `mean_squared_error_coeus` — differentiable MSE loss reduction (MIG-471);
- `sample_linear_1d_coeus` / `sample_trilinear_coeus` — differentiable
  sampling (MIG-472/473);
- `translate_axis_coeus` / `affine_transform_coeus` — differentiable
  transforms (MIG-474/476);
- `translation_mse_coeus` / `affine_mse_coeus` — composed metrics (MIG-474/477);
- `optim::sgd_step_var` — a proven-convergent gradient-descent step (MIG-475).

The existing `ritk_core` `Transform<B: Backend, D>` and
`ritk_registration::metric::Metric<B: Backend, D>` traits are hard-bound to
`burn::tensor::{Backend, Tensor}` and used across many crates. The migration
needs a Coeus-native equivalent of those seams.

Two open questions this ADR resolves:

1. **Generalize the existing traits over the tensor substrate, or a parallel
   Coeus trait family?**
2. **Coordinate convention:** the sampler consumes per-axis `[N]` `Var`s; the
   affine transform emits `[N, 3]`. Which is canonical at the trait boundary?

## Decision

**1. Parallel Coeus-native trait family, not substrate-generalization.**
Introduce `CoeusTransform` (and a generic MSE metric over it) in
`ritk_registration::metric::coeus_autograd`, leaving the burn-bound `ritk_core`
traits untouched. Rationale:

- The burn traits are consumed by the classical/burn registration engine across
  crates; generalizing them over a tensor-substrate abstraction is a large
  breaking [major] change that would have to land atomically across every
  caller. The migration doc's sequence explicitly keeps Burn as the production
  backend and grows the Coeus path in parallel until all callers migrate, then
  removes Burn. A parallel Coeus trait family matches that: the Coeus path
  matures and is validated behind the `coeus` feature with zero risk to the
  Burn path.
- It also mirrors the parallel-path template already used throughout this
  migration (Coeus free functions alongside Burn ones, verified differentially).

**2. Canonical coordinate convention at the trait boundary: `[N, 3]`.**
`CoeusTransform::transform_points(&self, points: &Var[N,3]) -> Var[N,3]`, mirroring
the burn `Transform::transform_points` shape exactly. The generic MSE metric
splits the transformed `[N, 3]` into the three per-axis `[N]` `Var`s the
trilinear sampler consumes, via the differentiable `slice` + `reshape` (verified
tape-transparent in MIG-477). This unifies the split convention that was
previously spread between the sampler (per-axis) and the affine (`[N, 3]`): the
trait boundary is `[N, 3]`; per-axis is an internal sampler detail.

**3. One generic metric, transforms as implementors (DRY/SSOT).**
`mse_metric<Tf: CoeusTransform>(moving, dims, fixed, grid, &transform)` is the
single composition SSOT (`mse ∘ trilinear ∘ split ∘ transform`). `Translation`
and `Affine` are the initial `CoeusTransform` implementors — two real
implementors justify the seam (not speculative; seam-first design validated
against ≥2 implementors). `affine_mse_coeus` is refactored to delegate to
`mse_metric` (removing its duplicated composition). A future NCC/MI metric adds
a second metric type; a future rigid/versor transform adds a third implementor.

## Consequences

- The Coeus registration seam exists and is exercised, behind the `coeus`
  feature, with no change to the Burn path.
- `translation_mse_coeus` (per-axis signature, MIG-474) is superseded by
  `Translation` + `mse_metric`. It is retained this increment to avoid churning
  its merged tests; its consolidation onto the trait path (migrating its
  per-axis callers to `[N, 3]`) is filed as MIG-479-01.
- The Coeus-native `Metric` trait itself (analogous to the burn `Metric`) is
  **not** introduced yet: only one metric type (MSE) exists, so a `CoeusMetric`
  trait would be a single-implementor abstraction (YAGNI). It is filed to be
  introduced alongside the second metric type (NCC), when a second implementor
  justifies the seam. `CoeusTransform` *is* introduced now because it already
  has two implementors.

## Verification

- Trait-based path (`mse_metric` with `Affine`) is differentially identical to
  the `affine_mse_coeus` free function it now delegates through; `Translation`
  via `mse_metric` matches the translation composition; a gradient-descent loop
  through the trait converges. All under the analytical/`SequentialBackend`
  regime used across the primitive suite.
