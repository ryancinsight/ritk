# Coeus Tensor Integration

Integration of RITK with the Coeus autodiff and tensor stack.

## Architecture

RITK's `apply_native` path exercises the identical substrate-agnostic host
core as Coeus's `Tensor::apply`. Both paths call `convolve_separable`,
`discrete_gaussian_smooth_flat`, or other pure functions — the only
difference is the tensor backend (Sequential vs Moirai).

## SSOT Boundaries

- `ritk-image::Image` — image data structure with Coeus backend
- `coeus_core::SequentialBackend` — sequential execution backend
- `coeus_core::MoiraiBackend` — parallel execution backend

## Comparison Infrastructure

`ritk-filter::native_support::assert_coeus_matches_coeus` differentially
compares the Coeus-generic `apply` path against `apply_native`:
- Builds identical input tensors on both backends
- Applies the filter via both paths
- Asserts bitwise-identical output

## Zero-cost Guarantee

Every `apply_native` wrapper is zero-cost: the compiler monomorphizes
the generic body to concrete specializations, producing machine code
identical to hand-written concrete implementations.
