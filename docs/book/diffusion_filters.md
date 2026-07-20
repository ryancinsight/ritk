# Diffusion Filtering

PDE-based image smoothing: Perona-Malik, curvature flow, curvature diffusion,
and gradient anisotropic diffusion.

## Design

All diffusion filters operate on flat host buffers via substrate-agnostic pure
functions. Each filter implements a host core shared by the Coeus-generic
`apply` path and the native `apply_native` path. No Coeus tensor is constructed
in the hot path.

## Perona-Malik

Edge-preserving smoothing via the Perona-Malik PDE:
```
dI/dt = div(g(|∇I|) ∇I)
```
where `g` is the edge-stopping function. Two variants: `g1` (exponential)
and `g2` (inverse quadratic).

## Curvature Flow

```
dI/dt = ∇·(∇I/|∇I|) |∇I|
```
Smooths along edges while preserving across-edge structure.

## Curvature Diffusion

```
dI/dt = ∇²I
```
Isotropic smoothing equivalent to heat equation.

## Gradient Anisotropic Diffusion

```
dI/dt = ∇·(g(|∇I|) ∇I)
```
where `g` is a function of gradient magnitude. Smoothing is
anisotropic: strong along edges, weak across edges.

## Verification

Each filter is differentially tested against its Coeus-generic counterpart
via `assert_coeus_matches_coeus`.
