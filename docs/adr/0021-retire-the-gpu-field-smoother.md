# ADR 0021: Retire the GPU field smoother

- **Status:** Accepted
- **Board item:** `RITK-GPU-SMOOTHER-REACH`
- **Class:** [major] [arch]
- **Date:** 2026-08-21

## Context

`ritk-registration` exposes two smoother surfaces alongside `CpuFieldSmoother`:
`GpuFieldSmoother<B: Backend>` and `CpuOrGpu<B>`, a two-variant enum selecting
between them. Four facts about them, each verified against the tree:

**It cannot be instantiated on a device.** `Backend` resolves to
`coeus_core::Backend`, which is `ComputeBackend + Default` plus `parallel_for`.
The only implementors anywhere in the stack are `SequentialBackend` and
`MoiraiBackend`, both CPU. `coeus_wgpu::WgpuBackend` implements `ComputeBackend`
and not `Backend`, so no GPU backend can be substituted. No RITK manifest
declares `coeus-wgpu`, and nothing in the workspace constructs a
`GpuFieldSmoother`.

**Its convolution runs on the host regardless.** `GaussianFilter::apply_tensor`
calls `input.to_vec()`, runs `convolve_zero_pad_nd` — a scalar loop over a
`Vec<f32>` — and rebuilds with `Tensor::from_slice`. No device kernel is
dispatched. The sibling `DiscreteGaussianFilter::apply_native` says so in its own
documentation: it "runs the identical separable discrete-Gaussian convolution …
on the image's contiguous host buffer". Making the type instantiable would
therefore not make it a GPU path; it would be a host convolution wrapped in two
device transfers, which is strictly slower than `CpuFieldSmoother`.

**It documents a measurement no code path can produce.** Its Rustdoc states "On
an RTX 3060, smoothing a 256³ field takes ~4 ms vs ~80 ms for the CPU `moirai`
-based path." Nothing in this repository can generate that number.

**The enum's second variant is dead.** Every construction site in the workspace —
`atlas/mod.rs:121` and `diffeomorphic/multires_syn/registration.rs:47` — builds
`CpuOrGpu::Cpu(CpuFieldSmoother::new(..))`. `CpuOrGpu::Gpu` is never constructed.

A surface named for a capability it does not have, carrying a benchmark it cannot
reproduce, is the mock class the engineering rules prohibit: the output does not
depend on the capability the name advertises.

## Decision

Delete `GpuFieldSmoother` and `CpuOrGpu`. Keep `CpuFieldSmoother`, and keep the
`FieldSmoother` trait as the extension seam.

The per-level factory parameters change from a concrete enum to the trait:

```text
smoother_factory: &mut impl FnMut([usize; 3]) -> CpuOrGpu<B>
smoother_factory: &mut impl FnMut([usize; 3]) -> S   where S: FieldSmoother
```

Nothing is lost by this. `CpuOrGpu` existed to avoid `Box<dyn FieldSmoother>` in
multi-resolution loops, and a generic parameter avoids the box just as well while
admitting *any* implementor rather than exactly two. The trait's own
documentation already describes this as the intended shape — "Registration
engines accept `impl FieldSmoother` so callers choose the backend at the call
site" — so this brings the code to the design its docs already state.

A genuine device smoother remains possible and becomes easier: it is a new
`FieldSmoother` implementor, added when a real device convolution exists behind
it, with no enum to widen and no bound to relax.

## Consequences

- `GpuFieldSmoother` and `CpuOrGpu` leave the public API. This is [major];
  in-repo callers migrate in the same change and no compatibility re-export is
  left behind.
- The registration entry points become generic over the smoother rather than over
  a backend they never used, which removes the `B: Backend` parameter from
  signatures whose bodies never touch a backend.
- The RTX 3060 claim is removed rather than restated, because no run in this
  repository produced it.
- RITK still declares no dependency on any GPU backend, and after this change it
  no longer implies one.

## Alternatives rejected

- **Re-bound to `ComputeBackend` so `WgpuBackend` fits.** Tempting, and
  `DiscreteGaussianFilter::apply_native` already establishes `ComputeBackend +
  Default` as sufficient for this work. Rejected because it fixes the wrong half:
  the type would become instantiable while still convolving on the host, so it
  would be a slower CPU path wearing a GPU name — the false claim preserved and
  made harder to notice.
- **Implement a real device separable convolution now.** The correct end state,
  and the reason the `FieldSmoother` seam is kept. Rejected as this change's
  scope: it needs a device convolution primitive and a differential test against
  the CPU smoother at a derived tolerance, which is its own item with its own
  acceptance. Shipping the deletion first means the tree stops advertising a
  capability while that work is done, rather than after.
- **Document the limitation and keep the types.** Rejected: a caller cannot
  discover from a green build that the GPU variant they selected is unreachable,
  and the variant is unconstructible in any case.
