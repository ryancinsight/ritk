# Coeus Tensor Integration

Coeus provides the N-dimensional tensor type, backend abstraction, and
automatic differentiation engine that RITK's image processing pipeline
is built on. The `Image<T, B, D>` type in `ritk-core` is a Coeus
`Tensor<T, B>` with fixed rank `D` and spatial metadata; every image
filter, registration transform, and diffusion model that operates on
pixel data does so through the Coeus tensor/backend contract.

This chapter documents the integration surface — the Coeus types and
traits that RITK consumes — rather than the provider internals, which
are documented in the Coeus book.

## Architecture

The Coeus tensor stack has three layers:

| Crate | Role |
|---|---|
| `coeus-core` | Fundamental abstractions: `Scalar`, `Layout`, `Storage`, `Backend`, `ComputeBackend` |
| `coeus-tensor` | Generic `Tensor<T, B>` with COW semantics, zero-copy views, constructors |
| `coeus-autograd` | `Var<T, B>` differentiable variable with forward/reverse-mode autodiff |

RITK consumes all three through `ritk-core`'s `Image` type and the
diffusion pipeline's model fitting.

## Backend Abstraction

### ComputeBackend

`ComputeBackend` is the device abstraction — the trait that lets the same
`Tensor<T, B>` code run on different hardware:

```rust,ignore
pub trait ComputeBackend: Send + Sync + Clone + 'static {
    type DeviceBuffer<T: Scalar>: StorageMut<T>;

    fn name(&self) -> &'static str;
    fn num_threads(&self) -> usize;
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T>;
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T);
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>);
    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]);
}
```

Two concrete backends ship in `coeus-core`:

| Backend | `num_threads()` | `parallel_for` behaviour |
|---|---|---|
| `SequentialBackend` | 1 | Runs indices in order, single-threaded |
| `MoiraiBackend` | `available_parallelism()` | Dispatches through Moirai's work-stealing scheduler |

`MoiraiBackend` is the default for `Tensor<T>` (when no backend type
parameter is supplied). It routes parallel work through the same unified
hybrid scheduler that serves both async and parallel tasks — see the
[Moirai Execution Backend](moirai_execution.md) chapter.

### Backend

The `Backend` trait extends `ComputeBackend` with a `parallel_for`
method:

```rust,ignore
pub unsafe trait Backend: ComputeBackend + Default {
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static;
}
```

The `Backend` trait is `unsafe` because implementors must guarantee that
`parallel_for` returns only after every invocation of `f` has completed.
CPU kernels use this synchronization to keep scoped borrows alive across
dispatch — the diffusion pipeline's voxel-chunk pattern depends on this.

## Storage Hierarchy

The storage traits form a sealed hierarchy that backs `Tensor<T, B>`:

| Trait | Provides |
|---|---|
| `Storage<T>` | `len()`, `is_empty()`, `allocate()`, `try_as_slice()` |
| `StorageMut<T>` | `try_as_mut_slice()`, `make_unique()` (COW trigger) |
| `CpuAddressableStorage<T>` | `as_slice()` — guaranteed CPU-readable |
| `CpuAddressableStorageMut<T>` | `as_mut_slice()` — CPU-writable with COW |

`CpuStorage<T>` is the standard CPU-side implementation backed by the
Mnemosyne allocator. `CowStorage<T>` wraps any `StorageMut<T>` with
copy-on-write semantics: `make_unique()` triggers a deep copy when the
allocation is shared.

## Tensor Type

`Tensor<T, B = MoiraiBackend>` is the generic N-dimensional array:

```rust,ignore
pub struct Tensor<T: Scalar, B: ComputeBackend = MoiraiBackend> {
    storage: B::DeviceBuffer<T>,
    layout: Layout,
    _backend: PhantomData<B>,
}
```

### Construction

| Method | Behaviour |
|---|---|
| `Tensor::zeros(shape)` | Zero-filled, default backend |
| `Tensor::ones(shape)` | One-filled |
| `Tensor::full(shape, value)` | Constant fill |
| `Tensor::from_slice(shape, &[T])` | Copy from host slice |
| `Tensor::from_vec(data)` | 1-D from owned `Vec<T>` |
| `Tensor::alloc_on(shape, &backend)` | Uninitialized allocation |
| `Tensor::zeros_on(shape, &backend)` | Zero-filled on named backend |
| `Tensor::from_raw_parts(storage, layout)` | Reconstruct from components |

Every constructor has an `_on` variant that accepts an explicit backend
reference, and a default-backend variant that uses `B::default()`.

### Accessors

| Method | Returns |
|---|---|
| `ndim() -> usize` | Number of dimensions |
| `numel() -> usize` | Total element count |
| `shape() -> &[usize]` | Shape as slice |
| `strides() -> &[usize]` | Strides as slice |
| `layout() -> &Layout` | Full layout descriptor |
| `is_contiguous() -> bool` | Row-major contiguous check |
| `as_slice() -> &[T]` | CPU host slice (panics if non-contiguous) |
| `as_mut_slice() -> &mut [T]` | Mutable host slice (COW triggers) |
| `get(&[usize]) -> T` | Element at logical index |
| `set(&[usize], T)` | Mutate element (COW triggers) |
| `to_vec() -> Vec<T>` | Materialize logical values row-major |

### Views

Zero-copy views share the underlying storage with the parent tensor:

| Operation | Method |
|---|---|
| Slice | `t.slice(&[(start, end), ...])` |
| Transpose | `t.t()` or `t.transpose(dims)` |
| Reshape | `t.reshape(shape)` |
| Permute | `t.permute(dims)` |
| Broadcast | `t.broadcast(shape)` |

Views carry their own `Layout` but reference the same storage allocation.
Mutation triggers copy-on-write: the view's `make_unique()` deep-copies
the shared buffer first.

### Backend Transfer

`to_backend(&new_backend)` copies tensor memory to a different device.
When source and destination are the same type, the transfer is a
zero-copy clone. When they differ, the data is read from the source
backend to host memory, then written to the destination backend.

```rust,ignore
use coeus_tensor::Tensor;
use coeus_core::{SequentialBackend, MoiraiBackend};

let t: Tensor<f32, SequentialBackend> = Tensor::from_slice([2, 3], &[1.0; 6]);
let t_moirai: Tensor<f32, MoiraiBackend> = t.to_backend(&MoiraiBackend::new());
```

## Image Integration in RITK

RITK's `Image<T, B, D>` is a rank-fixed wrapper around `Tensor<T, B>`
with spatial metadata (origin, spacing, direction cosine matrix).
The relationship is:

```rust,ignore
// Simplified — the actual Image type is in ritk-core
pub struct Image<T: Scalar, B: ComputeBackend, const D: usize> {
    tensor: Tensor<T, B>,   // pixel data, Coeus-owned
    origin: [f64; 3],       // physical origin (mm, LPS)
    spacing: [f64; 3],      // voxel spacing (mm)
    direction: [[f64; 3]; 3], // direction cosine matrix
}
```

This means:
- **Every image operation** (pixel access, filtering, resampling) runs
  through the Coeus tensor/backend contract.
- **Backend substitution** is seamless: `Image<f32, SequentialBackend, 3>`
  for deterministic single-threaded work, `Image<f32, MoiraiBackend, 3>`
  for parallel throughput — same pixel access API.
- **Zero-copy views** (slice, transpose, permute) share pixel storage
  with the parent image via Coeus's COW semantics.

## Autodiff Integration

`coeus-autograd` provides `Var<T, B>` — a differentiable variable that
wraps a `Tensor<T, B>` and tracks operations for gradient computation.
The diffusion pipeline uses this for nonlinear model fitting (DKI, NODDI):

```rust,ignore
use coeus_autograd::Var;
use coeus_tensor::Tensor;

let x: Var<f64> = Var::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0]));
let y = x.sin() + x.cos();
y.backward();
// x.grad() contains the gradient dy/dx
```

The DKI Levenberg-Marquardt path uses `coeus-autograd` to compute the
analytic Jacobian of the kurtosis signal model with respect to the 21
free parameters. The NODDI model combines analytic gradients for the
intra-axonal compartment with finite-difference gradients for the Watson
dispersion integral.

## Relationship to the Diffusion Pipeline

| Pipeline stage | Coeus type used | Role |
|---|---|---|
| Image I/O | `Tensor<T, B>` via `Image<T, B, D>` | Pixel storage and access |
| DTI design matrix | `Tensor<T>` (Leto adapter) | Matrix assembly into Leto arrays |
| DKI Jacobian | `Var<f64>` | Analytic derivative of signal model |
| NODDI gradient | `Var<f64>` | Mixed analytic + finite-diff Jacobian |
| Batch dispatch | `MoiraiBackend` | Per-voxel parallel fitting |

The Coeus tensor is the universal data carrier: image pixels, signal
vectors, design matrices, and Jacobian matrices all flow through
`Tensor<T, B>`. RITK adds spatial metadata and domain-specific
interpretation; Coeus provides the generic N-D array operations.

## Boundary

Coeus owns tensor storage, backends, autodiff, and the `Tensor<T, B>`
type. RITK owns the `Image<T, B, D>` wrapper with spatial metadata,
format-crate I/O, and domain-specific image processing. A RITK-local
tensor type or local gradient computation is a boundary violation —
the Coeus tensor is the single source of truth for pixel data.

## References

- [Coeus Nonlinear Least-Squares Solver](coeus_optim.md) — the LM solver
  that consumes Coeus tensors for Jacobian evaluation
- [Moirai Parallel Execution Backend](moirai_execution.md) — the default
  backend for `Tensor<T>`
- [Backend Dispatch](backend_dispatch.md) — RITK's backend selection
  and dispatch patterns
- [Diffusion Models](ritk_diffusion.md) — the models that consume Coeus
  tensors for design-matrix assembly and fitting
