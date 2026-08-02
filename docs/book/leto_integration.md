# Leto Operations Integration

Leto is the Atlas-owned N-dimensional strided array library that provides the
array storage, layout descriptors, zero-copy views, and storage backends used
by every numeric operation in the RITK pipeline. The diffusion models assemble
design matrices into `Array2<f64>`, signal vectors into `Array1<f64>`, and
pass them to `leto-ops` solvers through the `.view()` method.

This chapter documents the integration surface — the Leto array types, views,
and storage that RITK consumes — rather than the solver operations themselves,
which are covered in the [Leto Linear Algebra Operations](leto_linalg.md)
provider chapter. The two chapters are paired: the provider chapter documents
what Leto computes; this chapter documents how RITK holds and passes the data.

## Architecture

Leto has a single domain crate (`leto`) that carries arrays, views, and
storage, plus a companion operations crate (`leto-ops`) that carries the
linear algebra and statistical kernels. The crates are separated so that
consumers that only need array storage (like `apollo-sht`) can depend on
`leto` without pulling in the decompositions and solvers.

| Crate | Role |
|---|---|
| `leto` | N-dimensional `Array<T, S, N>`, `ArrayView`, `ArrayViewMut`, storage backends, layout descriptors |
| `leto-ops` | `solve_least_squares`, `nnls`, `cholesky_solve`, `qr_decompose`, `eigenvalues`, norms, iterative solvers |
| `coeus-leto` | Dynamic-rank (Coeus) to const-rank (Leto) dispatch shim |

RITK consumes `leto` directly for `Array1`/`Array2` construction, and
`leto-ops` for every linear solve in the diffusion pipeline.

## Array Type

`Array<T, S, N>` is the generic N-dimensional strided array:

```rust,ignore
pub struct Array<T, S, const N: usize> {
    layout: Layout<N>,    // shape, strides, offset
    storage: S,           // heap-allocated or borrowed backing
    _marker: PhantomData<T>,
}
```

The type is rank-generic (`const N: usize`) and storage-generic (`S`),
which means the same array API works for 1-D signals, 2-D matrices,
and 3-D image volumes, backed by any storage that implements `Storage<T>`.

### Type Aliases

Most RITK code uses rank-specific type aliases that fix `S = VecStorage<T>`:

| Alias | Rank | Typical use in RITK |
|---|---|---|
| `Array1<T>` | 1 | Signal vectors, coefficient vectors |
| `Array2<T>` | 2 | Design matrices, augmented systems |
| `Array3<T>` | 3 | Image volumes |
| `Array4<T>` | 4 | 4-D DWI series |
| `ArrayD<T>` | Fixed-large rank | Higher-rank tensors (still const-rank) |

```rust,ignore
use leto::{Array1, Array2};

// DTI: one design matrix per voxel (N_dwi x 6)
let design: Array2<f64> = Array2::zeros([n_dwi, 6]);

// CSD: non-negative fODF coefficients
let coefficients: Array1<f64> = result.solution;
```

### Construction

| Method | Description |
|---|---|
| `Array1::zeros([n])` | Zero-filled 1-D array |
| `Array2::zeros([rows, cols])` | Zero-filled 2-D array |
| `Array1::from_vec(n, data)` | 1-D from owned `Vec<T>` |
| `Array2::from_shape_vec([r, c], data)` | 2-D from `Vec<T>`, validates shape |
| `Array::new(layout, storage)` | From explicit layout + storage |

```rust,ignore
use leto::Array1;

// Signal vector from per-voxel DWI values
let n_dwi = 60;
let signal: Vec<f64> = (0..n_dwi).map(|i| measured_signal[i]).collect();
let rhs = Array1::from_vec(n_dwi, signal);
```

### Accessors

| Method | Returns |
|---|---|
| `shape() -> [usize; N]` | Shape as const array |
| `strides() -> [isize; N]` | Strides as const array |
| `offset() -> usize` | Physical base offset |
| `len() -> usize` | Logical element count |
| `is_empty() -> bool` | True when any axis is zero |
| `layout() -> Layout<N>` | Full layout descriptor |
| `get([i, j]) -> Result<&T>` | Element at logical index |
| `as_slice() -> Option<&[T]>` | Contiguous row-major slice |
| `as_slice_mut() -> Option<&mut [T]>` | Mutable contiguous slice |
| `iter() -> ElementIter` | Logical row-major iterator |
| `indexed_iter() -> IndexedIter` | `(multi-index, &elem)` iterator |
| `to_contiguous() -> Array<T, VecStorage<T>, N>` | Materialize C-contiguous copy |

### Views

The `.view()` method returns an `ArrayView<T, N>` — a zero-copy borrowed
reference that shares the parent array's storage:

```rust,ignore
let design: Array2<f64> = Array2::zeros([60, 6]);
let signal: Array1<f64> = Array1::zeros([60]);

// Pass zero-copy views to leto-ops solvers
let solution = leto_ops::solve_least_squares(
    &design.view(),    // ArrayView2<f64>
    &signal.view(),    // ArrayView1<f64>
)?;
```

This is the universal pattern in the diffusion pipeline: construct arrays,
then pass `.view()` to solvers. The view is read-only, zero-cost, and
validates that the layout does not exceed the backing storage.

### ArrayView and ArrayViewMut

`ArrayView` provides the same shape, strides, indexing, slicing, and
iteration methods as `Array`, but borrows its data from a parent:

| View method | Description |
|---|---|
| `view.shape()` | Const shape |
| `view.get([i, j]) -> Result<&T>` | Logical indexing |
| `view.slice(&ranges)` | Zero-copy sub-view |
| `view.transpose(axes)` | Zero-copy transposed view |
| `view.reshape(shape)` | Reinterpret shape (same numel) |
| `view.to_contiguous()` | Materialize owned `Array<T, VecStorage<T>, N>` |
| `view.as_array()` | Zero-copy borrowed `Array<T, SliceStorage<T>, N>` |

`ArrayViewMut` adds `get_mut`, `fill`, `assign`, and mutable slice accessors.

## Storage Backends

Leto's storage trait hierarchy parallels Coeus's but is independently owned:

| Type | Description |
|---|---|
| `VecStorage<T>` | Heap-allocated `Vec<T>` — the default for `Array1`/`Array2` |
| `SliceStorage<'a, T>` | Borrowed `&'a [T]` — created by `view.as_array()` |
| `CowStorage<T>` | Copy-on-write wrapper over any `StorageMut<T>` |
| `MnemosyneStorage<T>` | Custom-allocated via Mnemosyne (feature-gated) |
| `StackStorage<T, N>` | Fixed-size stack allocation |

In RITK, `VecStorage<T>` is the default for construction; `SliceStorage` is
used by zero-copy view-to-array conversion; `MnemosyneStorage` is used by
the Coeus tensor stack for custom allocator integration.

## Layout System

`Layout<N>` carries shape, strides, and offset — the metadata that
describes how logical indices map to physical positions in a flat slice:

```rust,ignore
pub struct Layout<const N: usize> {
    pub shape: [usize; N],
    pub strides: [isize; N],
    pub offset: usize,
}
```

Layout validation ensures that no logical index accesses memory before the
offset or beyond the storage length. Contiguity predicates determine
whether `as_slice()` is safe:

| Method | Condition |
|---|---|
| `is_c_contiguous()` | Row-major strides, offset 0 |
| `is_f_contiguous()` | Column-major strides, offset 0 |
| `is_c_dense()` | Row-major strides, any offset |
| `is_contiguous()` | Dense in either C or F order, any offset |

The diffusion pipeline's arrays are always C-contiguous at offset 0, so
`.as_slice()` and `.view()` always succeed without materialization.

## coeus-leto Interop

The `coeus-leto` crate bridges the dynamic-rank Coeus `Layout` to Leto's
const-rank `Layout<N>`:

```rust,ignore
use coeus_leto::{to_leto_view, contiguous_values};

// Dynamic-rank Coeus tensor to const-rank Leto view
let coeus_layout = tensor.layout(); // coeus_core::Layout (rank at runtime)
let leto_view = to_leto_view::<f64, 2>(coeus_layout, tensor.as_slice())?;

// Materialize logical values from any layout
let values = contiguous_values(coeus_layout, tensor.as_slice())?;
```

The dispatch shim resolves Coeus's runtime rank to a Leto `const N` through
a bounded `match` (up to `MAX_DISPATCH_RANK = 6`), then calls the
monomorphized Leto kernel. This keeps Leto purely const-rank (the source of
its compile-time shape safety) and consolidates all CPU array operations
through one authoritative kernel set.

In the diffusion pipeline, `contiguous_values` is called by the DTI/DKI
estimators when building Leto arrays from Coeus tensor slices — it extracts
logical row-major values from potentially strided or offset Coeus storage.

## Diffusion Pipeline Usage

Every linear solve in the diffusion pipeline follows the same pattern:

```rust,ignore
use leto::{Array1, Array2};
use leto_ops::solve_least_squares;

// 1. Assemble design matrix and RHS in Leto arrays
let mut design = Array2::zeros([n_dwi, 6]);
let mut log_signals = Array1::zeros([n_dwi]);

for i in 0..n_dwi {
    let (x, y, z) = scheme.directions[i].unit_direction();
    design[[i, 0]] = x * x;
    design[[i, 1]] = y * y;
    // ... fill Voigt design matrix ...
    log_signals[i] = log(signal[i]);
}

// 2. Solve via leto-ops
let solution = solve_least_squares(
    &design.view(),
    &log_signals.view(),
)?;
```

The mapping between model and Leto constructs is uniform:

| Model | Design matrix | RHS vector | Solver | Result type |
|---|---|---|---|---|
| DTI | `Array2<f64>` (N×6 Voigt) | `Array1<f64>` (log signals) | `solve_least_squares` | `Array1<f64>` (6 tensor elements) |
| ODF | `Array2<f64>` (augmented, N+K×K) | `Array1<f64>` (augmented RHS) | `solve_least_squares` | `Array1<f64>` (K SH coefficients) |
| CSD | `Array2<f64>` (N×K deconvolution) | `Array1<f64>` (signals) | `nnls` | `Array1<f64>` (K non-negative coeffs) |

## Iterators

Leto provides several zero-allocation iterator types over arrays and views:

| Iterator | Yields | Notes |
|---|---|---|
| `iter()` | `&T` in logical row-major order | Works on any strides |
| `indexed_iter()` | `([usize; N], &T)` pairs | Multi-index + value |
| `exact_chunks([cs; N])` | `ArrayView<T, N>` sub-views | Non-overlapping tiles |
| `axis_chunks_iter(axis, len)` | `ArrayView<T, N>` sub-views | Chunks along one axis |
| `windows([ws; N])` | `ArrayView<T, N>` sub-views | Sliding windows |
| `lanes::<M>(axis)` | 1-D lane views | Strided axis-as-lane views |
| `axis_iter::<M>(axis)` | Sub-views of rank N-1 | Dimension-reducing iteration |

These iterators are used by the diffusion pipeline's volumetric fitting
loops — for example, `axis_chunks_iter(0, 256)` parcels voxels into
256-element chunks for batched dispatch through Moirai.

## Sparse Storage

Leto provides three sparse matrix formats in `leto::sparse`:

| Format | Type | Description |
|---|---|---|
| CSR | `CsrArray<T>` | Compressed Sparse Row — efficient SpMV |
| CSC | `CscArray<T>` | Compressed Sparse Column — efficient SpMV transpose |
| COO | `CooArray<T>` | Coordinate list — construction format |

These are consumed by `coeus-leto`'s `spmm_into` / `spmv_into` dispatch
kernels for sparse matrix-matrix and matrix-vector multiplication.
Non-diffusion pipelines (graph algorithms, registration) use these formats.

## Boundary

Leto owns the `Array<T, S, N>` type family, `ArrayView`/`ArrayViewMut`,
the `Layout<N>` descriptor, storage backends, and all iterator types.
`leto-ops` owns the linear algebra kernels and decompositions.

RITK constructs `Array1`/`Array2` from diffusion data, assembles design
matrices, and calls `leto-ops` solvers through `.view()`. A RITK-local
array type, local linear solve, or local decomposition is a boundary
violation — the Leto array is the single source of truth for numeric data
in the RITK pipeline.

## References

- [Leto Linear Algebra Operations](leto_linalg.md) — the provider chapter
  documenting `solve_least_squares`, `nnls`, `cholesky_solve`, etc.
- [Coeus Tensor Integration](coeus_integration.md) — the Coeus `Tensor<T, B>`
  type that interops with Leto arrays through `coeus-leto`
- [Diffusion Models](ritk_diffusion.md) — the models that assemble Leto
  arrays and call leto-ops solvers
