# Moirai Parallel Execution Backend

Moirai is the Atlas-native hybrid concurrency library that provides the parallel
execution backend for batched voxel-wise fitting in the diffusion MRI pipeline.
It blends synchronous data-parallel operators (the Rayon-replacement surface)
with a lock-free work-stealing scheduler, a unified hybrid runtime that serves
both async and parallel work, and compile-time execution policies that
monomorphize to zero-cost concrete paths.

At the RITK level, Moirai is surfaced through the `MoiraiBackend` type in
`ritk-image` and the broader Coeus backend traits. Compute-heavy diffusion
kernels — DTI log-linear solves, DKI Levenberg-Marquardt iterations, CSD
non-negative deconvolution — distribute across CPU cores through Moirai's
data-parallel operators. The important design point is that Moirai does not
require a separate image type or parallel-only API: readers, writers, filters,
and registration code are parameterized over a backend, so the same
`Image<f32, B, 3>` contract can execute on the sequential backend for
deterministic single-threaded work or on Moirai for throughput.

## Architecture

Moirai has three layers, each a separate domain crate:

| Crate | Role |
|---|---|
| `moirai-scheduler` | Lock-free work-stealing deques (Chase-Lev, SplitDeque), NUMA topology discovery, adaptive backoff |
| `moirai-executor` | The unified hybrid runtime that schedules both `SyncTask` (parallel) and async work on one pool |
| `moirai-parallel` | The data-parallel operator surface — extension traits, free functions, compile-time execution policies |

A `.par()` worker task runs on the *same unified hybrid scheduler* as async
work, not a separate thread pool. The sync return shape is a property of the
*operation*, not an isolation boundary: a parallel kernel can spawn or drive
async work on the same runtime, and sync operators compose cleanly inside pure
compute kernels without async contagion.

The scheduler uses **lock-free work-stealing**: each worker owns a private
Chase-Lev deque and a `SplitDeque` that keeps the freshest work on a local stack
(zero atomics) and spills only the oldest half to a shared deque for thieves.
Batch stealing transfers up to 16 items at a time, reducing steal contention
when spawn rate greatly exceeds steal rate.

## Execution Policies

Every data-parallel operator is generic over an `ExecutionPolicy` — a zero-sized
type-level marker that decides *whether* to parallelize. Because the decision is
an associated function, generic code over `P: ExecutionPolicy` monomorphizes to
one concrete path with no value passed and no dynamic dispatch.

| Policy | Behaviour | Use case |
|---|---|---|
| `Sequential` | `parallelize(_) → false` — always single-threaded | Deterministic runs, nested parallel regions, debugging |
| `Parallel` | `parallelize(_) → true` — always multi-threaded | Force parallelism below threshold |
| `Adaptive` | `parallelize(n) → n >= 1024` — auto-routes | **Default** for `.par()` and `par_*` helpers |
| `AdaptiveWithThreshold<N>` | `parallelize(n) → n >= N` — custom threshold | Tuning for specific problem sizes |

The `Adaptive` policy is the everyday entry point. It parallelizes only at or
above 1,024 elements, where dispatch and join overhead is amortized; below that
threshold the sequential path eliminates all scheduling cost. Because `Adaptive`
is itself a zero-sized type, `.par()` is a fully monomorphized, zero-cost
abstraction — the parallel/sequential decision is a cheap inlined runtime check.

```rust,ignore
use moirai_parallel::{Adaptive, Parallel, Sequential, ExecutionPolicy};

// Force sequential (deterministic / nested)
for_each_with::<Sequential, _, _>(&data, |x| process(x));

// Force parallel (ignore threshold)
for_each_with::<Parallel, _, _>(&data, |x| process(x));

// Adaptive default (everyday use)
data.par().for_each(|x| process(x));
```

## Extension Traits

The primary surface is two extension traits on slices:

### `ParallelSlice<T>`

```rust,ignore
pub trait ParallelSlice<T> {
    /// Adaptive, auto-routing parallel view over `&[T]`.
    fn par(&self) -> ParRef<'_, T, Adaptive>;
}
```

`ParRef` provides read-only parallel combinators:

| Method | Equivalent sequential | Returns |
|---|---|---|
| `par().for_each(f)` | `iter().for_each(f)` | `()` |
| `par().enumerate(f)` | `iter().enumerate().for_each(f)` | `()` |
| `par().map_collect(f)` | `iter().map(f).collect()` | `Vec<R>` |
| `par().map_collect_index(f)` | `iter().enumerate().map(f).collect()` | `Vec<R>` |
| `par().map_reduce(id, map, reduce)` | `iter().map(map).fold(id, reduce)` | `R` |

### `ParallelSliceMut<T>`

```rust,ignore
pub trait ParallelSliceMut<T> {
    /// Adaptive, auto-routing mutable parallel view over `&mut [T]`.
    fn par_mut(&mut self) -> ParMut<'_, T, Adaptive>;
}
```

`ParMut` provides mutable in-place combinators:

| Method | Equivalent sequential |
|---|---|
| `par_mut().for_each(f)` | `iter_mut().for_each(f)` |
| `par_mut().enumerate(f)` | `iter_mut().enumerate().for_each(f)` |

```rust,ignore
use moirai_parallel::{ParallelSlice, ParallelSliceMut};

let v: Vec<u64> = (0..1_000_000).collect();

// Map-reduce over a read-only slice
let sum = v.par().map_reduce(0u64, |&x| x, |a, b| a + b);

// In-place mutation of every element
let mut doubled = v.clone();
doubled.par_mut().for_each(|x| *x *= 2);
```

## Data-Parallel Free Functions

For rare cases that need to force a specific policy via turbofish, every
operator is also available as a `*_with::<P>` free function. These are the
low-level override and are what the extension traits delegate to.

### Read-only operators

| Function | Signature |
|---|---|
| `for_each_with::<P>(data, f)` | `(&[T], Fn(&T))` |
| `enumerate_with::<P>(data, f)` | `(&[T], Fn(usize, &T))` |
| `map_collect_with::<P>(data, f)` | `(&[T], Fn(&T) -> R) -> Vec<R>` |
| `map_reduce_with::<P>(data, identity, map, reduce)` | `(&[T], R, Fn(&T) -> R, Fn(R, R) -> R) -> R` |

### Mutable operators

| Function | Signature |
|---|---|
| `for_each_mut_with::<P>(data, f)` | `(&mut [T], Fn(&mut T))` |
| `enumerate_mut_with::<P>(data, f)` | `(&mut [T], Fn(usize, &mut T))` |
| `map_collect_mut_with::<P>(data, f)` | `(&mut [T], Fn(usize, &mut T) -> R) -> Vec<R>` |

### Index-domain operators

| Function | Signature |
|---|---|
| `for_each_index_with::<P>(len, f)` | `(usize, Fn(usize))` |
| `map_collect_index_with::<P>(len, f)` | `(usize, Fn(usize) -> R) -> Vec<R>` |
| `reduce_index_with::<P>(len, identity, map, reduce)` | `(usize, R, Fn(usize) -> R, Fn(R, R) -> R) -> R` |
| `fold_reduce_with::<P>(len, init, fold, reduce)` | `(usize, Fn() -> A, Fn(A, usize) -> A, Fn(A, A) -> A) -> A` |

```rust,ignore
use moirai_parallel::{for_each_with, map_collect_with, map_reduce_with, Sequential, Parallel};

// Force sequential: deterministic, no scheduling overhead
for_each_with::<Sequential, _, _>(&small_data, |x| debug_assert!(x > 0));

// Force parallel: even for small inputs (benchmarking, nested regions)
let results: Vec<_> = map_collect_with::<Parallel, _, _, _>(&data, expensive);

// Elementwise product across multiple slices (index-aligned)
let dot = reduce_index_with::<Adaptive>(
    n, 0.0_f64,
    |i| a[i] * b[i],
    |x, y| x + y,
);
```

## Chunk Operators

For batched workloads — like per-voxel model fitting where each "chunk" is a
contiguous slice of the voxel array — chunk operators amortize dispatch overhead
by assigning each worker a range of consecutive elements. The diffusion pipeline
typically uses `for_each_chunk_mut_with_state` to give each worker a reusable
scratch buffer (design matrix, workspace arrays) across its assigned voxels.

| Operator | Description |
|---|---|
| `for_each_chunk_mut_with::<P>(data, chunk_size, f)` | Apply `f(&mut [T])` to each chunk |
| `for_each_chunk_mut_enumerated_with::<P>(data, chunk_size, f)` | Same, with chunk index |
| `for_each_chunk_mut_with_state::<P>(data, chunk_size, init, f)` | Per-worker scratch buffer `S` |
| `for_each_chunk_pair_mut_enumerated_with::<P>(a, b, size, f)` | Two-buffer fused kernel |
| `for_each_chunk_triple_mut_enumerated_with::<P>(a, b, c, size, f)` | Three-buffer fused kernel |
| `for_each_chunk_quad_mut_enumerated_with::<P>(a, b, c, d, size, f)` | Four-buffer fused kernel |

```rust,ignore
use moirai_parallel::for_each_chunk_mut_with_state;

// Per-voxel DTI fitting: each worker gets a private design matrix
for_each_chunk_mut_with_state::<Adaptive, _, _, _, _>(
    &mut voxel_results,
    256,                             // chunk size (voxels per task)
    || VoxelWorkspace::new(),         // init per-worker scratch
    |ws, chunk| {
        for result in chunk {
            *result = fit_dti(ws, &signal[result.index]);
        }
    },
);
```

The multi-buffer variants (`pair`, `triple`, `quad`) are fused kernels that
update several output arrays in one authoritative pass — for example, writing FA,
MD, and PEV fields simultaneously — without allocating intermediate tuples.

## Borrowing Scope

`scope` provides a Rayon-style region for spawning parallel sub-tasks that
capture non-`'static` references. It blocks until every spawned job has
completed, so borrowed data cannot escape.

```rust,ignore
use moirai_parallel::scope;

let data: Vec<u64> = (0..1000).collect();
let sum = std::sync::atomic::AtomicU64::new(0);

scope(|s| {
    s.spawn(|| sum.fetch_add(data.iter().sum::<u64>(), Ordering::Relaxed));
    s.spawn(|| sum.fetch_add(data.len() as u64, Ordering::Relaxed));
});
// Both tasks guaranteed complete; `data` is still alive.
```

## Join

`join` runs two closures concurrently. The left closure is scheduled on a worker
thread; the right closure runs on the caller. If the scheduler refuses the left
closure (shutting down or admission queue full), it runs on the caller instead,
so `join` gracefully degrades to sequential but never drops work.

```rust,ignore
use moirai_parallel::join;

let (a, b) = join(
    || expensive_first_half(),
    || expensive_second_half(),
);
```

For explicit policy selection, use `join_with::<P>`.

## Work-Stealing Deques

The scheduler's lock-free work-stealing deques live in `moirai-scheduler`:

| Type | Description |
|---|---|
| `ChaseLevDeque<T>` | Canonical Chase-Lev deque: O(1) wait-free local `push`/`pop` for the owner, lock-free `steal` for thieves |
| `ChaseLevStealer<T>` | Cloneable top-side endpoint created by `deque.stealer()` |
| `SplitDeque<T>` | Private owner stack backed by a shared deque — reduces steal contention |
| `StolenBatch<T>` | Allocation-free, panic-safe iterator over up to 16 batch-stolen items |
| `StealResult<T>` | `Success(T)` / `Empty` / `Retry` (distinct outcomes) |

The split-deque design is the key performance optimisation for the diffusion
pipeline: an owner's `push`/`pop` against its private stack costs zero atomics,
and only work old enough to be worth stealing reaches the shared deque. A thief
that finds the victim empty returns `StealResult::Empty` (look elsewhere)
rather than spinning; a thief that loses a race returns `StealResult::Retry`
(the victim may still hold work, so retrying the same deque is worthwhile).

Batch stealing (`steal_batch`) transfers half the available items (up to 16) in
one atomic CAS, cutting steal contention by an order of magnitude when many
small tasks are spawned rapidly — the diffusion pipeline's voxel-chunk dispatch
pattern.

Memory ordering follows the weak-memory-correct Chase-Lev formulation of Lê,
Pop, Cohen & Nardelli (PPoPP 2013), with a Morrison-Afek fence-free fast path
for `pop` on x86/x86_64 when the deque holds enough items that no steal can
reach the popped slot. Old buffers freed by resize are retired to an
epoch-reclamation list and freed only when no accessor is in-flight.

## NUMA Awareness

On multi-socket systems, the scheduler discovers hardware topology via
`CpuTopology` and uses `AdaptiveBackoff` (spin → yield → sleep) for
NUMA-aware victim selection. A thief prefers victims on the same NUMA node to
reduce cross-socket memory latency, and the backoff strategy ramps from a brief
spin (good when work is about to appear) to a park (good when the system is
fully idle).

## Diffusion Pipeline Integration

Moirai is the execution substrate for every compute-heavy pass in the diffusion
MRI pipeline. Each model's fitting kernel dispatches across voxels through
Moirai's chunk operators:

| Model | Moirai operator | What's parallelised |
|---|---|---|
| DTI | `for_each_chunk_mut_with_state` | Per-voxel design-matrix assembly + `leto_ops::solve_least_squares` |
| DKI | `for_each_chunk_mut_with_state` | Per-voxel LM iterations with per-worker Jacobian workspace |
| ODF | `for_each_chunk_mut_with_state` | Per-voxel SH expansion + Laplace-Beltrami regularised solve |
| CSD | `for_each_chunk_mut_with_state` | Per-voxel response convolution + `leto_ops::nnls` |
| NODDI | `for_each_chunk_mut_with_state` | Per-voxel 3-compartment LM with Watson-stick quadrature |

The pattern is the same in every case:

1. A flat `Vec<ModelResult>` is allocated with one element per voxel (all
   elements `MaybeUninit`).
2. `for_each_chunk_mut_with_state` parcels the result array into chunks and
   assigns each to a worker.
3. Each worker initialises a scratch workspace once (`init` closure), then
   processes its assigned voxels sequentially within the chunk, reusing the
   workspace.
4. On completion, every slot is initialised and the `Vec<MaybeUninit<T>>` is
   safely reinterpreted as `Vec<T>`.

This is zero-copy (no intermediate per-voxel allocations), cache-friendly
(consecutive voxels share the same signal array and scheme), and NUMA-aware (the
scheduler prefers to assign chunks to workers on the same socket as the memory).

```rust,ignore
// Pattern for batched voxel-wise fitting (simplified)
fn fit_volume(
    signal: &[f64],
    scheme: &GradientScheme,
    config: &DtiConfig,
) -> Vec<DtiResult> {
    let n_voxels = signal.len() / scheme.len();
    let mut results: Vec<MaybeUninit<DtiResult>> = Vec::with_capacity(n_voxels);
    unsafe { results.set_len(n_voxels); }

    for_each_chunk_mut_with_state::<Adaptive, _, _, _, _>(
        &mut results,
        256,                           // voxels per task
        || DtiWorkspace::new(scheme),  // one design matrix per worker
        |ws, chunk| {
            for (i, slot) in chunk.iter_mut().enumerate() {
                let voxel_start = (ws.chunk_base + i) * scheme.len();
                slot.write(ws.fit(&signal[voxel_start..]));
            }
        },
    );

    // SAFETY: every slot written by the parallel region above.
    unsafe { std::mem::transmute(results) }
}
```

## Performance Characteristics

| Metric | Value |
|---|---|
| Task scheduling overhead | < 1 µs per task |
| Push/pop (owner) | O(1) wait-free; zero atomics for SplitDeque local stack |
| Steal (thief) | O(1) lock-free; batch steal up to 16 items per CAS |
| Memory efficiency | Zero-copy task passing, `DisjointMutPtr` sub-slice handouts |
| Scalability | Linear scaling up to CPU core count |
| False sharing | `CacheAligned` wrappers on deque `bottom`/`top` indices |
| NUMA | Topology-aware victim selection, adaptive backoff |

