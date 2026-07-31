# ADR 0016 — Type-erased Coeus module-error bridge

- Status: Accepted
- Date: 2026-07-30
- Refs: atlas `backlog.md#atlas-coeus-main-sync-1`

## Context

Coeus made `Module::forward` and the autograd backward pass fallible
(`Result<_, ModuleError<B::Error>>`, coeus `81eeec09`/`5e64ee75`). RITK's
model layer reports failures through the non-generic `ModelError`, and its
graph code is generic over the backend `B`, so `ModuleError<B::Error>`
cannot be stored in `ModelError` without genericizing the enum over the
backend error type — a viral signature change through every model,
registration, CLI, and Python consumer.

## Decision

`ModelError` gains one variant:

```rust
#[error(transparent)]
Module(Box<dyn std::error::Error + Send + Sync>),
```

with a generic `impl<E> From<ModuleError<E>> for ModelError`, so call sites
propagate with plain `?`. Boxing erases only the backend type parameter;
the boxed value remains the typed `ModuleError` with its `#[source]` chain
intact, and `#[error(transparent)]` preserves its rendered contract
(module family, rank/axis details, backend source).

## Alternatives

- `ModelError<E>` genericization: fully typed, rejected for the viral
  signature change across all consumers with no present matcher needing
  the concrete backend error type downstream of the model layer.
- Stringly `Module { message: String }`: rejected — collapses failure
  modes and drops the source chain (error-handling restraint).

Trait-`Module` impls whose contract guarantees success on validated input
keep their pre-existing `expect("invariant: …")` form; inherent forwards
propagate.
