# ADR 0049: Free-threaded Python module contract

- Status: Accepted
- Date: 2026-09-21
- Item: [RITK-PYTHON-FREETHREADED-001](../../backlog.md#RITK-PYTHON-FREETHREADED-001)

## Decision

The RITK Python extension uses PyO3 0.29.2 with
`#[pymodule(gil_used = false)]`. The default `abi3` feature continues to
build the Python 3.9 stable-ABI wheel. The explicit `abi3t` feature enables
PyO3's `abi3t-py315` contract and the release workflow builds the Python
3.15t stable free-threaded wheel.

Binding methods detach from the interpreter while running Rust computation.
The Python contract tests concurrent value-semantic reads of one immutable
image from a thread pool on a free-threaded interpreter. The binding does not
add a zero-copy promise; `Image.to_numpy()` retains its existing ownership and
copy semantics.

## Alternatives

Keeping PyO3 0.22 cannot express `gil_used = false` or the `abi3t` feature,
so it cannot satisfy the contract. Declaring the feature without building and
testing a 3.15t wheel would be an unverified packaging claim.

## Verification

The Rust crate check and nextest suite cover the binding implementation. The
hosted Python matrix builds and installs the `abi3t` wheel with CPython 3.15t,
asserts the interpreter is free-threaded, and runs the concurrent image test.
