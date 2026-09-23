# ADR 0049: Free-threaded Python module contract

- Status: Accepted
- Date: 2026-09-21
- Delivery: [RITK PR #574](https://github.com/ryancinsight/ritk/pull/574), merge `28e2acdf4ee8a937d55f452635a93c532df5487a`.

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

The Cargo feature set does not enable PyO3's `extension-module` feature. That
feature suppresses Python linking for extension builds and makes Rust test
binaries fail to link; maturin 1.9.4 or newer supplies the extension-module
build environment only for wheel builds. The CI workflow keeps `3.15t` as the
artifact identity while resolving setup-python with the `3.15` prerelease
range and `freethreaded: true`. The free-threaded contract job installs only
NumPy and pytest from `requirements-free-threaded.txt`; the full parity
dependency set remains on the regular Python matrix because native parity
packages do not necessarily publish prerelease free-threaded wheels.
Package metadata keeps NumPy `>=2.0.2,<2.6` for Python versions below 3.15
and requires `>=2.5,<2.6` from Python 3.15 onward, so installed abi3t wheels
receive the same C API floor as the contract job.

## Alternatives

Keeping PyO3 0.22 cannot express `gil_used = false` or the `abi3t` feature,
so it cannot satisfy the contract. Declaring the feature without building and
testing a 3.15t wheel would be an unverified packaging claim.

## Verification

The Rust crate check and nextest suite cover the binding implementation. The
hosted Python matrix builds and installs the `abi3t` wheel with CPython 3.15t,
asserts the interpreter is free-threaded, and runs the concurrent image test.

### Revision 2026-09-21

Merged-main runs 35645202445 and 35645201560 exposed the extension-module
test-link configuration and an unavailable exact 3.15t tool-cache lookup.
This revision records the build-feature and prerelease-resolution correction;
the hosted 3.15t wheel job remains the acceptance oracle.

### Revision 2026-09-21 (contract dependency split)

The first merged-main run after that correction reached the free-threaded
interpreter but failed before the wheel build because VTK had no compatible
CPython 3.15t distribution. The contract test imports only NumPy and pytest,
so the workflow now installs its minimal, explicitly scoped dependency file;
the regular matrix still installs VTK and SimpleITK for parity coverage.

### Revision 2026-09-21 (pinned toolchain components)

The follow-up run reached the wheel build but failed when Cargo honored the
repository toolchain declaration: the free-threaded job had installed 1.97.0
without its required `rustfmt` and `clippy` components. The job now requests
the same components as the pinned toolchain file before invoking maturin.

### Revision 2026-09-21 (explicit ABI feature selection)

The next hosted run built successfully but produced the default `cp39-abi3`
wheel because Cargo keeps default features enabled when `--features abi3t` is
passed alone. The free-threaded job now passes `--no-default-features` so the
artifact is built from the `abi3t` feature exclusively and is installable by
CPython 3.15t.

### Revision 2026-09-21 (NumPy bridge compatibility)

The first run that reached the installed wheel and free-threaded contract,
35661207418, failed when `Image` extracted a NumPy array: the registry
`numpy` 0.29.0 bridge reported `TypeError: 'ndarray' object is not an
instance of 'ndarray'` under abi3t. The fix is the merged rust-numpy abi3t
implementation and its extraction-error corrections at revision
[`9df4023`](https://github.com/PyO3/rust-numpy/commit/9df402373716a60f0e7825b6043568abe4e63b46),
consumed temporarily by the RITK lock. The free-threaded contract pins NumPy
to `>=2.5,<2.6`, matching that bridge's supported C API surface. Remove the
Git revision when a crates.io `numpy` release contains the abi3t support and
extraction fixes; the hosted 3.15t contract remains the acceptance oracle.

### Revision 2026-09-21 (accepted hosted contract)

PR [#574](https://github.com/ryancinsight/ritk/pull/574), merged as
`28e2acdf4ee8a937d55f452635a93c532df5487a`, passed the merged-main CI run
`35664515797` and Python CI run `35664515333`. The CPython 3.15t contract
installed the abi3t wheel, verified the interpreter remained free-threaded,
and passed the concurrent `Image` value-semantic test. The rust-numpy Git
revision and NumPy 2.5 floor remain quarantined until a crates.io release
contains the same abi3t bridge and extraction fixes.
