# ADR 0056: The viewer crate returns to the workspace

- Status: Accepted

This decision is retroactive. It records the workspace return of
`crates/ritk-snap` delivered in [PR #844](https://github.com/ryancinsight/ritk/pull/844),
following the provider-cohort advance in
[metis PR #508](https://github.com/ryancinsight/metis/pull/508).

## Context

The Coeus-CPU migration (commit `45bcd5dac`) moved the workspace onto the
0.12 Coeus
provider cohort and could not resolve the viewer's metis provider
requirements at the time. `crates/ritk-snap` left the workspace members; the
root manifest recorded the exclusion with the condition that it "rejoins once
metis advances to the new provider cohort", and the metis-platform, metis-web
and metis-ui-lang workspace dependencies were removed with it.

The exclusion had two costs that the cohort advance has now removed.

First, the crate's 704 tests did not run in CI. The crate was outside the
workspace, so `cargo nextest run --workspace` never compiled it, and the
member's own suite reported green without ever building the viewer. The
multi-series cross-wiring work that landed in the same delivery
([PR #844](https://github.com/ryancinsight/ritk/pull/844)) added 22 files of
panel tests that no CI job had compiled.

Second, the exclusion left the crate on the pre-cohort provider API. The
workspace's `data_cow_on` became fallible with the Coeus 0.12 tensor
contract, and metis encapsulated the display list's command storage so a
caller outside the layout module cannot desynchronize the owner index.

## Decision

`crates/ritk-snap` rejoins the workspace members, and the metis provider
dependencies return to the root manifest at the cohort's version.

The port is native to the current APIs; no compatibility surface is added.

- **Fallible pixel access.** `Image::data_cow_on` returns
  `Result<Cow<'_, [f32]>>` because a non-contiguous buffer materializes
  through the fallible Coeus 0.12 contraction. The call sites propagate the
  error with `?`; the borrow stays a borrow when the buffer is contiguous,
  so the zero-copy path is unchanged.
- **An overlay accumulator owns the command vector.** The viewer builds its
  painter-ordered commands in `OverlayCommands`, a module-owned vector, and
  converts once with `DisplayList::from_commands` at the boundary the
  renderer reads. The overlay no longer reaches into the display list's
  storage, and merging the application and projection overlays extends one
  vector instead of copying a list.
- **Reads go through accessors.** `DisplayList::commands` is the only way to
  read commands, and the viewer's own tests read the converted display list
  rather than a test-only accessor on the accumulator.

The exclusion's condition is discharged: metis advanced its provider cohort,
so the viewer resolves against metis-platform, metis-web and metis-ui-lang
0.5.0.

## Rejected alternatives

- **Keep the viewer outside the workspace.** Rejected: the crate's tests
  cannot compile or run in CI, and the workspace's dependency alignment and
  conformance gates cannot see it. A crate that the gates cannot build is
  not covered by them.
- **Adapt the viewer to the old metis provider API.** Rejected: the cohort
  advance removed those provider versions; pinning them would restore the
  advisory exposure the cohort cleared.
- **Ask metis to expose the display list's command storage.** Rejected: the
  encapsulation is deliberate. A caller that pushes raw commands cannot keep
  the owner index consistent, and the host path for commands that belong to
  no element is `from_commands`.
- **A compatibility shim over the fallible pixel access.** Rejected: the
  shim would hide a real failure from a caller that cannot recover from it.

## Consequences

- The workspace lock grows by the viewer's dependency tree, including the
  optional desktop shell's windowing stack. `scripts/lockfile.py` regenerates
  the lock outside the overlay, and the committed lock resolves under
  `--locked` with 65 first-party git sources.
- The viewer's 704 tests run in the member's CI; the workspace suite is 6280
  tests.
- The workspace's dependency alignment gate sees the viewer: every one of
  its dependencies is inherited from the root manifest, and it declares no
  forbidden runtime stack.

## Evidence

- `cargo nextest run --workspace --lib --tests`: 6280 passed, 25 skipped, at
  the delivered revision; `cargo nextest run -p ritk-snap`: 704 passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
  `cargo fmt --all --check`, and `cargo doc --locked --workspace --no-deps`:
  clean.
- `python scripts/lockfile.py --check`: the committed lock resolves under
  `--locked` outside the overlay.
