# AsterSlice executable identity

Status: Accepted

## Context

The viewer's visible product name is AsterSlice. The desktop binary still
uses the package name `ritk-snap`, and Windows normalizes its debug-symbol
basename to the library's `ritk_snap.pdb`. Live launch and installer commands
therefore need a coordinated executable migration.

## Decision

Name the single desktop binary `asterslice` and set its command-line name
explicitly. Use AsterSlice in installer display metadata and current workflow
labels. Keep the Cargo package, Rust/WebAssembly library, installer application
ID and upgrade code, persisted sessions, browser element IDs, and DICOM
network identities stable; these identify separate contracts.

Migrate every live binary caller and current manual command together.
Historical capture records keep the names of the artifacts they actually
observed. [The migration guide](../migration_selected_dicom.md#asterslice-executable)
gives the replacement invocation. Delivery: RITK PR #998.

## Alternatives and consequences

Keeping the old binary leaves product identity inconsistent and preserves
the debug-symbol collision. Renaming the Cargo package or library widens the
API and dependency migration without serving the executable contract.
A second executable retains an obsolete invocation and is not provided.

This is an intentional executable-name break, with no version bump or
release. CLI argument behavior and measurement ownership remain unchanged.

## Verification and overturning evidence

Metadata must expose one `asterslice` binary beside `ritk_snap` library
targets. Built help/version, native capture, both host configurations, and
package inventory must use the product name. All-target build output must
show distinct executable/library symbol outputs. A supported caller requiring
another executable contract or a persisting output collision warrants
revising this decision.
