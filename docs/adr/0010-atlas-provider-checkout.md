# ADR 0010 — Atlas-owned provider checkout

- Status: Accepted
- Change class: [arch] [patch]
- Date: 2026-07-20
- Related: CI-664-01, [Atlas ADR 0027](https://github.com/ryancinsight/atlas/blob/9a651ff539e314ff26c4a5b69fe89448c1770859/docs/adr/0027-provider-checkout-ssot.md)

## Context

RITK's hosted workflows require sibling path dependencies before Cargo can
resolve the workspace. A local composite action repeated eleven provider
repository URLs and revisions. That list duplicated the Atlas `.gitmodules`
URLs and gitlinks, so a provider advance required synchronized edits in two
repositories and could leave local and hosted graphs different.

The RITK root manifest already declares the dependency closure. Atlas merge
`9a651ff539e314ff26c4a5b69fe89448c1770859` provides a composite action that
parses those Cargo path declarations and materializes their providers at the
gitlinks recorded by the same immutable Atlas commit.

## Decision

Delete the RITK-owned provider checkout action. Every Rust, Python, migration
audit, and wheel-release workflow invokes the Atlas action at
`9a651ff539e314ff26c4a5b69fe89448c1770859` with:

- manifest `ritk/Cargo.toml`;
- provider destination `.` relative to the hosted workspace; and
- the same exact Atlas commit as `atlas_ref`.

The action reference and graph reference remain identical. Provider names,
URLs, and individual revisions have no RITK-owned representation.

## Rejected alternatives

- Retaining the static action preserves the duplicate graph and its drift.
- Wrapping the Atlas action in another local action adds an owner and a
  forwarding layer without changing the contract.
- Following Atlas `main` makes otherwise identical workflow runs resolve
  different provider graphs over time.
- Initializing every Atlas submodule downloads repositories outside RITK's
  Cargo dependency closure.

## Consequences

Advancing RITK's provider graph is one explicit Atlas SHA update across the
eight workflow call sites. GitHub Actions requires `uses` references to be
literal, so those call sites repeat the immutable graph identifier; no
provider-specific data is repeated. The release workflow changes only its
build input reconstruction and does not authorize a release.

## Verification

Local structural checks require exactly eight Atlas action references and
eight matching `atlas_ref` values, reject any residual local-action reference
or provider repository list, and parse every workflow as YAML. Exact-head
hosted PR CI remains the integration oracle for Linux, macOS, Windows, Python,
and migration-audit paths. The tag-only release workflow is syntax-checked but
is not executed because this change does not authorize a release.
