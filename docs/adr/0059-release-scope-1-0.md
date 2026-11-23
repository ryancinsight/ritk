# ADR 0059: Release scope for 1.0

- Status: Accepted
- Item: RITK-RELEASE-LADDER

## Context

ADR 0057 scopes the viewer against RadiAnt; the board's ladder
(`M-viewer-reading` 0.2.0, `M-viewer-compare` 0.3.0, `M-viewer-access`
0.4.0, `M-viewer-advanced` 1.0.0) orders the reading, comparison, access,
and advanced-tooling work with runnable nextest exits. Thirty-five of eighty
open item files carry a `milestone:` tag; forty-five do not, and no record
states the 1.0 authority gate or the documentation condition.

## Decision

The 1.0 rung is `M-viewer-advanced`, completed with four conditions that
together are the M-1.0 exit: `cargo semver-checks` for `ritk-snap`, `cargo
publish --dry-run` for the viewer's closure, `cargo doc` with no new
warnings, and zero open correctness items — **under release authority**,
which sets the API-freeze contents. This milestone never forces `publish =
true` without that authority.

On 2026-11-22 the release owner selected four public surfaces for the 1.0
compatibility guarantee. AsterSlice's documented controls and gestures,
the RITK command line, the public Python API, and the public APIs of
published Rust crates enter the freeze at 1.0. Verify control behavior in
the native viewer, invoke the installed CLI and Python package as consumers,
and compare the Rust interfaces against published baselines. These checks
supplement the milestone exits above.

Every open item file carries exactly one `milestone:` tag: format
conversions, inventories, parser fuzzing, and rescue reconciliation read
studies in (`M-viewer-reading`); oblique, session, interaction, and browser
items compare and navigate (`M-viewer-compare`); open/archive/export/PACS
plus the CI gate and registry publication move studies and releases out
(`M-viewer-access`); filters, fusion, VR, dynamic, documents, movie, shell,
segmentation, and RT tools are the advanced surface (`M-viewer-advanced`).

Consumer-owned migrations (metis pane embedding, kwavers/CFDrs volume
consumers) do not gate 1.0; their receipts live on their own boards.

### Veto points

1. Changes to the four approved compatibility surfaces require release-owner review.
2. The encrypted-ZIP / optical-disc / unlisted-RadiAnt-add-on non-goals
   (recorded on `M-viewer-advanced`; only a chartered driver reopens them).

## Rejected alternatives

- A separate `M-1.0` rung duplicating `M-viewer-advanced`: two names for
  one exit would split the board's references; the 1.0 conditions attach
  to the existing rung instead.
- Freezing 1.0 while parser-fuzz items stay open: an unreadable study is a
  correctness failure in a viewer, so `M-viewer-reading` gates 1.0.

## Overturning evidence

A consumer failure traceable to a frozen API, or a fuzz crash in a
shipped parser, reopens the affected rung.
