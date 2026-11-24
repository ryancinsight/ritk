# ADR 0036: RITK browser gallery ownership

Status: Accepted

Date: 2026-09-16

Revision 2026-11-20: replace the expired driver reference and consolidate
append-only delivery history into current source/link evidence in
[PR #1086](https://github.com/ryancinsight/ritk/pull/1086), without adding
runtime certification.

Delivery: [RITK PR 414](https://github.com/ryancinsight/ritk/pull/414)

Upstream contract: [Metis ADR 0035](../../../metis/docs/adr/0035-dicom-consumer-ownership.md)

## Context

RITK owns DICOM discovery, decoding, geometry, clinical presentation and the
real-study image oracles. Metis owns the format-neutral browser host, bounded
file handoff and canvas lifecycle. The historical gallery page lived in Metis
only because its packager copied the page beside the generated RITK module.
That location made DICOM labels, canvas identifiers and slice controls appear
to be framework behavior.

## Decision

The standalone browser host in `crates/ritk-snap/web/workstation` runs
`NativeViewerSession` in WebAssembly. Windows and browsers submit client-coordinate
events to one reducer and display its composed framebuffer. This makes menu hit
regions, clinical overlays, series selection, measurement and pane geometry part
of the same implementation on both targets.

Métis transfers bounded file selections. The browser groups validated Part 10
instances by series UID and retains the encoded sources for subsequent panel
assignments. Discovery builds the same patient/study tree as the Windows path.
The browser picker requires a trusted input event. Windows owns filesystem
export and asynchronous disk reads. Decode errors retain the displayed study.

The RITK packager validates the page and generated modules against the Métis
content policy and serves the workstation at the root and the previous URL.
Embedding clients can continue to mount individual image canvases.

## Alternatives

The previous gallery had its own HTML controls, geometry and shortcuts. Matching
its colors and dimensions left two independent implementations of reader actions.
Sharing the workstation also shares later fixes across hosts.

## Verification

`scripts/browser_workspace.py` opens the public MRI-DIR study, compares displayed
samples with independently presented reference planes and checks all three
original RGBA hashes. Trusted browser input exercises the native toolbar,
rotation, cine, measurements and multi-series dialog. The local content-box
regression uses a temporary fixture for the published embedding API.
