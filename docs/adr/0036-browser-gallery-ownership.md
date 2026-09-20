# ADR 0036: RITK browser gallery ownership

Status: Accepted

Date: 2026-09-16

Driver: [RITK-BROWSER-GALLERY-001](../backlog.md#RITK-BROWSER-GALLERY-001)

Upstream contract: [Metis ADR 0035](../../../metis/docs/adr/0035-dicom-consumer-ownership.md)

## Context

RITK owns DICOM discovery, decoding, geometry, clinical presentation and the
real-study image oracles. Metis owns the format-neutral browser host, bounded
file handoff and canvas lifecycle. The historical gallery page lived in Metis
only because its packager copied the page beside the generated RITK module.
That location made DICOM labels, canvas identifiers and slice controls appear
to be framework behavior.

## Decision

The DICOM consumer page is stored at
`crates/ritk-snap/web/gallery`. It contains the page, stylesheet and module
that import the generated RITK WebAssembly package, configure the DICOM chooser
after Metis mounts, start and stop the three RITK canvases, and synchronize the
slice controls with RITK's `data-ritk-*` attributes.

The RITK browser workflow supplies this directory together with the generated
package through Metis's explicit `--consumer-gallery` and
`--consumer-package` arguments. Metis validates and copies the bounded assets;
it does not classify files or interpret viewer state. The workflow remains the
single source for saved-study commands, real image captures and pixel oracles.

## Alternatives rejected

Keeping the page in Metis was rejected because a framework repository would
continue to own a consumer format and its clinical controls. Generating a page
from the RITK package was rejected because it hides the consumer dependency and
weakens review of the HTML/CSS policy. Moving the browser host or byte provider
into RITK was rejected because it duplicates Metis's first-party transport and
DOM boundary.

## Failure modes and security boundary

The workflow fails before browser execution when either consumer input is
missing, malformed or outside Metis's size and CSP bounds. A page that broadens
the canonical same-origin policy is rejected. RITK surfaces read, scan or
decode errors without publishing an incomplete study or synthetic pixels.
The gallery accepts only user-activated chooser events and the existing
bounded host handoff; it does not turn browser names into filesystem paths.

## Verification

RITK unit tests cover the DICOM loader, three-plane presentation and the page's
slice and lifecycle contract. The locked browser workflow builds the RITK
WASM package, passes the gallery to Metis, selects the saved 94-file MRI study
and checks file hashes, three exact RGBA canvases, bounded rejections, trusted
input and listener teardown on each configured engine. The committed manual
images and provenance remain the visual evidence.

Revision 2026-09-16 (runtime package boundary): the RITK workflow invokes
`wasm-bindgen --target web --no-typescript`, so the explicit consumer package
contains exactly one JavaScript module and one WebAssembly module. Metis can
validate that runtime pair without copying declaration sidecars; the consumer
page and DICOM behavior are unchanged.

Revision 2026-09-16 (consumer mount ordering): the gallery mounts the Metis host
before querying its format-neutral file-picker controls. This keeps consumer
customization deterministic when module initialization and host DOM insertion
are scheduled differently across browser engines.

Revision 2026-09-16 (remount picker policy): each gallery mount reapplies the
consumer label and DICOM filter after the host inserts its controls. Lifecycle
teardown replaces those controls, so a one-time customization would silently
restore the generic picker on the next cycle.

Revision 2026-09-16 (harness ownership): the slice-control browser harness now
lives in `scripts/browser_gallery*.py` beside the RITK consumer tests. Its
wrapper imports the generic Metis runner through an explicit `--metis-root`,
passes RITK's post-transfer capture callback, and records consumer evidence
under the RITK output contract. The Metis scripts no longer contain RITK canvas
IDs, DICOM filters or slice-control behavior.
The dependent RITK lock and browser/package workflow defaults pin Metis merge
`88c60a0b6410c0e07700e965bcdbea43b7b20789`; hosted evidence has been regenerated
from this wrapper.

Revision 2026-09-16 (consumer evidence): hosted run
`35133971196` rebuilt this wrapper at RITK merge
`db390b8616e4cc58f2555f49a816e61eed1fadd1` against the pinned Metis revision.
The Chromium and Firefox raster jobs exercise the saved 94-file MRI study;
the WebKit job remains the selected-file authorization residual. The Chromium
WebGPU job reaches the RITK page but reports no browser WebGPU adapter, so it
does not claim a rendered GPU study. These outcomes preserve the ownership
boundary: RITK owns DICOM behavior and Metis remains format-neutral.

Revision 2026-09-20 (merged-main replay): hosted run
[`35519330780`](https://github.com/ryancinsight/ritk/actions/runs/35519330780)
rebuilt RITK `b220bf4ec2bf613fb23f2da20152881a1a3b4cf2` against Metis
`b58d64b1bebe76bb32570339c4a349cc1b0d7086` and Moirai
`2a54e010532f76c88027fec8a468620c92fe66b3`. Chromium and Firefox passed the
94-file MRI chooser replay and exact raster oracles; Chromium also passed the
application-window and scalar MIP projection captures. Safari 26.6.2 accepted
the chooser but rejected the bounded whole-file read, and Chromium reported no
WebGPU adapter. The machine-readable provenance records each artifact and
failure at this revision; neither residual changes the DICOM ownership boundary.

## Residuals

Safari/WebKit selected-file authorization, physical file-manager drag input,
provider-private browser resource measurements and WebGPU presentation remain
separate items. They do not change ownership of the DICOM consumer page or
permit a Metis fallback decoder.
