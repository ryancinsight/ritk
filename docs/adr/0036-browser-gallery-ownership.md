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

The workflow invokes `wasm-bindgen` with `--target web --no-typescript`,
producing the JavaScript and WebAssembly runtime pair that the Metis package
boundary accepts. The gallery mounts the Metis host before it queries or
customizes the format-neutral file controls, and reapplies its DICOM label and
filter after every mount because lifecycle teardown replaces those controls.

The consumer browser harness lives in `scripts/browser_gallery*.py`. It
requires an explicit Metis checkout, imports the generic host runner from that
checkout, and owns the RITK capture callbacks and evidence output.

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

The current ownership seams are executable and reviewable at these locations:

- [`gallery.js`](../../crates/ritk-snap/web/gallery/gallery.js) owns mount,
  picker customization, slice controls and RITK presentation attributes.
- [The browser workflow](../../.github/workflows/metis-browser-dicom.yml)
  supplies the package and gallery inputs and defines the saved-study matrix.
- [`browser_gallery.py`](../../scripts/browser_gallery.py) requires the Metis
  root and composes the generic host runner with RITK capture behavior.

The [loader tests](../../crates/ritk-snap/src/dicom/loader/tests/mod.rs),
[three-plane presentation tests](../../crates/ritk-snap/src/dicom/loader/tests/presentation.rs)
and [gallery harness tests](../../scripts/tests/test_browser_gallery.py) cover
their respective source contracts. A browser evidence run must bind its RITK
and provider revisions, public study hashes, exact raster oracles, bounded
rejections and teardown results. This ADR records ownership and required
evidence; it does not claim that a historical browser run establishes current
runtime support.

## Residuals

Browser-specific selected-file authorization, physical file-manager drag
input, provider-private resource measurements and WebGPU adapter availability
remain separate verification concerns. Their outcomes do not change ownership
of the DICOM consumer page or permit a Metis fallback decoder.
