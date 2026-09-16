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

## Residuals

Safari/WebKit selected-file authorization, physical file-manager drag input,
provider-private browser resource measurements and WebGPU presentation remain
separate items. They do not change ownership of the DICOM consumer page or
permit a Metis fallback decoder.
