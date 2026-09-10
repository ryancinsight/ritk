# ADR 0030: DICOM grayscale presentation semantics

Status: Accepted

Date: 2026-09-10

Driver: [RITK-SNAP-GRAYSCALE-001](../../backlog.md#RITK-SNAP-GRAYSCALE-001).

## Context

RITK IO already decodes signed pixel samples and applies the DICOM modality
rescale slope and intercept. The viewer's scalar path applied one historical
window equation regardless of the DICOM `VOI LUT Function`, and did not carry
`MONOCHROME1` inversion through slice, MIP, volume-rendering, fused, or GPU
paths. This made the displayed intensity dependent on the renderer selected
instead of the source object's presentation contract.

DICOM PS3.3 C.11.2 defines `LINEAR` as the default function with half-sample
thresholds, `LINEAR_EXACT` with exact centre and width boundaries, and
`SIGMOID` as a logistic mapping. The table-based VOI LUT Sequence is a
separate form and needs a real lookup-table implementation before it can be
admitted.

## Decision

`render::grayscale` is the sole owner of scalar presentation semantics. It
resolves `MONOCHROME1`/`MONOCHROME2`, admits the three defined VOI functions,
rejects unsupported functions and table-based VOI LUTs at the scalar load
boundary, and applies inversion exactly once after the selected function. A
missing function uses DICOM's `LINEAR` default. Inconsistent explicit
functions across a series are rejected rather than choosing one silently.

All CPU scalar paths call this presentation value: orthogonal slices, MIP,
volume rendering, and fused comparison. Native GPU MIP and volume rendering
receive the same function and inversion as a compact uniform bitfield and use
the corresponding shader equations. The application selects finite metadata
window centre and width when available, retaining the hanging protocol only as
the explicit fallback for non-DICOM or incomplete metadata.

Métis remains format-neutral. DICOM parsing, metadata preservation, modality
rescale, and this presentation contract belong to RITK so the future Métis
viewer consumes one validated volume surface.

## Alternatives

Keeping the historical equation for every object is rejected because it
misrepresents the DICOM default and cannot distinguish `LINEAR_EXACT` or
`SIGMOID`. Inverting in each renderer is rejected because a missed path or a
second inversion produces a clinically different image; one resolved value
keeps the operation at one owner. Approximating a VOI LUT Sequence with a
window is rejected because a table is data, not a window and requires a
separate bounded lookup implementation. Moving DICOM parsing into Métis is
rejected because it would duplicate the RITK trust boundary and make every
format-aware consumer own incompatible semantics.

## Verification

The presentation unit suite checks the DICOM C.11.2 boundary values for
`LINEAR`, `LINEAR_EXACT`, width-one windows, and `SIGMOID` monotonicity and
symmetry. Loader tests use signed stored samples with slope `2` and intercept
`-10`, preserve `MONOCHROME1` and `LINEAR_EXACT` through both file and byte
entry points, assert the inverted rendered bytes, and reject an unknown
function. Existing RGB tests remain green because color bypasses scalar
presentation. The deterministic `dicom_workflow` example emits a grayscale
capture from the same path as the viewer, and `scripts/viewer.py` compares its
enlarged pixels with the reviewed manual image.

The equations and boundary rules follow [DICOM PS3.3 C.11.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html).
