# ADR 0035: Reject synthetic browser canvas input

Status: Accepted

Date: 2026-09-15

Driver: [RITK-SNAP-BROWSER-TRUST-2026-09-15](../backlog.md#RITK-SNAP-BROWSER-TRUST-2026-09-15)

Provider contract: [Metis ADR 0034](../../../metis/docs/adr/0034-browser-event-trust.md)

Upstream decision: [Moirai ADR 0060](../../../moirai/docs/adr/0060-browser-event-trust.md)

## Context

RITK's browser presenter translates the format-neutral Métis canvas contract
into viewer actions. Moirai now captures the browser `Event.isTrusted` value and
Metis preserves it on pointer, wheel and keyboard events. Without a consumer
policy, a script-created event could mutate the clinical viewer state exactly
like a user-mediated event.

## Decision

`WebCanvasPresenter::take_events` admits only trusted browser canvas events.
The adapter checks the event-level trust snapshot before translating any
pointer, wheel or keyboard value and drops a false snapshot as an expected
policy result. Trusted events retain the existing bounded translation,
coordinate normalization, key mapping and reducer behavior. Native events and
browser file-drop payloads use their existing contracts and are unaffected.

The drop occurs before RITK's action reducer, so synthetic input cannot change
slice, cine, gesture or tool state. The browser queue still drains and releases
pointer capture through Métis, preserving bounded lifecycle cleanup.

## Alternatives rejected

1. Accepting every canvas event discards the provider's provenance signal and
   permits synthetic state changes.
2. Rejecting false values inside Metis would encode a clinical application
   policy in a format-neutral GUI provider.
3. Re-reading `web-sys::Event.isTrusted` in RITK duplicates the DOM boundary
   and couples the viewer to browser ownership.

## Threat model and limits

The policy addresses script-dispatched DOM events marked untrusted. A true
browser value is not proof of a physical human, secure browser session,
operating-system permission or automation authenticity. The existing browser
trace validator remains an independent evidence path and still distinguishes
protocol trust from physical input. WebKit's selected-file authorization
residual is unrelated to this event policy and remains tracked separately.

DICOM scanning, decoding, geometry, pixel presentation and clinical semantics
remain RITK-owned; Metis and Moirai carry only bounded host values.

## Verification

The RITK translation policy has value-semantic trusted/false tests, and the
Metis event tests prove the provider preserves both values. Locked native and
WASM checks, strict Clippy and Rustdoc run against the merged provider pins.
Existing Chromium/Firefox real-study galleries remain the visual oracle; any
new hosted run records the exact Metis, Moirai and RITK revisions and retains
the Safari residual without fallback behavior.
