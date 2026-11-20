# ADR 0035: Reject synthetic browser canvas input

Status: Accepted

Date: 2026-09-15

Revision 2026-11-20: replace the expired driver reference and historical
verification claims with durable delivery provenance and current source/link
evidence in [PR #1086](https://github.com/ryancinsight/ritk/pull/1086), without
adding runtime certification.

Delivery: [RITK PR 407](https://github.com/ryancinsight/ritk/pull/407)

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
protocol trust from physical input. Browser-specific selected-file
authorization is unrelated to this event policy.

DICOM scanning, decoding, geometry, pixel presentation and clinical semantics
remain RITK-owned; Metis and Moirai carry only bounded host values.

## Verification

The current policy lives in
[`browser_policy.rs`](../../crates/ritk-snap/src/presentation/browser_policy.rs),
whose value-semantic test admits `Trusted` and drops `Untrusted`.
[`WebCanvasPresenter`](../../crates/ritk-snap/src/presentation/web.rs) applies
that policy before translating a canvas event. Native and WebAssembly checks
must cover the consumer against the selected Metis and Moirai revisions.
Real-study browser galleries test integration behavior; they do not strengthen
the browser trust snapshot into evidence of physical input.
