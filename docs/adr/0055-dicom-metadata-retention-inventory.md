# ADR 0055: DICOM metadata retention inventory

- Status: Accepted

This decision records the source-owned DICOM metadata inventory implemented in
`crates/ritk-io/src/format/dicom/`. It completes the accounting requirement of
[architecture §24](../../docs/architecture.md) and supplies the source-owned
input that the shared conversion preflight in
[ADR 0054](0054-stored-volume-contract.md) consumes.

## Context

The DICOM series reader interprets a small set of tags semantically and retains
the rest so a read/write cycle does not lose them. Retention was expressed as
two outcomes: a typed node (`DicomObjectModel`) or an opaque byte record
(`DicomPreservedElement`). Several paths produced neither.

The preservation loop skipped an element when the value could not be converted,
and the recursive sequence walker returned an empty item when the value exposed
no items or when recursion passed its depth guard. In each case the parser had
already parsed the element and then discarded it with no record, so a
conversion preflight could not distinguish "the source never had this field"
from "RITK dropped it". ADR 0054 requires scoped metadata losses to be reported
and rejected before output changes; a silent drop defeats that boundary because
there is nothing to report.

## Decision

`ritk-io::format::dicom` owns a source-owned retention inventory. Every parsed
element has exactly one of three dispositions:

1. **Interpreted** into a named field.
2. **Retained opaquely** as a `DicomObjectNode` or `DicomPreservedElement`.
3. **Recorded** as a `DicomRetentionLoss { tag, reason }`.

There is no fourth disposition. The retention loop and the recursive sequence
walker never discard a parsed element without a record.

`DicomRetentionReason` is `#[non_exhaustive]` and covers the three cases where
retention is impossible:

- `SequenceItemsUnavailable` — a sequence-typed element exposed no items and
  could not be re-encoded as raw bytes. The walker first attempts opaque byte
  retention, so this reason is recorded only when that also fails.
- `ValueBytesUnavailable` — the value could not be re-encoded to raw bytes, so
  neither a typed node nor an opaque record could be produced.
- `NestingDepthExceeded` — recursion reached `MAX_RETAINED_SEQUENCE_DEPTH` (8).
  The sequence element whose child would exceed the limit is recorded, so the
  truncated subtree is visible instead of absent.

The recursion bound is unchanged from the previous guard: levels 0 through 8
are walked. Two things change at the boundary. The boundary element is now
recorded as a loss; previously the walker returned an empty item, which the
parent inserted as an *empty sequence* — a node that asserted the sequence had
no items. The boundary element is therefore no longer retained as a node: it
is a recorded loss with no `DicomObjectNode`, which is why the loss record is
the only evidence the element existed.

`DicomPreservationSet::losses` carries the records. The inventory is
source-owned: it is produced by the reader and never derived from a
destination.

Losses are recorded **per slice**, on `DicomSliceMetadata::preservation`,
because a dropped element belongs to one instance and only the slice can scope
it to an exact frame. The series-level `DicomReadMetadata::preservation`
carries series-scope losses and is empty today: every element the reader
records is recorded while parsing one instance. A preflight must therefore
read the slices, not the series set alone — `dicom_series_metadata_losses`
does exactly that and is the series-level entry point.

`dicom_metadata_losses(preservation, location)` is the single projection from
the inventory into the shared `FormatMetadataLoss` vocabulary, so
`prepare_conversion` consumes one representation. Each record becomes
`FormatMetadataLoss::UnknownSemantics` scoped to the caller's
`ConversionLocation`, because an adapter that could not read a value cannot
claim to know its semantics. The projection lives in
`crates/ritk-io/src/format/dicom/inventory.rs`, not in the object model, so
`DicomPreservationSet` stays a pure data model with no conversion dependency.

## Rejected alternatives

- **Report losses as `UnsupportedByTarget`.** The target is not known when the
  source is parsed, and the failure is in reading, not in writing.
- **Carry losses inside `DicomPreservedElement` with empty bytes.** An element
  that was retained and one that was dropped would share a type and a
  zero-length payload would carry meaning. A distinct record keeps the two
  outcomes distinguishable in the type system.
- **Raise the nesting limit instead of recording truncation.** The limit guards
  against malformed input; raising it changes the failure mode rather than
  accounting for it.
- **Return `Result` from the sequence walker.** One unreadable element does not
  invalidate the series, and a `Result` would force an all-or-nothing decision
  where the inventory can report each element separately.
- **Keep the drop paths and document them.** A silent drop is invisible to
  every downstream check; documenting it does not let a preflight reject the
  source.

## Consequences

`DicomPreservationSet::is_empty` now includes the loss list, so a set with only
losses is not empty. Existing consumers that read `object` and `preserved`
are unaffected; `writer::emit_preservation_nodes` still emits exactly those two
collections, because a loss is by definition not retained.

`ritk-io` gains a dependency on `ritk-image-io` to name the shared loss
vocabulary. The edge is acyclic and consistent with the existing graph:
`ritk-image-io` depends only on `ritk-codecs`, `ritk-image`, and `ritk-spatial`,
and `ritk-io` already depends on `ritk-nifti` and `ritk-nrrd`, which depend on
`ritk-image-io`. The dependency direction follows DIP: `ritk-io` consumes the
`FormatMetadataLoss` abstraction; `ritk-image-io` still depends on no format
crate.

The DICOM conversion call site that passes `dicom_metadata_losses` to
`prepare_conversion` lands with the DICOM stored read
([RITK-DICOM-STORED-IMPORT-001](../../backlog.md#RITK-DICOM-STORED-IMPORT-001)),
which is the item that first produces a `StoredSeries` for a DICOM source.
Until then the projection is the tested seam and the reader guarantees that no
metadata is dropped without a record.

## Evidence and revision criteria

`ritk-io` tests assert that an opaquely retained element projects to no loss;
that each `DicomRetentionReason` projects to a scoped
`FormatMetadataLoss::UnknownSemantics` carrying the `(GGGG,EEEE)` tag and the
reason text; that a sequence nested past `MAX_RETAINED_SEQUENCE_DEPTH` records
`NestingDepthExceeded` at the boundary element; and that
`dicom_series_metadata_losses` scopes each slice's losses to that slice's
frame. Round-trip preservation tests confirm that private creator scopes,
private text, and private bytes still survive a write/read cycle.

Revise this decision if a DICOM element can be parsed but neither interpreted,
retained, nor recorded, or if the shared conversion contract replaces
`FormatMetadataLoss` with a representation the projection cannot produce.
