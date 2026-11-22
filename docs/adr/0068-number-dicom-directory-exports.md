# 0068 — Number DICOM directory exports

Status: Accepted

## Context

Directory anonymization reused source filenames. Those names can disclose a
patient identifier or an original DICOM UID after the selected metadata rules
have run. Repeated or concurrent calls could replace an instance on disk.
Python exposes this Rust directory function and returns its per-file counts.

## Decision

Share the study export's one-based `NNNNNN.dcm` naming rule. Format only the
ordinal into each filename. Directory mode visits immediate files in path
order. Each run increments its candidate ordinal and publishes a completely
encoded file only if its destination name is unoccupied. On collision it
tries the next ordinal with the same encoded bytes. The destination can hold
existing files. Compare directory identities before writing to reject aliases
of the source directory.

The filename function is injective over one-based ordinals, and exclusive
publication prevents two successful files from claiming the same path. The
final name is visible only after the DICOM writer completes. Anonymization
uses the same profile, options and UID transformation as before. Per-file
failures contribute to `error_count` and retain their source paths in `errors`.

## Alternatives

Calling `anonymize_dicom_study` would add recursive traversal, omit DICOMDIR,
require an empty destination, run the verification gate, roll back on failure,
and return different statistics. Keep directory-mode semantics for existing
Rust and Python callers. Creating numbered files directly would prevent
replacement with an exclusive open, but readers could see unfinished files.

## Consequences and verification

Callers use a separate output directory and discover generated filenames there.
Anonymized metadata provides instance identity. Destination files written by
previous calls can include duplicate DICOM UIDs even though each path is unique.
Diagnostic paths may contain identifying information. The filename rule applies
to files written by the current call. Optional pixel zero-fill and verification
of Annex E profile conformance have their own requirements.

The synthetic Python fixture checks fresh names, retries, concurrent writers,
as-is pixel bytes, UID relationships, source bytes, and write failures. The
Rust test passes a numbered export through the directory verification gate.
