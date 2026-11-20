# 0063 — RITK owns the DICOM dataset and Part 10 writer

- Status: Accepted
- Item: RITK-NATIVE-DICOM

## Decision

Move the file, element, value, sequence, and encapsulated frame contracts
into `ritk-dicom`. Use the bounded scanner for both borrowed metadata and
owned datasets. The same dataset feeds the Part 10 writer, framed pixel
decode, and DIMSE. Resolve transfer syntax before dispatch to the codec.
Retain independent DIMSE peer coverage in `ritk-io` tests.

## Alternatives

Keeping an external mutable object requires a second DICOM dictionary and
object traversal. Changing only parse would preserve the external dependency
through repair and export. A parallel writer would create two definitions of
sequence lengths and file metadata.

## Consequences

`ritk-dicom` 0.5 and `ritk-io` 0.6 use their own dataset types. Migrate
dependent code with the [changelog](../../CHANGELOG.md) and review the
[wire design](../../crates/ritk-dicom/DESIGN.md). The previous pre-1.0
versions remain available for consumers waiting to migrate.
