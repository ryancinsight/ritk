# ADR 0063: Reconstruct Enhanced PET concatenations

- Status: Proposed
- Class: [minor]
- Date: 2026-11-19

A selected Enhanced PET member identifies its concatenation by UID. Scans
retain the validated Part 10 bytes. The reader checks the source SOP
Instance UID, member identifiers, shared functional groups, dimensions,
and scalar pixel layout before reconstruction.

For member rank k in increasing offset order, the expected zero-based
offset is the sum of frame counts of preceding members. Member numbers
start at one. A declared total must match the number of supplied members.
An absent total permits detecting a gap or overlap, while a missing
trailing member leaves no observable gap.

Reconstruct volumes by common nonspatial ordinals before geometry analysis.
Retain an instance index and local frame index for each plane. This address
selects the original transfer syntax, display transform, pixel padding, and
real-world value mapping. Sorting the original planes together makes
interleaved phases and gates equivalent to a standalone object with the same
frame order, positions, and mappings. Decoding each member into independent
volumes would infer the wrong spacing at member boundaries and could split
one gate across multiple display volumes.

The reader charges aggregate encoded and decoded byte ceilings before using
the reconstruction. The viewer consumes the same volume-set reader for
scanned files and retained Part 10 inputs. Elapsed time divided by the
isotope half-life yields a dimensionless decay exponent in Aequitas.

The DICOM contract is defined by [PS3.3 C.7.6.16][groups] and
[C.7.6.17][dimensions]. Tests compare an eight-frame object with two
four-frame members, including voxel values, SUV, and patient geometry.

[groups]: https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.html
[dimensions]: https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.html

## Slice provenance after spatial reconstruction

Store the interpolation decision per target plane when resampling occurs.
The exact acquired position, a linear combination of neighboring
positions, and a copy from another position are separate cases. A
complete regular stack needs no additional allocation. Account for
provenance in the per-volume and aggregate decoded-byte limits. The
display and SUV buffers share the same target positions. Provenance is
independent of the pixel values and follows the selected reconstruction
when temporal or respiratory gates change.

Show the flag on the storage-depth view, where a displayed slice
corresponds to one reconstructed plane. An orthogonal reformatted slice
can contain several such planes.
