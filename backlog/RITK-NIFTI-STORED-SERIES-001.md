<a id="RITK-NIFTI-STORED-SERIES-001"></a>

## RITK-NIFTI-STORED-SERIES-001 — Construct NIfTI documents from stored series — todo
- outcome: Create NIfTI-1 and NIfTI-2 documents from stored series without an intermediate file.
- acceptance: Supported stored encodings retain exact bytes, spatial forms, units, calibration, and representable axes; unrepresentable metadata yields scoped typed loss before destination mutation.
- scope: crates/ritk-nifti/, crates/ritk-image-io/, NIfTI conversion tests, and the guide
- next: Re-derive and split closed PR #794 against current main using the merged scalar contract.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: correctness
