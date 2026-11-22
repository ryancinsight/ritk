<a id="RITK-FIXTURE-RETENTION-001"></a>
## RITK-FIXTURE-RETENTION-001 — Decide imaging fixture retention — blocked
- priority: verification; milestone: M-viewer-reading
- outcome: Preserve identifiable or unique imaging data until its owner records a retention decision.
- acceptance: Each inventory row with unknown provenance, license, or identifying metadata has an explicit owner decision before removal or redistribution.
- scope: test_data/, payload manifest, and fixture documentation
- next: After the provenance inventory, ask the owner about each unresolved row; keep all bytes until answered.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: The inventory finds 308 files under 2_skull_ct and 2,334 under paired_mri_ct. All 304 and 1,167 parsed DICOM headers respectively contain a populated patient name or ID tag. These counts include directory records and reveal no tag values. The owner prefers a single shared catalog and supplied no release terms for these two collections. Keep their bytes in place.
- needs: RITK-GAP-2026-08-20-04
