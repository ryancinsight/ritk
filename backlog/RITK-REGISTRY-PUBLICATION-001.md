<a id="RITK-REGISTRY-PUBLICATION-001"></a>
## RITK-REGISTRY-PUBLICATION-001 — Publish the required registry dependency graph — blocked
- priority: correctness; milestone: M-viewer-access
- outcome: crates.io consumers can install native DICOM 0.5, I/O 0.6, and the corrected registration graph.
- [major] Spatial 0.5.0 and Diffusion Scheme 0.4.0 are published. Annotation 0.4 uses the same Spatial identity. See [ADR 0062](../docs/adr/0062-spatial-provider-051.md).
- [arch] Native DICOM and DIMSE cutover implementation and proofs are recorded in [ADR 0063](../docs/adr/0063-native-dicom-dataset.md) and the [gap audit](../gap_audit.md#ritk-native-dicom).
- The checkout resolves from no lockfile with published Apollo and Coeus providers. The viewer keeps its separate Métis source.
- Existing [first release checkpoint](https://github.com/ryancinsight/ritk/actions/runs/44659786417) verifies and publishes Spatial 0.5, Diffusion Scheme 0.4, and DICOM 0.4 in dependency order.
- [RITK #1065](https://github.com/ryancinsight/ritk/pull/1065) prepares an independent eleven-package registry cohort. The native [cutover](https://github.com/ryancinsight/ritk/pull/1066) includes the remaining dependency closure.
- Verify exact-head required checks and normalized archives. Publish providers before dependents and verify the registry receipts and tags.
- Block Matching, Image I/O, and Parcellation require first publication. Merging to main triggers the existing crates.io workflow.
