<a id="RITK-SNAP-NATIVE-DOCUMENTS-001"></a>
## RITK-SNAP-NATIVE-DOCUMENTS-001 — Display reports, PDFs and video objects — todo
- outcome: Structured reports, encapsulated PDFs and MPEG DICOM objects open from the series browser.
- acceptance: Tests load public or synthetic objects and assert the extracted report text, PDF bytes, and decoded frame count.
- scope: `crates/ritk-io/src/format/dicom/`, `crates/ritk-snap/src/dicom/loader/`, `crates/ritk-snap/src/presentation/native_session/`
- next: Classify non-image SOP classes at series discovery.
- basis: 30b8ba098
- needs: none
- milestone: M-viewer-advanced
- priority: feature
