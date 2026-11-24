<a id="RITK-DICOM-PIXEL-ASPECT-RESOLUTION-001"></a>
## RITK-DICOM-PIXEL-ASPECT-RESOLUTION-001 — Resolve and retain pixel aspect on DICOM metadata — todo
- priority: feature; milestone: M-viewer-reading
- outcome: The reader resolves one `DicomPixelAspect` per slice or frame from the retained elements and keeps it on the series metadata, so a viewer can tell a retained aspect from the 1:1 default. Absence maps to `None`; nothing is invented.
- acceptance: `git grep -n 'DicomPixelAspect::try_from' -- crates ':!crates/ritk-io/src/format/dicom/reader/pixel_aspect.rs'` prints a reader caller (it prints nothing at the basis; only three `pub use` lines name the type). Value-semantic tests assert PS3.3 C.7.6.3.3: Pixel Aspect Ratio (0028,0034) is Type 1C and applies only when Pixel Spacing (0028,0030), Imager Pixel Spacing (0018,1164) and Nominal Scanned Pixel Spacing (0018,2010) are all absent, in the image or its Shared or Per-Frame Functional Groups; with none of the four the metadata holds `None`; the rule for spacing and ratio both present is chosen from the standard's text (PS3.3 states no precedence) and pinned.
- scope: crates/ritk-io/src/format/dicom/reader/parse.rs, crates/ritk-io/src/format/dicom/reader/scan/finalize.rs, crates/ritk-io/src/format/dicom/reader/pixel_aspect.rs
- next: start at parse.rs:160, where Pixel Spacing is read as a geometry array, and route the four elements through `DicomPixelAspect`; keep the detector and scanner sources out of patient-space distance.
- evidence: the merged parser module is only a node parser; the row-then-column order, the IS and DS length limits and the typed errors live there, and nothing in the reader calls it.
- basis: 779d3742bf9808973a3c03d5bc37c4a668270503
- needs: none
