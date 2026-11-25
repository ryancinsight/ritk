<a id="RITK-DICOM-PIXEL-ASPECT-PINS-001"></a>
## RITK-DICOM-PIXEL-ASPECT-PINS-001 — Pin pixel aspect parser edge cases and its zero-spacing error — done [patch]
- priority: verification; milestone: M-viewer-reading
- outcome: The parser's value-domain edges are pinned by tests, a conformant zero spacing is not reported as a malformed value, and the public `DicomPixelAspect` addition has its CHANGELOG entry.
- acceptance: `git grep -n -E '"(a|-1|\+6)[\\]+5"|" 6[\\]+ 5"|"[\\]+5"|, ""\)' -- crates/ritk-io/src/format/dicom/reader/pixel_aspect.rs` prints the alphabetic, small-negative, leading-plus, leading-space, empty-component and empty-value cases (nothing at the basis), and `cargo nextest run -p ritk-io -E 'test(pixel_aspect)'` passes with each case asserting its exact value or error. A spacing source given `0\1` returns an error variant other than `MalformedValue`, asserted by a test (PS3.3 10.7.1.3 allows zero for a single row or column; pixel_aspect.rs:203-204 returns `MalformedValue` for it today). `git grep -n DicomPixelAspect -- CHANGELOG.md` prints a [minor] entry (nothing at the basis).
- scope: crates/ritk-io/src/format/dicom/reader/pixel_aspect.rs, CHANGELOG.md
- basis: 779d3742bf9808973a3c03d5bc37c4a668270503
- needs: none
- verification: The parser tests cover signed and padded IS values, malformed components, numeric underflow, and zero in either DS component for all three spacing attributes.
