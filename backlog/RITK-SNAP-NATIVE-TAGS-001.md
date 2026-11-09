<a id="RITK-SNAP-NATIVE-TAGS-001"></a>
## RITK-SNAP-NATIVE-TAGS-001 — Show searchable DICOM tags natively — todo
- outcome: A native dialog lists the current image's tags with search over tag, description and value.
- acceptance: A native test asserts the filtered rows for known tags from the public phantom.
- scope: `crates/ritk-snap/src/presentation/native_session/window_controls/`, `crates/ritk-snap/src/dicom/metadata_table*.rs`
- next: Add a modal over the metadata table. Series tree labels and the match count are in `414da7491:crates/ritk-snap/src/ui/sidebar/panel.rs`.
- basis: 30b8ba098
- status: todo
- needs: none
- milestone: M-viewer-access
- priority: feature
