<a id="RITK-MGH-CONVERSION-001"></a>
## RITK-MGH-CONVERSION-001 — Preserve MGH and MGZ volume semantics — todo
- priority: correctness; milestone: M-viewer-reading
- outcome: convert MGH/MGZ volumes through RITK without changing represented samples or geometry.
- acceptance: supported scalar types, affine geometry, calibration, and compressed/uncompressed packaging round-trip or return typed loss before output changes.
- scope: crates/ritk-mgh/, crates/ritk-io/, MGH tests and manual
- next: Implement stored-sample adapters against the [field inventory](../docs/book/mgh_format.md#conversion-inventory), including preflight for unknown geometry, DOF and trailing metadata.
- evidence: Same-format MGH/MGZ repacking preserves stored bytes and metadata. Cross-format projection to the shared stored-volume model remains open.
- basis: 4ebc650d6e25a7a6775910ba23bc35c8c7cb78e4
- needs: RITK-MGH-CONVERSION-INVENTORY-001
