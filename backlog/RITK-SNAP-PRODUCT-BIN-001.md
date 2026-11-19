<a id="RITK-SNAP-PRODUCT-BIN-001"></a>
## RITK-SNAP-PRODUCT-BIN-001 — Name the executable AsterSlice — todo
- milestone: M-viewer-reading
- outcome: the AsterSlice executable has its own product name and no longer collides with the ritk_snap library PDB.
- acceptance: all explicit binary callers and CLI help use asterslice; native launch and capture work; all-targets gate reports no duplicate ritk_snap PDB output; package/library/storage identities stay stable.
- scope: `crates/ritk-snap/Cargo.toml`, CLI definitions, explicit binary callers, manual, indexed ADR and migration guide
- next: enumerate binary callers; record the CLI-name break and migration, rename the binary without a compatibility executable, then verify both shell configurations.
- basis: c4c271b5b253b0ec92880747dbc67933b96ef6ca
- needs: none
- priority: verification
