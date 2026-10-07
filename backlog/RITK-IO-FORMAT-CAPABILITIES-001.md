<a id="RITK-IO-FORMAT-CAPABILITIES-001"></a>

## RITK-IO-FORMAT-CAPABILITIES-001 — Declare shared format route capabilities — done
- outcome: The shared `ritk-io` image-format dispatch names every codec it can reach, so a consumer never depends on a format crate for a route the dispatch already owns.
- acceptance: Every ImageFormat capability query matches an executable adapter; PNG writes and MINC/MIF image codecs route through ritk-io; value-semantic tests cover real codec outputs and rejected directions.
- scope: crates/ritk-io/src/{dispatch.rs,format/}, crates/ritk-io/Cargo.toml, crates/ritk-cli/src/commands/, route tests, and format docs
- next: Landed. `ImageFormat` gained `Minc` and `Mif`, with `.mnc`/`.mif` path inference, `as_str`, and `from_str_name` to match. `ritk-io::format::mif` is a new route over `ritk-mif`, which `ritk-io` could not reach before (no dependency, no module); `ritk-io::format::png::native::PngWriter` is a zero-sized adapter over `ritk-png::write_png`, closing PNG's read-only gap. `is_native_write_capable` now includes `Png`, `Minc`, and `Mif`; DICOM stays read-only because its writer is a series directory, and that is now stated rather than implied. The CLI's `is_read_capable`/`is_write_capable` were a second copy that had already drifted from the dispatch; they now delegate to `ritk-io`, and `OutputFormat` gained `minc`/`mif`/`png`. Tests: `native_capability_matrix_matches_dispatch` asserts per format that `write_image_native` succeeds exactly when `is_native_write_capable` says so, and that a file the dispatch wrote reads back — a capability flag that drifts from its route fails there. `every_format_round_trips_through_path_and_name` pins `from_path` and `from_str_name` against both enumerations. MIF and PNG contract round-trips plus a PNG volume-rejection case were added to the shared route harness. Falsified by dropping `Png` from `is_native_write_capable` while the route still wrote it: the matrix test failed with `Png: write_image_native disagreed with is_native_write_capable (capable=false, result=Ok(()))`.
- basis: 2929ff307af7119673c47b6f3570f1095b848bf9
- status: done
- needs: none
- priority: architecture
