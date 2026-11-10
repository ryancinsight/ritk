<a id="SEC-MIF-DETACHED-PATH-001"></a>

## SEC-MIF-DETACHED-PATH-001 — Confine a .mif detached data path to the header's directory — todo
- outcome: A `.mif` `file:` key cannot make the reader load an arbitrary file as voxels; today `crates/ritk-mif/src/reader.rs:219-225` joins it to the header's directory without rejecting an absolute path or `..` (`Path::join` with an absolute argument discards the base).
- acceptance: an absolute path and one with `..` each return a typed `MifError` variant before any file opens, asserted with `matches!` and failing with the guard reverted; a plain sibling name still reads exact values.
- scope: crates/ritk-mif/src/{reader.rs,error.rs,reader_tests.rs}
- next: check components where the detached path is built. Found by the ritk#874 judge at 8081bafb6.
- basis: 90621b8b7472e691407b7c2deb8e21831cf4f203
- milestone: M-viewer-reading
- status: todo
- priority: correctness
