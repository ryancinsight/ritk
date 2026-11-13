<a id="SEC-MIF-DETACHED-DEVICE-NAME-001"></a>

## SEC-MIF-DETACHED-DEVICE-NAME-001 — Refuse device names as .mif detached data and pin the check's order — todo
- outcome: A `.mif` `file:` name cannot make the reader open a Windows device, and the refusal provably precedes any open. A reserved name such as `NUL` is a plain path component, so the confinement check (ritk#889) passes it; on Windows 11 build 26340 `<dir>\NUL` opens the null device and reads zero bytes, and `CON`, `AUX`, `COM1` gave os error 2 there but may resolve to devices on older Windows. The #889 judge's mutant K5 (open first, return the check's error) survives every test, and with the guard off a UNC name made the reader attempt an SMB connection.
- acceptance: on Windows, `NUL`, `CON`, `COM1` and `nul.dat` each return a typed `MifError` before any open, asserted with `matches!` and failing with the guard removed; a test observes that no open precedes a refusal and fails under K5; a committed case covers a real UNC and a verbatim prefix; the variant doc, error message, `# Errors` lines and book name drive-relative and rooted names beside "absolute or contains `..`"; the banner comment in `reader_tests.rs` is gone.
- scope: crates/ritk-mif/src/{reader.rs,error.rs,reader_tests.rs,reader_tests/detached_path.rs}, docs/book/mif_format.md
- next: establish from Microsoft's file-naming documentation which reserved names still resolve to devices inside a directory path on the supported Windows versions; choose the ordering observable (a Unix FIFO or an opener seam).
- basis: 729bbd78254c1eada68211225f350ecfebaa3666
- milestone: M-viewer-reading
- status: todo
- needs: none
- priority: correctness
