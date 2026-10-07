<a id="RITK-IO-FORMAT-CAPABILITIES-001"></a>

## RITK-IO-FORMAT-CAPABILITIES-001 — Declare shared format route capabilities — todo
- outcome: Make shared RITK image-format dispatch accurately describe the codecs and models it reads and writes.
- acceptance: Every ImageFormat capability query matches an executable adapter; PNG writes and MINC/MIF image codecs route through ritk-io; value-semantic tests cover real codec outputs and rejected directions.
- scope: crates/ritk-io/src/dispatch.rs, crates/ritk-io/src/format/, Cargo.toml, route tests, and format docs
- next: Map codec reader/writer endpoints, then add the missing registrations with real value round-trips.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: todo
- needs: none
- priority: architecture
