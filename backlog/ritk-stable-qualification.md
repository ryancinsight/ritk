<a id="ritk-stable-qualification"></a>
## RITK-STABLE-QUALIFICATION — ritk stable-release qualification [patch] — todo
- priority: verification; milestone: M-viewer-access; basis: `533f2e07724d`.
- Outcome: Qualify the DICOM export boundary in the real viewer. Follow the [release scope](../docs/adr/0059-release-scope-1-0.md) and [milestone exits](../backlog.md).
- Representative manifest `ritk-snap 0.2.0`. Target 1.0.0.
- [ ] [patch] Integrate the [URL lock update](https://github.com/ryancinsight/ritk/pull/1107). Its all-feature audit clears RUSTSEC-2026-0592. Refresh the Metis/Moirai dependency on the [patched RSA provider](https://github.com/ryancinsight/Moirai/pull/721), then rerun the locked audit. Triage the 13 unmaintained package-version findings and the yanked async-executor warning separately.
- [ ] [patch] Coordinate directory export with [PR #1104](https://github.com/ryancinsight/ritk/pull/1104). Export a study with identifying source filenames and references, reopen the result and check path names, UID relationships and unchanged source files.
- [ ] [patch] Run the native reading, comparison and access exits with a real study and the Metis browser jobs. Assert completed tool results and patient-position alignment in the rendered session.
- [ ] [major] Complete M-viewer-advanced under ADR 0059. Document the viewer freeze and verify its installable distribution while honoring ritk-snap's private Cargo manifest and the independently released libraries.
- [ ] [patch] Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json).
