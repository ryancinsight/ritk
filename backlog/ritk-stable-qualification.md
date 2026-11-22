<a id="ritk-stable-qualification"></a>
## RITK-STABLE-QUALIFICATION — ritk stable-release qualification [patch] — todo
- priority: verification; milestone: M-viewer-access; basis: `3a2bcc77b`.
- Outcome: Qualify the DICOM export boundary in the real viewer. Follow the [release scope](../docs/adr/0059-release-scope-1-0.md) and [milestone exits](../backlog.md).
- Representative manifest `ritk-snap 0.2.0`. Target 1.0.0.
- [x] [patch] Resolve URL 2.5.9 and the [Metis provider update](https://github.com/ryancinsight/metis/pull/589) in the viewer lock. The all-feature audit clears RUSTSEC-2026-0592 and RUSTSEC-2023-0071.
- [ ] [patch] Triage the 13 unmaintained package-version findings and the yanked async-executor warning in the locked all-feature graph. Record each dependency owner and replacement path.
- [ ] [patch] Verify the viewer-delivered directory export after the [numbered export](https://github.com/ryancinsight/ritk/pull/1104) and [DICOM conformance follow-up](https://github.com/ryancinsight/ritk/pull/1105) merged. Export a study with identifying source filenames and references, reopen the result and check path names, UID relationships and unchanged source files.
- [ ] [patch] Run the native reading, comparison and access exits with a real study. Close the [WebKit file-read and hosted WebGPU exits](https://github.com/ryancinsight/ritk/pull/1108) in the Metis browser matrix. Assert completed tool results and patient-position alignment in the rendered session.
- [ ] [major] Complete M-viewer-advanced under ADR 0059. Document the viewer freeze and verify its installable distribution while honoring ritk-snap's private Cargo manifest and the independently released libraries.
- [ ] [patch] Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json).
