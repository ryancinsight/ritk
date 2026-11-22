<a id="ritk-stable-qualification"></a>
## RITK-STABLE-QUALIFICATION — ritk stable-release qualification [patch] — todo
- priority: verification; milestone: M-viewer-access; basis: `3a2bcc77b`.
- Outcome: Qualify the DICOM export boundary in the real viewer. Follow the [release scope](../docs/adr/0059-release-scope-1-0.md) and [milestone exits](../backlog.md).
- Representative manifest `ritk-snap 0.2.0`. Target 1.0.0.
- [x] [patch] Resolve URL 2.5.9 and the [Metis provider update](https://github.com/ryancinsight/metis/pull/589) in the viewer lock. The all-feature audit clears RUSTSEC-2026-0592 and RUSTSEC-2023-0071.
- [x] [patch] Triage all 13 unmaintained package versions and the yanked executor. Remove four maintenance findings, update the executor, and record the [owner and migration for every remaining dependency](../gap_audit.md#viewer-dependency-audit-patch).
- [ ] [patch] Retire the nine remaining unmaintained versions across the independent DICOM test peer, native accessibility and font stack, Metal backend and EXR codec. Replay the unwaived all-feature audit after each migration.
- [ ] [patch] Verify the viewer-delivered directory export after the [numbered export](https://github.com/ryancinsight/ritk/pull/1104) and [DICOM conformance follow-up](https://github.com/ryancinsight/ritk/pull/1105) merged. Export a study with identifying source filenames and references, reopen the result and check path names, UID relationships and unchanged source files.
- [ ] [patch] Run the native reading, comparison and access exits with a real study. Assert completed tool results and patient-position alignment in the rendered session.
- [ ] [patch] Read all 94 MRI files through the native Safari chooser, verify decoded pixels and malformed-file rejection, and confirm cleanup. The [WebKit item](RITK-BROWSER-READ-001.md) records the hosted file-access denial.
- [x] [patch] Verify the hosted [WebGPU saved-study scenario](RITK-BROWSER-WEBGPU-001.md), including its three-plane cine-rate trace and session teardown.
- [ ] [patch] Merge once all eight saved-study scenarios pass together, then replay the full matrix and verify CI at the resulting main revision.
- [ ] [major] Complete M-viewer-advanced under ADR 0059. Document the viewer freeze and verify its installable distribution while honoring ritk-snap's private Cargo manifest and the independently released libraries.
- [ ] [patch] Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json).
