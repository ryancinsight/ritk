<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 — Resolve WebKit study-file reads [patch] — in progress
- priority: verification; milestone: M-viewer-compare
- outcome: Read the public MRI-DIR study through the supported browser file chooser.
- acceptance: A 94-file replay checks decoded pixels, bounded malformed-file rejection, and listener cleanup on Chromium, Firefox, and WebKit.
- scope: RITK browser presentation, Metis file boundary, and cross-engine replay
- basis: 54140426be0173d8aa8f937bc49c6255647231b4
- [x] [patch] The [macOS Open panel](https://github.com/ryancinsight/ritk/actions/runs/45102078038/job/137454983653) delivered all 94 files (49,807,236 bytes) to the workstation. Safari decoded 94 × 512 × 512 voxels at indices 47, 256 and 256. The [independent picker replay](https://github.com/ryancinsight/ritk/actions/runs/45098982271) matched every file checksum.
- [x] [patch] [Native Safari on Intel macOS](https://github.com/ryancinsight/ritk/actions/runs/45103222522/job/137458583614) passed the three full-plane reference digests, visible samples, three real rejected batches, reading tools, independent CT/MR comparison and listener cleanup. The [ARM macOS 26 runner](https://github.com/ryancinsight/ritk/actions/runs/45103222522/job/137458583318) decoded the MRI but failed framebuffer readback.
- [x] [patch] [Safari opened both workstation and independent reference selections](https://github.com/ryancinsight/ritk/actions/runs/45106042786) with the native panel’s focused file list. The test verified all three reference images and listener teardown.
- next: Run the Intel Safari job with the other seven scenarios on the integrated candidate and on main.
- needs: none
