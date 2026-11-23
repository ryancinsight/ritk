<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 — Resolve WebKit study-file reads [patch] — in progress
- priority: verification; milestone: M-viewer-compare
- outcome: Read the public MRI-DIR study through the supported browser file chooser.
- acceptance: A 94-file replay checks decoded pixels, bounded malformed-file rejection, and listener cleanup on Chromium, Firefox, and WebKit.
- scope: RITK browser presentation, Metis file boundary, and cross-engine replay
- basis: 54140426be0173d8aa8f937bc49c6255647231b4
- [x] [patch] The [macOS Open panel](https://github.com/ryancinsight/ritk/actions/runs/45102078038/job/137454983653) delivered all 94 files (49,807,236 bytes) to the workstation. Safari decoded 94 × 512 × 512 voxels at indices 47, 256 and 256. The [independent picker replay](https://github.com/ryancinsight/ritk/actions/runs/45098982271) matched every file checksum.
- [ ] [patch] Confirm the three displayed planes, rejected file batches and listener cleanup in the native Safari workstation. The [hosted canvas read](https://github.com/ryancinsight/ritk/actions/runs/45103222522/job/137458583318) returned InvalidStateError at getImageData on the 1024 × 622 framebuffer.
- next: Qualify the rendering surface on the supported macOS Safari runner, then replay the eight scenarios together.
- needs: none
