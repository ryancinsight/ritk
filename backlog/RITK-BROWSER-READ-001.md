<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 — Resolve WebKit study-file reads [patch] — in progress
- priority: verification; milestone: M-viewer-compare
- outcome: Read the public MRI-DIR study through the supported browser file chooser.
- acceptance: A 94-file replay checks decoded pixels, bounded malformed-file rejection, and listener cleanup on Chromium, Firefox, and WebKit.
- scope: RITK browser presentation, Metis file boundary, and cross-engine replay
- next: Verify the packaged workstation and three-plane reference through the native Safari Open panel on macOS. The [picker test](https://github.com/ryancinsight/ritk/actions/runs/45098982271) read and verified all 94 files. SafariDriver upload behavior remains tracked [upstream](https://bugs.webkit.org/show_bug.cgi?id=322034).
- basis: 54140426be0173d8aa8f937bc49c6255647231b4
- blocker: In [hosted Safari 27.2](https://github.com/ryancinsight/ritk/actions/runs/45079835823/job/137389340300), the trusted chooser reported all 94 files and 49,807,236 bytes. macOS denied WebContent the first file read. The workflow verified identical copies of all public fixtures, readable files and traversable parent directories. Safari 26.6.2 previously failed the same read in run 35759891764.
- needs: none
