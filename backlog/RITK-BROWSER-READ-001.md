<a id="RITK-BROWSER-READ-001"></a>

## RITK-BROWSER-READ-001 — Resolve WebKit study-file reads — blocked
- outcome: Read the public MRI-DIR study through the supported browser file chooser.
- acceptance: A 94-file replay checks decoded pixels, bounded malformed-file rejection, and listener cleanup on Chromium, Firefox, and WebKit.
- scope: RITK browser presentation, Metis file boundary, and cross-engine replay
- next: Reopen when hosted WebKit permits bounded Blob reads.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Safari 26.6.2 accepts all files but rejects the first bounded Blob.arrayBuffer read; run 35759891764 records the failure.
- needs: none
- priority: verification
