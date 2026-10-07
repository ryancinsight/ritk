<a id="RITK-RASTER-CONVERSION-001"></a>

## RITK-RASTER-CONVERSION-001 — Convert PNG TIFF and JPEG rasters — blocked
- outcome: Convert raster inputs through image-specific models without inventing physical geometry.
- acceptance: Lossless PNG/TIFF routes preserve supported decoded pixels and represented metadata or return typed loss before output changes; JPEG routes report lossy encoding before output changes, and decoded samples match RITK's independent T.81/quantizer oracle within its input-derived bound; absent physical geometry stays absent and dispatch matches codec capabilities.
- scope: crates/ritk-io/src/format/{png,tiff,jpeg}/, raster model, and conversion tests
- next: Derive raster model semantics from each codec, then test lossless values and JPEG reconstruction against the T.81/quantizer oracle.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IO-FORMAT-CAPABILITIES-001
- priority: architecture
