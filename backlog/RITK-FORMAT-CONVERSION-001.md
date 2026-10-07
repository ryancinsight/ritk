<a id="RITK-FORMAT-CONVERSION-001"></a>

## RITK-FORMAT-CONVERSION-001 — Convert supported formats through RITK — blocked
- outcome: Keep medical and scientific readers, writers, and conversions in RITK behind typed models and explicit capabilities.
- acceptance: DICOM, NIfTI, NRRD, MetaImage, MINC, MIF, MGH/MGZ, Analyze, VTK, PNG, TIFF, JPEG, GIFTI, mesh, and tractogram routes preserve represented samples and semantics or report typed loss before output changes; Métis contains no parser or converter.
- scope: crates/ritk-image-io/, crates/ritk-io/, listed format crates, conversion tests, and the RITK manual
- next: Deliver capability and NIfTI/NRRD stored-sample slices first, then continue by the dependency graph.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-IO-FORMAT-CAPABILITIES-001, RITK-IMAGE-CONVERSION-ADAPTERS-001, RITK-NIFTI-STORED-SERIES-001, RITK-NIFTI-STORED-READ-001, RITK-NRRD-NIFTI-001, RITK-DICOM-CONVERSION-001, RITK-METAIMAGE-CONVERSION-001, RITK-MINC-CONVERSION-001, RITK-MIF-CONVERSION-001, RITK-MGH-CONVERSION-001, RITK-ANALYZE-CONVERSION-001, RITK-VTK-VOLUME-CONVERSION-001, RITK-RASTER-CONVERSION-001, RITK-GIFTI-SURFACE-001, RITK-MESH-CONVERSION-001, RITK-TRACTOGRAM-CONVERSION-001
- priority: architecture
