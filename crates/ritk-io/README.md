# ritk-io

Unified medical image I/O for [RITK](https://github.com/ryancinsight/ritk).

Owns cross-format dispatch and the `ImageReader` / `ImageWriter` contracts; the
byte-level parsing lives in the per-format crates.

| Format | Read | Write |
|---|---|---|
| DICOM (series) | yes | yes |
| NIfTI (`.nii` / `.nii.gz`) | yes | yes |
| MetaImage (`.mha` / `.mhd`) | yes | yes |
| NRRD | yes | yes |
| PNG | yes | yes |
| TIFF / BigTIFF | yes | yes |
| MGH / MGZ (FreeSurfer) | yes | yes |
| Analyze 7.5 | yes | yes |
| MINC2 | yes | yes |
| VTK legacy structured points | yes | yes |
| JPEG | yes | 2-D grayscale only |

`read_image_native` and `write_image_native` select the format by path and
content. The crate also ships a native-only DICOMweb client (QIDO / WADO /
STOW), backed by a blocking desktop transport, and a PS 3.15 Annex E
de-identification toolset with an export-time metadata integrity gate — see
`examples/anonymize_pacs_export.rs`. Browser hosts use their platform fetch
implementation and pass completed, bounded DICOM bytes to the same RITK byte
reader; no second decoder or browser-specific volume model is introduced.

## Usage

```toml
[dependencies]
ritk-io = "0.3.0"
```

DICOM writers derive pixel bit attributes from their encoded sample type:
unsigned 16-bit for series and most multi-frame syntaxes, unsigned 8-bit for
baseline JPEG. Input and serialization preflight finish before output changes;
invalid input preserves existing files. A `DicomWriteError` cause identifies
invalid pixels, dimensions, or metadata through the existing writer result.
Preflight retains serialized output in memory; persistence errors can leave a
partially written series. See the [DICOM format chapter](../../docs/book/dicom_format.md).

DICOM opening preserves acquisition identity: `scan_dicom_path` accepts a
selected instance, directory, or explicit DICOMDIR; `scan_dicom_files` scans
an exact discovered member set. Feed the resulting descriptor to
`load_dicom_from_series` to decode pixels and retain metadata. Multiple
SeriesInstanceUID values require explicit selection; missing or invalid image
UIDs and inconsistent dimensions fail before geometry assembly. Named byte
batches use the same single-series contract. Scanned slices retain the bytes
whose identity and metadata were validated. Pixel decoding consumes those
bytes, so replacing a path after scanning cannot substitute another image.
Presentation hosts, including browser or desktop shells, call this public
byte-batch API and receive an RITK `Image` plus `DicomReadMetadata`; they do not
own a second DICOM decoder or volume model.

`read_dicom_stored_series` and `load_dicom_stored_series` preserve exact stored
integer samples, raw per-slice patient geometry, modality calibration, and its
unit label in `StoredSeries`. The stored-scalar reader currently accepts native
little-endian single-frame monochrome instances. Color, multi-frame, and
encapsulated pixel data return typed errors rather than entering a scalar
volume.

Native directory callers that already have the acquisition UID use
`read_native_dicom_series_with_uid`; it scans the directory, matches the exact
UID, and only then decodes the selected series. Omitting an explicit selection
continues to require one unambiguous image series.

Multi-series workflows use `DicomStudyCatalog::scan`. Its `series()` slice
reports each `SeriesInstanceUID`, `instance_count()`, description, modality,
patient identifier, and member paths. `load(uid, backend)` validates only the
selected member set, then returns its decoded image and complete geometry
metadata. `scan_with_budget` rejects an excessive candidate count or cumulative
encoded size before parsing any candidate. The catalog never selects the
largest or first series implicitly.

The bounded DICOM reader scan and series-load entry points have a `*_with_budget`
form accepting the typed `DicomReadBudget`. Its parser component is the Atlas
`ritk_dicom::ParseBudget`; the other fields set independent candidate-count,
retained-study, and decoded-workspace ceilings. Construct one with
`DicomReadBudget::try_new(parser, max_retained_bytes, max_decoded_bytes)` when a
host needs limits below the finite default, or
`DicomReadBudget::try_new_with_max_instances` to set the candidate ceiling
independently. Structural validation runs before dicom-rs object
materialization, retained bytes are charged before each slice is stored, and
the loader checks the planned peak frame/resample/volume workspace before
allocation. Budgeted filesystem reads resolve a path once,
compare the opened handle with the resolved path metadata, and read from that
handle. Scanned slices retain those validated bytes for later decoding.

An existing DICOMDIR is authoritative. Its index is subject to the parser
budget, and invalid or missing references fail instead of falling back to
unrelated folder contents. The reader validates RecordInUseFlag, linked
next/lower record offsets, root-chain termination, cycles, and active
reachability. Only reachable active IMAGE records contribute members, and each
record's referenced SOP class, SOP instance, and transfer syntax must match the
referenced Part 10 file. Reference checks reject absolute paths, traversal, and
canonical paths outside the file-set root. The reader retains validated member
bytes, so replacing a path after scanning cannot substitute another image during
decode. The handle metadata check detects replacement between path resolution and
handle inspection. On Unix, the final open uses `O_NOFOLLOW`; on Windows, it
requests a reparse-point handle and rejects a final reparse point.
Parent-directory traversal through directory handles remains a platform
integration boundary.
