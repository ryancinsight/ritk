# RITK Registration Test Data

Test datasets for side-by-side RITK vs SimpleITK registration validation.

## Directory Structure

### brain_mni/ — Inter-subject T1 brain registration

Primary registration test pair. Both images are skull-stripped T1-weighted brain volumes suitable for affine and deformable registration testing.

| File | Source | Shape | Spacing | Dtype | Size |
|------|--------|-------|---------|-------|------|
| mni152.nii.gz | ANTs example data (MNI152 atlas) | (215, 256, 207) | (0.74, 0.74, 0.74) mm | float32 | 4.1 MB |
| sub-01_T1w.nii.gz | OpenNeuro sub-01 T1w | (256, 256, 176) | (1.0, 1.0, 1.0) mm | int16 | 10.1 MB |
| r16slice.nii.gz | ANTs r16 test slice | (256, 256) | (1.0, 1.0) mm | float32 | 65 KB |
| r64slice.nii.gz | ANTs r64 test slice | (256, 256) | (1.0, 1.0) mm | uint8 | 18 KB |

Registration pair: `mni152.nii.gz` (atlas) ↔ `sub-01_T1w.nii.gz` (subject)

### rire/ — RIRE retrospective image registration evaluation

Multi-modal (CT ↔ MR T1) registration pair from the Retrospective Image Registration Evaluation project. These are 3D volumes with known ground-truth registration available in the RIRE dataset.

| File | Source | Shape | Spacing | Dtype | Size |
|------|--------|-------|---------|-------|------|
| training_001_ct.mha | RIRE patient 001 CT | (29, 512, 512) | (0.65, 0.65, 4.0) mm | int16 | 14.5 MB |
| training_001_mr_T1.mha | RIRE patient 001 MR T1 | (26, 256, 256) | (1.25, 1.25, 4.0) mm | int16 | 3.2 MB |

Registration pair: `training_001_ct.mha` (CT) ↔ `training_001_mr_T1.mha` (MR T1)

Source: SimpleITK notebook data (via Kitware SHA512 hash store)
S3 URL: `https://s3.amazonaws.com/simpleitk/public/notebooks/SHA512/<hash>`

### simpleitk_notebooks/ — SimpleITK tutorial head images

Head CT and MRI volumes from the Visible Male dataset, used in SimpleITK registration tutorials.

| File | Source | Shape | Spacing | Dtype | Size |
|------|--------|-------|---------|-------|------|
| vm_head_ct.mha | Visible Male CT head | (8, 512, 512) | (0.49, 0.49, 1.0) mm | int16 | 1.5 MB |
| vm_head_mri.mha | Visible Male MRI head | (33, 256, 256) | (1.02, 1.02, 5.0) mm | int16 | 1.8 MB |
| head_mr_oriented.mha | Oriented MR head (thin slab) | (3, 256, 256) | (0.86, 0.86, 1.6) mm | int16 | 386 KB |

Registration pair: `vm_head_ct.mha` (CT) ↔ `vm_head_mri.mha` (MRI)

Source: SimpleITK-Notebooks Data manifest
S3 URL: `https://s3.amazonaws.com/simpleitk/public/notebooks/SHA512/<hash>`

### learnitk/ — ITK tutorial registration pair

B1/B2 brain TIFF image pair from the ITK registration tutorial. Same-subject, different-acquisition brain slices used for intra-subject registration demonstration.

| File | Source | Shape | Mode | Size |
|------|--------|-------|------|------|
| B1.tiff | ITK tutorial (fixed) | (197, 254) | 8-bit grayscale | 49 KB |
| B2.tiff | ITK tutorial (moving) | (197, 254) | 8-bit grayscale | 49 KB |

Registration pair: `B1.tiff` (fixed) ↔ `B2.tiff` (moving)

Source: SimpleITK-Notebooks Data manifest (via Kitware SHA512 hash store)

### synthetic_shifted/ — Synthetic test pairs

No downloads required. Synthetic test pairs (translated, rotated, deformed versions of existing images) are generated at runtime by the Python test script.

### emsocket/ — EM-SOCRATES registration benchmark

**Status: Unavailable** — PUMA repository (https://github.com/rii-mango/PUMA) returned HTTP 404. Data not accessible at the referenced URL.

### mrbrains/ — MRBrainS18 multi-modal

**Status: Requires authentication** — MRBrainS18 (https://mrbrains18.isi.uu.nl/) requires login for data access. Skipped per spec.

### cachalot/ — Cachalot registration benchmark

**Status: Unavailable** — Insight Journal MIDAS Cachalot page (https://www.insight-journal.org/midas/cachalot/) returned HTTP 404. The Insight Journal has been restructured and the legacy MIDAS data is no longer available at the original URL.

### ixi/ — IXI dataset from NITRC

**Status: Failed** — Download from NITRC (https://www.nitrc.org/frs/download.php/853/IXI-T1.tar.gz) returned a 7.2 MB ZIP file that did not contain NIfTI images — it contained ANTs utility executables (ARCTIC, CortThickCLP, ImageMath, etc.), not IXI brain data. The NITRC download link appears to be mislabeled or redirected. Deleted.

## Acquisition Details

### Successful Downloads

| Dataset | Method | Endpoint |
|---------|--------|----------|
| RIRE CT/MR | curl | `https://s3.amazonaws.com/simpleitk/public/notebooks/SHA512/<sha512>` |
| vm_head_mri | curl | S3 SHA512 endpoint |
| vm_head_ct | curl | `https://data.kitware.com/api/v1/file/hashsum/sha512/<sha512>/download` |
| head_mr_oriented | curl | S3 SHA512 endpoint |
| B1/B2 TIFF | curl | S3 SHA512 endpoint |
| r16/r64 slice | curl | Kitware SHA512 endpoint |
| mni152 | local copy | from `test_data/ants_example/mni152.nii.gz` |
| sub-01_T1w | local copy | from `test_data/openneuro/sub-01_T1w.nii.gz` |

### Failed Sources

| Target | URL | Error | Reason |
|--------|-----|-------|--------|
| SimpleITK-Notebooks GitHub raw | `github.com/.../raw/master/Data/*.nii.gz` | HTTP 404 | Data not in repo tree; uses hash-based external storage |
| PUMA/EM-SOCRATES | `github.com/rii-mango/PUMA/tree/master/data` | HTTP 404 | Repository does not exist |
| Kitware LearnITK blobs | `data.kitware.com/api/v1/file/5e5d0ffd...` | HTTP 404 | Invalid item ID format |
| IXI NITRC | `nitrc.org/frs/download.php/853/IXI-T1.tar.gz` | Wrong content | File contained ANTs executables, not brain data |
| RIRE Insight Journal | `insight-journal.org/rire/download_data.php` | HTTP 404 | Legacy page removed during site restructure |
| Cachalot Insight Journal | `insight-journal.org/midas/cachalot/` | HTTP 404 | Legacy MIDAS data no longer hosted |
| MRBrainS18 | `mrbrains18.isi.uu.nl` | Skipped | Requires authentication |
| S3 MD5-based | `s3.amazonaws.com/.../MD5/<hash>` | HTTP 403 | Only SHA512 hash storage exists |
| vm_head_ct via S3 SHA512 | S3 endpoint | HTTP 403 | SHA512 hash was only in min_manifest (md5-only) |

### Download Method Notes

- SimpleITK-Notebooks data uses a content-addressable storage system: files are stored by SHA512 hash on S3 (`https://s3.amazonaws.com/simpleitk/public/notebooks/SHA512/<sha512>`) and as a fallback on Kitware (`https://data.kitware.com/api/v1/file/hashsum/sha512/<sha512>/download`).
- The `manifest.json` (SHA512-based) is the authoritative source; `min_manifest.json` and `brains_manifest.json` use md5sum which is not supported by the S3 storage.
- The S3 endpoint sometimes returns HTTP 403 even for valid hashes; the Kitware SHA512 endpoint is a more reliable fallback.
- For `vm_head_ct.mha`, the SHA512 hash was not in `manifest.json` but was discoverable via the Kitware API: `https://data.kitware.com/api/v1/item?text=vm_head_ct` → item → files → sha512.
