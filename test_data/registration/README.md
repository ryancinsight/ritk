# RITK Registration Test Data

Test datasets for side-by-side RITK vs SimpleITK registration validation.

## Directory Structure

### brain_mni/ — Inter-subject T1 brain registration

Primary registration test directory. Contains multiple T1 brain volumes for same-modality and inter-subject registration testing.

| File | Source | Shape | Spacing | Dtype | Size |
|------|--------|-------|---------|-------|------|
| mni152.nii.gz | ANTs example data (MNI152 atlas) | (207, 256, 215) | (0.74, 0.74, 0.74) mm | float32 | 4.1 MB |
| sub-01_T1w.nii.gz | OpenNeuro sub-01 T1w | (176, 256, 256) | (1.0, 1.0, 1.0) mm | int16 | 10.6 MB |
| ants_ch2.nii.gz | ANTs Colin27 average T1 | (181, 217, 181) | (1.0, 1.0, 1.0) mm | float32 | 1.7 MB |
| ants_mni.nii.gz | ANTs ICBM MNI template | (182, 218, 182) | (1.0, 1.0, 1.0) mm | float32 | 4.1 MB |
| ants_surf.nii.gz | ANTs cortical surface | (266, 266, 190) | (0.94, 0.94, 1.20) mm | float32 | 273 KB |
| single_subj_T1.nii.gz | SPM12 canonical T1 | (91, 109, 91) | (2.0, 2.0, 2.0) mm | float32 | 472 KB |
| ants_r16.nii.gz | ANTs r16 2D slice | (256, 256) | (1.0, 1.0) mm | float32 | 26 KB |
| ants_r27.nii.gz | ANTs r27 2D slice | (256, 256) | (1.0, 1.0) mm | float32 | 28 KB |
| ants_r30.nii.gz | ANTs r30 2D slice | (256, 256) | (1.0, 1.0) mm | float32 | 27 KB |
| ants_r62.nii.gz | ANTs r62 2D slice | (256, 256) | (1.0, 1.0) mm | float32 | 28 KB |
| ants_r64.nii.gz | ANTs r64 2D slice | (256, 256) | (1.0, 1.0) mm | float32 | 29 KB |
| ants_r85.nii.gz | ANTs r85 2D slice | (256, 256) | (1.0, 1.0) mm | float32 | 28 KB |

Registration pairs:
- `ants_ch2.nii.gz` (Colin27) ↔ `ants_mni.nii.gz` (ICBM MNI) — same-modality, roughly pre-aligned, NCC_before ≈ 0.7-0.9
- `mni152.nii.gz` (atlas) ↔ `sub-01_T1w.nii.gz` (subject) — inter-subject, NCC_before ≈ 0.04
- `ants_r*.nii.gz` — 2D slices only, not suitable for 3D registration

Note: `brain_fixed.nii.gz` and `brain_moving.nii.gz` in the parent directory are **byte-identical** copies of `mni152.nii.gz` (NCC=1.0, MSE=0.0). They are NOT a meaningful registration pair.

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

### rire/ — RIRE retrospective image registration evaluation (updated)

Multi-modal (CT ↔ MR T1) registration pair from the Retrospective Image Registration Evaluation project.

| File | Source | Shape | Spacing | Dtype | Size |
|------|--------|-------|---------|-------|------|
| training_001_ct.mha | RIRE patient 001 CT | (29, 512, 512) | (0.65, 0.65, 4.0) mm | int16 | 14.5 MB |
| training_001_mr_T1.mha | RIRE patient 001 MR T1 | (26, 256, 256) | (1.25, 1.25, 4.0) mm | int16 | 3.2 MB |
| ct_T1.standard | RIRE fiducial ground truth (8 points) | — | — | — | 1 KB |
| training_001_ct_to_mr_T1_ground_truth.tfm | SimpleITK Euler3D ground-truth transform | — | — | — | 1 KB |
| ground_truth_registration.md | Full documentation with metadata and verification | — | — | — | 5 KB |

Ground-truth transform (CT → MR T1, Euler3D, center=(0,0,0)):
- Rotation: 4.440° Z, 1.901° X, 0.043° Y
- Translation: [5.04, -17.50, -27.16] mm
- Validation: 8 RIRE corner-point pairs reproduce with max residual 0.000176 mm

Registration pair: `training_001_ct.mha` (CT) ↔ `training_001_mr_T1.mha` (MR T1)

### emsocket/ — EM-SOCRATES registration benchmark

**Status: Unavailable** — PUMA repository (https://github.com/rii-mango/PUMA) returned HTTP 404.

### mrbrains/ — MRBrainS18 multi-modal

**Status: Requires authentication** — MRBrainS18 (https://mrbrains18.isi.uu.nl/) requires login.

### cachalot/ — Cachalot registration benchmark

**Status: Unavailable** — Insight Journal MIDAS Cachalot page returned HTTP 404.

### ixi/ — OpenNeuro ds000208 inter-subject T1w pair

Same-modality (T1↔T1) inter-subject registration pair from OpenNeuro ds000208.

| File | Source | Shape | Spacing | Dtype | Size |
|------|--------|-------|---------|-------|------|
| openneuro_ds000208_sub01_T1w.nii.gz | OpenNeuro ds000208 sub-01 T1w | (256, 256, 160) | (1.0, 1.0, 1.0) mm | float32 | 7.7 MB |
| openneuro_ds000208_sub02_T1w.nii.gz | OpenNeuro ds000208 sub-02 T1w | (256, 256, 160) | (1.0, 1.0, 1.0) mm | float32 | 8.4 MB |
| openneuro_ds000208_sub03_T1w.nii.gz | OpenNeuro ds000208 sub-03 T1w | (256, 256, 160) | (1.0, 1.0, 1.0) mm | float32 | 10.0 MB |

Registration pair: `openneuro_ds000208_sub01_T1w.nii.gz` ↔ `openneuro_ds000208_sub02_T1w.nii.gz`
Baseline NCC ≈ 0.75 (genuine inter-subject variability)

Source: OpenNeuro S3 bucket (s3.amazonaws.com/openneuro.org/ds000208/...)

### ixi/ — IXI dataset from NITRC (deprecated)

**Status: Failed** — Download from NITRC returned a 7.2 MB ZIP file that did not contain NIfTI images — it contained ANTs utility executables, not IXI brain data. Deleted.

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
