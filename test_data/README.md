# RITK Test Data

Curated cranial imaging datasets for registration, filtering, and viewer validation.

---

## Dataset Inventory

The skull CT and paired MRI/CT collections are retained locally by their owner. Their redistribution terms are unresolved. Use the documented MRI-DIR phantom for checkout-based CT and MR workflows.

### `2_head_mri_t2/` — Head T2-Weighted MRI (DICOM, 94 slices)

| Field | Value |
|---|---|
| Modality | MR |
| Format | DICOM |
| Slices | 94 slices |
| Region | Head (porcine phantom, cranially representative) |
| Collection | MRI-DIR (TCIA) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation | Ger et al., Medical Physics 2018, DOI: 10.1002/mp.13090 |
| Purpose | MR window/level validation, registration moving image, T2 filter testing |

**Data source**: The Cancer Imaging Archive (TCIA), MRI-DIR collection.
Collection DOI: https://doi.org/10.7937/K9/TCIA.2018.3f08iejt

The MRI-DIR collection was created specifically to evaluate deformable image registration
accuracy. The porcine head phantom is implanted with 0.35 mm gold fiducial markers visible
on CT for ground-truth registration validation.

**Usage in ritk-snap**: Open `2_head_mri_t2/DICOM/` with File → Open DICOM Folder.

---

### `3_head_ct_mridir/` — Cranial CT from MRI-DIR (DICOM, 409 slices)

| Field | Value |
|---|---|
| Modality | CT |
| Format | DICOM |
| Slices | 409 axial slices |
| Voxel size | 512×512 in-plane, 0.625mm slice thickness, 0.390625mm pixel spacing |
| Region | Head/neck (porcine phantom) |
| PatientID | MRI-DIR-zzmeatphantom |
| Collection | MRI-DIR (TCIA) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation | Ger et al., Medical Physics 2018, DOI: 10.1002/mp.13090 |
| Purpose | CT↔MRI registration testing, CT window/level validation, CT-only preprocessing |

**Pairing with MRI**: This CT and the `2_head_mri_t2/` T2 MRI are from the same porcine head phantom (`MRI-DIR-zzmeatphantom`). The phantom contains implanted 0.35 mm gold fiducial markers visible on CT, providing a ground-truth reference for registration accuracy validation (target registration error measurement using fiducial centroids).

**Usage in ritk-snap**: Open `3_head_ct_mridir/DICOM/` with File → Open DICOM Folder.

---

### `registration/` — Supplementary Registration Fixtures

| Field | Value |
|---|---|
| Format | NIfTI-1, MetaImage, TIFF, and supplementary registration formats |
| Files | ANTs, RIRE, SimpleITK, and other supplementary fixtures; see `registration/README.md` |
| Region | Brain (whole head) |
| Purpose | Registration algorithm and format testing |

The canonical inter-subject registration pair is the ANTs MNI152 atlas at
`test_data/ants_example/mni152.nii.gz` and the OpenNeuro subject at
`test_data/openneuro/sub-01_T1w.nii.gz`. The `registration/` subtree contains
only supplementary fixtures and no duplicate copies of either source.

**Data sources**: see the ANTs and OpenNeuro sections above and the
source-specific manifest in `test_data/registration/README.md`.

**Usage**:
```bash
ritk-cli register syn --fixed test_data/ants_example/mni152.nii.gz \
                      --moving test_data/openneuro/sub-01_T1w.nii.gz \
                      --output registered.nii.gz
```

---

### `openneuro/` — T1-Weighted Brain MRI (NIfTI)

| Field | Value |
|---|---|
| Format | NIfTI-1 (.nii.gz) |
| File | `sub-01_T1w.nii.gz` |
| Region | Brain |
| Source | OpenNeuro ds000102, Flanker task (event-related) |
| Purpose | T1 MRI filter testing, normalization validation, MR viewer display |

**Usage in ritk-snap**: Open `openneuro/sub-01_T1w.nii.gz` with File → Open File.

---

Registration tests read this volume from `ATLAS_FIXTURE_CACHE/openneuro/sub-01_T1w.nii.gz` when that environment variable is set. The Atlas fixtures catalog verifies the 10,581,116-byte upstream object against SHA-256 `5b2b2b2dba19d5a6bbcdc39a6cd6a8e2169a8288137f4f90e75015c113070cbe` and supplies the path. Existing checkouts can use their tracked copy. [Shared catalog and resolver](https://github.com/ryancinsight/atlas-fixtures/tree/35656a6a21b0b2a5f419fa4fe8fbc4e6fc488a41)

### `ants_example/` — MNI152 Brain Atlas and Visible Human (NIfTI)

| Field | Value |
|---|---|
| Format | NIfTI-1 (.nii.gz) |
| Files | `mni152.nii.gz`, `visiblehuman.nii.gz` |
| Region | Brain (MNI152 standard space); Whole body (Visible Human) |
| Purpose | Atlas-based registration reference, multi-scale filter testing |

The MNI152 template is the standard Montreal Neurological Institute brain atlas widely used
as a registration target for cranial studies. The Visible Human dataset provides
a high-resolution anatomical reference.

---

### NIfTI Codec Format Fixtures

`ritk-nifti` uses two fixture classes so format tests have explicit provenance
without adding opaque generated binary blobs:

| Format | Fixture source | Validation |
|---|---|---|
| NIfTI-1 `.nii.gz` | `test_data/ants_example/mni152.nii.gz`, the ANTs/MNI152 fixture | `ritk-nifti` imports the real repository gzip fixture and asserts ZYX shape, spacing, voxel count, non-zero content, and finite Float32 voxels |
| NIfTI-2 `.nii` / `.nii.gz` | Deterministic synthetic images generated by `ritk-nifti` tests through `write_nifti2` / `write_nifti2_labels` | Tests assert the NIfTI-2 header variant, voxel offset, shape, spatial metadata, gzip wrapping, and exact voxel/label preservation |
| Analyze 7.5 `.hdr` / `.img` | Owned by `ritk-analyze`; not a NIfTI fixture | `ritk-nifti` rejects Analyze-style headers without NIfTI magic so Analyze pairs cannot be silently imported as NIfTI |

Paired NIfTI `ni1`/`ni2` `.hdr`/`.img` remains a separate NIfTI variant and is
not represented by Analyze 7.5 fixtures.

---

## CT + MRI Registration Testing Workflow

The `3_head_ct_mridir` (CT) and `2_head_mri_t2` (MR) pair is the **primary paired
CT↔MRI registration test case** — both volumes are from the same `MRI-DIR-zzmeatphantom`
porcine phantom with implanted gold fiducial ground truth:

1. **Load CT fixed image** from `3_head_ct_mridir/DICOM/`
2. **Load MR moving image** from `2_head_mri_t2/DICOM/`
3. **Run registration** (rigid → affine → deformable SyN)
4. **Validate** using the gold fiducial markers embedded in the MRI-DIR phantom

> **CT↔MRI parity**: The `3_head_ct_mridir/` CT and `2_head_mri_t2/` MRI are from the
> **same phantom** with **gold fiducial ground truth**, making them the preferred pair for
> quantitative registration accuracy evaluation (target registration error via fiducial
> centroids).

The canonical `ants_example/mni152.nii.gz` + `openneuro/sub-01_T1w.nii.gz` pair
provides a NIfTI-format registration workflow without DICOM I/O.

---

## Filter Testing Workflow

| Filter | Recommended Dataset | Expected Outcome |
|---|---|---|
| BedSeparation | CT volume with a patient table | Table removed, anatomy retained |
| Gaussian | Any volume | Reduced high-frequency noise |
| Median | Any volume | Salt-and-pepper noise suppression |
| N4 Bias Field | `openneuro/sub-01_T1w.nii.gz` | Intensity non-uniformity corrected |
| Anisotropic Diffusion | `3_head_ct_mridir/DICOM/` | Edge-preserving smoothing |
| Frangi Vesselness | `openneuro/sub-01_T1w.nii.gz` | Vessel enhancement |

---

## Window/Level Reference Values

| Dataset | Modality | Recommended Centre | Recommended Width |
|---|---|---|---|
| `2_head_mri_t2` | MR | 600 | 1200 |
| `3_head_ct_mridir` | CT | Brain: 40 HU | Brain: 80 HU |
| `3_head_ct_mridir` | CT | Bone: 400 HU | Bone: 1000 HU |
| `openneuro/sub-01_T1w` | MR | 500 | 800 |

---

## Licensing

| Dataset | License |
|---|---|
| `2_head_mri_t2/` (MRI-DIR) | CC BY 4.0 — cite Ger et al. 2018 |
| `3_head_ct_mridir/` (MRI-DIR) | CC BY 4.0 — cite Ger et al. 2018 |
| `registration/` | Per source repository license |
| `openneuro/` | PDDL, per the [ds000102 dataset record](https://github.com/OpenNeuroDatasets/ds000102/blob/fdc1ce49d524fc1e77e5b763f10acb02f403c581/dataset_description.json) |
| `ants_example/` | Per ANTs/MNI152 distribution terms |

Any publication using the MRI-DIR data must cite:

> Ger, R. B., Yang, J., Ding, Y., Jacobsen, M. C., Cardenas, C. E., Fuller, C. D.,
> Howell, R. M., Li, H., Stafford, R. J., Zhou, S., & Court, L. E. (2018).
> Synthetic head and neck and phantom images for determining deformable image registration
> accuracy in magnetic resonance imaging. *Medical Physics*, 45(9), 4315–4321.
> https://doi.org/10.1002/mp.13090

## Real DICOM directory-export acceptance

Run against an installed wheel built from the revision under test. Use Python
3.10 or newer with pytest, NumPy, pydicom 3, and pylibjpeg with its libjpeg
plugin. From the repository root:

```sh
python -m pip install 'pytest>=8.4.2,<10' 'pytest-timeout>=2.4,<3' \
  'pydicom>=3,<4' 'pylibjpeg>=2,<3' 'pylibjpeg-libjpeg>=2.1,<3'
RITK_DICOM_TEST_DATA="$PWD/test_data" python -m pytest -q \
  crates/ritk-python/tests/test_dicom_anonymization_real.py
```

The MRI fixture exercises RITK volume reopening. The cranial CT fixture
exercises lossless JPEG decoding with pylibjpeg, five series, and DICOMDIR
rejection at its media root. The test compares stored and decoded samples,
calibrated and windowed pixels, image geometry, selected metadata and UID
relationships. Set the environment variable to opt in. Pixel preservation
does not assess burned-in annotations or disclosure safety.
