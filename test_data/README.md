# RITK Test Data

Curated cranial imaging datasets for registration, filtering, and viewer validation.

---

## Dataset Inventory

### `2_skull_ct/` — Cranial CT (DICOM, 303 slices)

| Field | Value |
|---|---|
| Modality | CT |
| Format | DICOM (series folder + DICOMDIR) |
| Slices | 303 axial slices |
| Region | Cranium / skull |
| Purpose | Viewer display, CT window/level validation, BedSeparation filter, registration fixed image |

**Usage in ritk-snap**: Open `2_skull_ct/DICOM/` with File → Open DICOM Folder.

---

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

### `registration/` — Paired Brain NIfTI Volumes

| Field | Value |
|---|---|
| Format | NIfTI-1 (.nii.gz) |
| Files | `brain_fixed.nii.gz`, `brain_moving.nii.gz` |
| Region | Brain (whole head) |
| Purpose | Registration algorithm testing (fixed + moving image pair) |

These paired volumes are used for evaluating SyN, B-spline, LDDMM, and Demons
registration pipelines. The fixed and moving images represent the same subject
in different acquisition positions/sessions, providing ground-truth registration
correspondence.

**Usage**:
```bash
ritk-cli register syn --fixed test_data/registration/brain_fixed.nii.gz \
                      --moving test_data/registration/brain_moving.nii.gz \
                      --output registered.nii.gz
```

---

### `openneuro/` — T1-Weighted Brain MRI (NIfTI)

| Field | Value |
|---|---|
| Format | NIfTI-1 (.nii.gz) |
| File | `sub-01_T1w.nii.gz` |
| Region | Brain |
| Source | OpenNeuro (https://openneuro.org) |
| Purpose | T1 MRI filter testing, normalization validation, MR viewer display |

**Usage in ritk-snap**: Open `openneuro/sub-01_T1w.nii.gz` with File → Open File.

---

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

## CT + MRI Registration Testing Workflow

The `3_head_ct_mridir` (CT) and `2_head_mri_t2` (MR) pair is the **primary paired
CT↔MRI registration test case** — both volumes are from the same `MRI-DIR-zzmeatphantom`
porcine phantom with implanted gold fiducial ground truth:

1. **Load CT fixed image** from `3_head_ct_mridir/DICOM/`
   *(previously `2_skull_ct/DICOM/`; `3_head_ct_mridir` is now the primary paired CT dataset)*
2. **Load MR moving image** from `2_head_mri_t2/DICOM/`
3. **Run registration** (rigid → affine → deformable SyN)
4. **Validate** using the gold fiducial markers embedded in the MRI-DIR phantom

> **CT↔MRI parity**: The `3_head_ct_mridir/` CT and `2_head_mri_t2/` MRI are from the
> **same phantom** with **gold fiducial ground truth**, making them the preferred pair for
> quantitative registration accuracy evaluation (target registration error via fiducial
> centroids). The `2_skull_ct/` dataset remains available for CT-only filter and viewer
> testing.

The `registration/brain_fixed.nii.gz` + `brain_moving.nii.gz` pair provides
a NIfTI-format equivalent for registration algorithm unit testing without DICOM I/O.

---

## Filter Testing Workflow

| Filter | Recommended Dataset | Expected Outcome |
|---|---|---|
| BedSeparation | `2_skull_ct/DICOM/` | Table removed, skull/brain retained |
| Gaussian | Any volume | Reduced high-frequency noise |
| Median | Any volume | Salt-and-pepper noise suppression |
| N4 Bias Field | `openneuro/sub-01_T1w.nii.gz` | Intensity non-uniformity corrected |
| Anisotropic Diffusion | `2_skull_ct/DICOM/` | Edge-preserving smoothing |
| Frangi Vesselness | `openneuro/sub-01_T1w.nii.gz` | Vessel enhancement |

---

## Window/Level Reference Values

| Dataset | Modality | Recommended Centre | Recommended Width |
|---|---|---|---|
| `2_skull_ct` | CT | Brain: 40 HU | Brain: 80 HU |
| `2_skull_ct` | CT | Bone: 400 HU | Bone: 1000 HU |
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
| `openneuro/` | CC0 (OpenNeuro default) |
| `ants_example/` | Per ANTs/MNI152 distribution terms |

Any publication using the MRI-DIR data must cite:

> Ger, R. B., Yang, J., Ding, Y., Jacobsen, M. C., Cardenas, C. E., Fuller, C. D.,
> Howell, R. M., Li, H., Stafford, R. J., Zhou, S., & Court, L. E. (2018).
> Synthetic head and neck and phantom images for determining deformable image registration
> accuracy in magnetic resonance imaging. *Medical Physics*, 45(9), 4315–4321.
> https://doi.org/10.1002/mp.13090