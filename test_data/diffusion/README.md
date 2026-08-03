# Diffusion MRI Test Data

Public CC0-licensed diffusion-weighted MRI datasets downloaded from OpenNeuro
for end-to-end pipeline integration testing.

---

## Download

```bash
bash test_data/diffusion/download.sh
```

The script is idempotent — re-running it skips files that already exist.

Requirements:
- `curl` (pre-installed on macOS, Linux, and Windows Git Bash)
- Or: [AWS CLI](https://aws.amazon.com/cli/) (`aws s3 cp --no-sign-request`)

---

## Dataset Inventory

### `ds002087/` — Single-Subject DWI with Deliberate Head Motion

| Field | Value |
|---|---|
| Title | MRI Datasets with and without deliberate head movements for... |
| Dataset ID | `ds002087` |
| Modality | Diffusion-weighted MRI (DWI) |
| Subject | sub-01 |
| Volumes | ≈33 (1 b0 + ~32 DWI directions) |
| Format | NIfTI-1 (.nii.gz) + FSL bval/bvec |
| License | CC0 (Creative Commons Zero v1.0 Universal) |
| Size | ≈15–30 MB (DWI files only) |
| Source | https://openneuro.org/datasets/ds002087 |
| Purpose | Real-data DTI/DKI/CSD/NODDI pipeline integration testing |

**Files downloaded** (placed directly in `test_data/diffusion/`):

| File | Description |
|---|---|
| `sub-01_dwi.nii.gz` | 4-D DWI volume (X × Y × Z × 33) |
| `sub-01_dwi.bval` | b-values (s/mm²), one per volume |
| `sub-01_dwi.bvec` | Gradient directions (3 rows × 33 columns), unit vectors |
| `sub-01_dwi.json` | BIDS sidecar with acquisition parameters |
| `dataset_description.json` | Dataset-level metadata |

---

## Usage

### Integration test (requires downloaded data)

```bash
# Download the dataset first
bash test_data/diffusion/download.sh

# Run the real-data integration test
cargo test -p ritk-diffusion --test integration_real_data -- --ignored
```

The integration test:
1. Reads `sub-01_dwi.nii.gz` through `ritk-nifti`'s native reader
2. Parses `sub-01_dwi.bval` / `sub-01_dwi.bvec` into a `GradientScheme`
3. Runs DTI, DKI, CSD, and NODDI on a central brain slice
4. Asserts model outputs are physically plausible (FA ∈ [0,1], MD ∈ [0, 0.004], etc.)
5. Verifies that the gradient-scheme codec (FSL) round-trips losslessly

### Manual CLI usage

```bash
# Convert DWI to NRRD
ritk-cli convert test_data/diffusion/sub-01_dwi.nii.gz dwi.nrrd
```

---

## Licensing

| Dataset | License |
|---|---|
| `ds002087` | CC0 (Creative Commons Zero v1.0 Universal) — public domain dedication |

No attribution is required for CC0 data, but the source dataset can be cited as:

> OpenNeuro. (2021). ds002087 — MRI Datasets with and without deliberate head
> movements for... https://openneuro.org/datasets/ds002087

---

## Adding More Datasets

To add a new diffusion dataset:

1. Choose a CC0 dataset from https://openneuro.org/search (filter: Modality → MRI → Diffusion, License → CC0)
2. Add its S3 source URLs to `download.sh`
3. Document the dataset above following the existing table format
4. Add a corresponding `#[ignore]` integration test in `crates/ritk-diffusion/tests/integration_real_data.rs`
