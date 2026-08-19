# Diffusion MRI Test Data

Public human diffusion MRI acquisitions used for end-to-end integration tests
and the book's tractography/connectomics workflow. Imaging bytes remain
gitignored; this directory records provenance, checksums, and reproducible
download commands.

## Download

```bash
bash test_data/diffusion/download.sh
```

The script is idempotent. It checks Stanford files against their published MD5
digests and distinguishes OpenNeuro imaging data from git-annex pointer files
by size.

Requirements:

- `git`
- `curl`
- `md5sum`, `md5`, or `certutil`

## Dataset inventory

### Stanford HARDI — whole-brain tractography and connectomics

| Field | Value |
|---|---|
| Repository record | [`yx282xq2090`](https://purl.stanford.edu/yx282xq2090) |
| Modality | Human high-angular-resolution diffusion MRI |
| Acquisition | 150 directions at \(b=2000\) s/mm² plus 10 \(b=0\) volumes |
| Image shape | 81 × 106 × 76 voxels |
| Preprocessing | Motion-corrected to the mean \(b=0\); no eddy-current correction |
| Anatomy | DWI-aligned reduced FreeSurfer parcellation and label table |
| License | Open Data Commons PDDL 1.0 public-domain dedication |
| Size | About 92 MB |
| Purpose | Human whole-brain tensor fitting, deterministic tracking, endpoint connectome, and book figure |

Files are placed in `test_data/diffusion/stanford_hardi/`:

| File | MD5 |
|---|---|
| `dwi.nii.gz` | `0b18513b46132b4d1051ed3364f2acbc` |
| `dwi.bvals` | `4e08ee9e2b1d2ec3fddb68c70ae23c36` |
| `dwi.bvecs` | `4c63a586f29afc6a48a5809524a76cb4` |
| `aparc-reduced.nii.gz` | `742de90090d06e687ce486f680f6d71a` |
| `label_info.txt` | `39db9f0f5e173d7a2c2e51b07d5d711b` |

The repository's use conditions prohibit attempts to identify participants or
otherwise infringe their privacy. The data are suitable for method
development, not subject identification or clinical inference.

### `ds002087` — DWI with deliberate head motion

| Field | Value |
|---|---|
| Source | [OpenNeuro `ds002087`](https://openneuro.org/datasets/ds002087) |
| Subject | `sub-01` |
| Acquisition | 99 volumes at \(b=0\) and \(b=700\) s/mm² |
| Image shape | 104 × 104 × 72 voxels at 2 mm isotropic |
| License | CC0 1.0 public-domain dedication |
| Size | About 55 MB for the DWI volume |
| Purpose | Real-data DTI, DKI, CSD, and NODDI integration coverage |

The script shallow-clones the OpenNeuro GitHub mirror for text sidecars and
fetches the DWI volume directly from public S3 because the clone contains a
git-annex pointer rather than imaging bytes.

### `ds004666` — EDDEN denoising acquisition

| Field | Value |
|---|---|
| Source | [OpenNeuro `ds004666`](https://openneuro.org/datasets/ds004666) |
| Acquisition | 199 multi-shell volumes at approximately \(b=1000\) and \(b=2000\) s/mm² |
| Resolutions | 0.9, 1.5, and 2.0 mm isotropic |
| License | CC0 1.0 public-domain dedication |
| Purpose | Real gradient-scheme coverage for multi-shell models |

## Usage

Run the ignored real-data integration target after downloading:

```bash
cargo nextest run -p ritk-diffusion --test integration_real_data \
  --run-ignored ignored-only
```

Regenerate the human tractography/connectome artifacts:

```bash
cargo run --release -p ritk-diffusion --example book_brain_tractography
```

The example writes `docs/book/figures/brain_tractography.svg` and the complete
upper-triangular streamline-count matrix at
`docs/book/figures/brain_connectome.json`.
