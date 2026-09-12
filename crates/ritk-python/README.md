# ritk-python

PyO3/maturin Python extension module wrapping the RITK Rust crate.

## Requirements

- Rust toolchain 1.97.0 (the repository pin)
- Python ≥ 3.9
- maturin ≥ 1.9.4, < 2.0
- NumPy ≥ 2.0.2, < 2.6

Install the parity-test dependencies from their canonical manifest:

```sh
py -m pip install -r crates/ritk-python/requirements-test.txt
```

## Build

Build the stable-ABI wheel from the repository root:

```sh
python -m pip install "maturin>=1.9.4,<2.0"
python -m maturin build --release --locked \
  --manifest-path crates/ritk-python/Cargo.toml --out dist
```

Install the wheel selected for the current interpreter:

```sh
python -m pip install --force-reinstall --no-index --find-links dist ritk
```

The release workflow builds the `abi3-py39` wheel matrix and publishes through
GitHub Actions OIDC Trusted Publishing. It stores no PyPI token or private key.

## Running Tests

```sh
# VTK parity tests
py -m pytest crates/ritk-python/tests/test_vtk_parity.py -v

# SimpleITK parity tests (requires installed ritk wheel + SimpleITK)
py -m pytest crates/ritk-python/tests/test_simpleitk_parity.py -v

# CT/MRI DICOM parity tests (requires MRI-DIR test data in test_data/)
py -m pytest crates/ritk-python/tests/test_ct_mri_registration_parity.py -v

# All Python parity tests
py -m pytest crates/ritk-python/tests/ -v
```

## Module API

| Submodule | Key functions |
|---|---|
| `ritk.filter` | `discrete_gaussian`, `median_filter`, `bilateral_filter`, `gradient_magnitude`, `laplacian`, `sobel_gradient`, `n4_bias_correct`, `anisotropic_diffuse`, `frangi_vesselness` |
| `ritk.registration` | `demons_register`, `diffeomorphic_demons_register`, `syn_register`, `bspline_ffd_register`, `multires_syn_register`, `bspline_syn_register`, `lddmm_register`, `build_atlas`, `joint_label_fusion_py` |
| `ritk.segmentation` | `otsu_threshold`, `li_threshold`, `multi_otsu_threshold`, `binary_threshold`, `connected_components`, `binary_fill_holes`, `binary_erode`, `binary_dilate`, `kmeans_segment` |
| `ritk.statistics` | `compute_statistics`, `masked_statistics`, `histogram_match`, `minmax_normalize`, `zscore_normalize`, `psnr`, `ssim`, `dice_coefficient`, `hausdorff_distance` |
| `ritk.io` | `read_image`, `write_image`, `read_transform`, `write_transform` |

## Architecture

The extension module (`_ritk.cdylib`) is compiled from `src/lib.rs` and registered as
submodules `filter`, `registration`, `segmentation`, `statistics`, and `io`.  All computation
delegates to Rust crates (zero business logic in the binding layer).  The image I/O boundary
uses `ritk-io`'s native readers/writers and converts only at the `PyImage` boundary.  The
`PyImage` wrapper holds an `Arc<ritk_image::Image<f32, MoiraiBackend, 3>>` over Coeus/Leto
storage and provides `.to_numpy()` for zero-copy-where-possible extraction.

### DICOM I/O

`ritk.io.read_image(path)` dispatches to `ritk_io::read_image_native`, which routes DICOM
directories through the native DICOM series reader before extension inference. A directory with
one image series can be passed directly. A mixed directory must name the acquisition explicitly:

```python
image = ritk.io.read_image(
    "test_data/3_head_ct_mridir/DICOM",
    series_instance_uid="1.3.6.1.4.1.14519.5.2.1.1706.4996.115936088547498980797393821518",
)
```

The UID is matched by RITK's scanner before pixels are decoded. Omitting it for a mixed directory,
passing an unknown or empty UID, or passing a non-directory with a UID fails with `IOError`; no
first-series fallback is used. Obtain the UID from RITK's series discovery or the DICOM metadata.

### Native image I/O coverage

Python image reads use the Atlas-native path for NIfTI, MetaImage, NRRD, PNG, DICOM
directories, MGH/MGZ, TIFF, JPEG, and Analyze.  Python image writes use the Atlas-native path
for NIfTI, MetaImage, NRRD, MGH/MGZ, TIFF, JPEG, and Analyze.  PNG, DICOM, and VTK image writes
are rejected until native writers exist; VTK image reads are rejected until the VTK image reader
migrates to the native substrate.

### Extension Points

| Abstraction | Mechanism | Adding a target |
|---|---|---|
| Compute backend | `ComputeBackend` trait | `impl ComputeBackend` for new device; no algorithm changes |
| GUI backend | `GuiBackend` trait | `impl GuiBackend` for new shell; no domain logic changes |
| Execution policy | GAT-based `ExecutionPolicy` | `impl ExecutionPolicy` for new regime |
| Scalar type | `Scalar` trait | `impl Scalar` for new numeric type |

All variation dimensions are encoded through traits, generics, associated types, and const
generics.  No algorithm is cloned per backend, precision, layout, or execution regime.
