# ritk-python

PyO3/maturin Python extension module wrapping the RITK Rust crate.

## Requirements

- Rust toolchain: `nightly-x86_64-pc-windows-gnu` (default) or `nightly-x86_64-pc-windows-msvc`
- Python ≥ 3.9
- maturin ≥ 1.7

## Build (Windows)

The default toolchain (`nightly-x86_64-pc-windows-gnu`) produces a wheel whose `_ritk.dll`
links against MinGW runtime libraries (`libgcc_s_seh-1.dll`, `libstdc++-6.dll`,
`libwinpthread-1.dll`).  Windows-native Python (CPython 3.9+, MSVC ABI) cannot load these
DLLs via the default DLL search path.  The `--auditwheel repair` flag copies them into a
`ritk.libs/` directory inside the wheel, which maturin patches into the DLL search path at
import time.

**Correct build command:**
```sh
rustup run nightly-x86_64-pc-windows-msvc py -m maturin build --release --auditwheel repair \
  -i "C:\Users\<USERNAME>\AppData\Local\Programs\Python\Python313\python.exe"
```

Then install the built wheel:
```sh
py -m pip install target/wheels/ritk-0.1.0-cp39-abi3-win_amd64.whl --force-reinstall
```

## Running Tests

```sh
# VTK parity tests (requires vtk >= 9.6, SimpleITK >= 2.5)
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
delegates to `ritk-core` (zero business logic in the binding layer).  The `PyImage` wrapper
holds an `Arc<Image<NdArray<f32>, 3>>` and provides `.to_numpy()` for zero-copy-where-possible
extraction.

### DICOM I/O

`ritk.io.read_image(path)` dispatches to `ritk_io::read_dicom_series` when `path` is a
directory, enabling transparent DICOM series loading.  Pass the DICOM series directory
directly; the reader selects the first series UID via `scan_dicom_directory`.

### Extension Points

| Abstraction | Mechanism | Adding a target |
|---|---|---|
| Compute backend | `ComputeBackend` trait | `impl ComputeBackend` for new device; no algorithm changes |
| GUI backend | `GuiBackend` trait | `impl GuiBackend` for new shell; no domain logic changes |
| Execution policy | GAT-based `ExecutionPolicy` | `impl ExecutionPolicy` for new regime |
| Scalar type | `Scalar` trait | `impl Scalar` for new numeric type |

All variation dimensions are encoded through traits, generics, associated types, and const
generics.  No algorithm is cloned per backend, precision, layout, or execution regime.