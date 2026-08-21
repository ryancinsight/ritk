# RITK — Rust Image Toolkit

A high-performance medical image processing and registration toolkit built in
Rust, inspired by ITK concepts and integrated with the Coeus and Leto compute
stack.

## Documentation

- [RITK medical imaging book](https://ryancinsight.github.io/ritk/) — hosted
  GitHub Pages mdBook with algorithms, runnable Rust examples, and filter and
  registration figure examples.
- [Book source](docs/book/) — Markdown chapters, source-linked examples, and
  figure-generation commands.

## Overview

RITK provides a comprehensive framework for medical image analysis:

- **Backend-Parametric Compute**: Coeus tensor and autograd contracts execute
  over Leto-owned storage; current RITK entry points use deterministic
  sequential or Moirai-parallel CPU backends
- **Deep Module Hierarchy**: Strict DIP/SSOT/SoC/SRP architecture across workspace crates
- **Broad Format Support**: DICOM, NIfTI, MetaImage, NRRD, PNG, TIFF/BigTIFF, MGH/MGZ, VTK, JPEG
- **Classical & Deformable Registration**: Rigid, affine, B-Spline FFD, Demons, SyN, LDDMM, Atlas/Groupwise
- **Deep-Learning Registration**: TransMorph and SSMMorph through Coeus
  autodiff
- **Image Processing Pipeline**: Filtering, segmentation, statistics, normalization
- **CT Visualization Support**: Bed separation filter for CT foreground/body masking
- **Native DICOM Viewer**: `ritk-snap` desktop viewer with DICOM folder/DICOMDIR launch, MPR viewports, overlays, measurements, and tag inspection
- **Python Bindings**: PyO3 + maturin with NumPy bridge, packaged type stubs, and `py.typed`
- **CLI**: `ritk` binary with `convert`, `filter`, `register`, `segment`, and `stats` subcommands

## Crate Structure

| Layer | Crates | Responsibility |
|---|---|---|
| Domain contracts | `ritk-spatial`, `ritk-image`, `ritk-transform`, `ritk-interpolation`, `ritk-annotation` | Physical coordinates, typed images, transforms, interpolation, and annotation state |
| Operations | `ritk-filter`, `ritk-segmentation`, `ritk-morphology`, `ritk-statistics`, `ritk-tensor-ops` | Image algorithms and shared Coeus host-buffer operations |
| Registration | `ritk-registration`, `ritk-model` | Classical, deformable, differentiable, and learned registration |
| Diffusion & tractography | `ritk-diffusion-scheme`, `ritk-diffusion`, `ritk-tractography`, `ritk-connectome` | Validated acquisition schemes, DTI/DKI/NODDI/Q-ball/CSD models, streamline tracking, and parcellation graph measures |
| Format owners | `ritk-dicom`, `ritk-codecs`, `ritk-nifti`, `ritk-nrrd`, `ritk-metaimage`, `ritk-mgh`, `ritk-analyze`, `ritk-png`, `ritk-jpeg`, `ritk-tiff`, `ritk-minc`, `ritk-vtk`, `ritk-mif` | Validated byte-level codecs and format-specific image I/O |
| Tractogram formats | `ritk-tck`, `ritk-trk`, `ritk-trx` | MRtrix3 `.tck`, TrackVis `.trk`, and TRX streamline I/O |
| Integration | `ritk-io`, `ritk-core`, `ritk-wgpu-compat` | Unified I/O dispatch, public facade contracts, and graphics interop |
| Deliverables | `ritk-cli`, `ritk-snap`, `ritk-python` | CLI, native viewer, and thin PyO3 bindings |

All 38 workspace crates appear above; `xtask` is the build-automation member and
is not a library crate.

Dependencies point inward toward domain contracts. Format crates own byte-level
parsing, `ritk-io` owns cross-format dispatch, and applications and bindings
depend on those contracts without moving domain logic into their boundaries.

Provider ownership follows the same hierarchy. Each provider repository owns
its implementation, Atlas owns repository checkout revisions, and RITK owns
only its versioned consumption contracts. The root manifest maps transitive
provider Git sources onto the same sibling path packages used directly, so one
crate identity carries each trait and type through the graph. This prevents
parallel provider types, makes superseded consumer code deletable, and leaves
`Cargo.lock` as a projection of one source graph rather than a second topology.

Domain-neutral normalized color laws and fixed lookup-table construction come
from the public [Iris](https://github.com/ryancinsight/iris) provider.
`ritk-snap` retains medical windowing, UI state, and GPU resource ownership;
`ritk-vtk` retains VTK mapping and serialization. Both consume the same Iris
`NamedColorMap` contract without local color interpolation.

### Viewer (`ritk-snap`)

`ritk-snap [PATH]` launches the native viewer directly against a DICOM folder,
a single DICOM file, a `DICOMDIR` file, or a supported medical image file. The viewer keeps DICOM
I/O in `ritk-io` and presentation logic in `ritk-snap`, with a vertical module
split for input path normalization, hanging-protocol selection, series
discovery, metadata row construction, session snapshot persistence, rendering,
tools, and egui widgets.

Current viewer capabilities include DICOM series browsing, axial/coronal/
sagittal MPR layout, modality-aware window presets, colormaps, measurement and
ROI tools, interactive segmentation label paint/erase with brush radius,
label visibility/active-label controls with undo/redo, viewport label overlays,
load-time hanging-protocol defaults for CT/MR series, linked MPR cursor
navigation across all three planes, DICOM-style patient-orientation labels,
linked-cursor HU overlay readout, linked-cursor physical LPS readout,
active-axis cine playback with FPS control, Ctrl/Cmd+scroll viewport zoom,
Ctrl/Cmd+0 zoom-to-fit,
Arrow Up/Down and Page Up/Down active-axis slice navigation,
Home/End active-axis first/last slice navigation,
tool keyboard shortcuts (L=length, A=angle, R=rect ROI, E=ellipse ROI, H=HU point, P=pan, Z=zoom, W=window/level, B=paint),
Zoom tool continuous drag zoom,
Segmentation keyboard undo/redo shortcuts,
Pan tool drag mapping (additive viewport offset) via SSOT `ui/pan`,
ROI Ellipse true pixel-mask statistics (ellipse membership test `((r−cy)/a)²+((c−cx)/b)²≤1`) via `Annotation::compute_roi_ellipse_stats`,
W/L drag mapping (horizontal width, vertical center) via SSOT `ui/window_level`,
PNG slice export, full axial/coronal/sagittal MPR PNG export, DICOM overlays,
RT-STRUCT contour overlay loading and rendering,
and a
deterministic Tags panel covering series metadata, first-slice geometry/display
tags, private scalar tags, preserved object-model nodes, and raw preserved
element byte counts. Viewer session save/load stores presentation state as
JSON, including source path, slice indices, window/level, colormap, active
tool, layout flags, overlay flags, sidebar tab, pan, and zoom.

### Browser / WASM (egui)

`ritk-snap` now exposes a wasm entrypoint for browser hosting:

- `ritk_snap::start_web(canvas_id: String)` (wasm-only, exported via `wasm-bindgen`)

The native binary (`ritk-snap`) remains desktop-only. For browser execution,
build `crates/ritk-snap` for `wasm32-unknown-unknown`, load the generated JS/WASM
bundle in a page with a `<canvas>` element, and invoke `start_web("<canvas-id>")`.

Minimal JS bootstrap pattern:

```javascript
import init, { start_web } from "./pkg/ritk_snap.js";

await init();
await start_web("ritk-snap-canvas");
```

## Features

### Core contracts

**Spatial types** — `Point<D>`, `Vector<D>`, `Spacing<D>`, `Direction<D>` built on
`leto::FixedVector` / `leto::FixedMatrix` (`ritk-spatial`).

**Image** — `ritk_image::Image<T, B, D>` carries typed Coeus storage plus
origin, spacing, and direction metadata with index-to-physical and
physical-to-index transforms.

**Transforms**

| Transform | Description |
|---|---|
| Translation | Pure translation |
| Rigid | Rotation + translation |
| Affine | Full affine (12 DOF in 3-D) |
| Scale | Axis-aligned scaling |
| Versor | Unit-quaternion rotation (3-D) |
| BSpline | Free-form deformation on a control-point lattice |
| DisplacementField | Dense voxel-wise displacement |
| ChainedTransform | Sequential composition of transforms |
| CompositeTransform | Named composite with JSON serialization |

**Interpolation**

| Method | Notes |
|---|---|
| Linear | Supports 1-D through 4-D |
| Nearest Neighbor | Supports 1-D through 4-D |
| BSpline | Cubic B-spline kernel |
| Tensor Trilinear | Optimized separable 3-D path |

**Filters**

| Category | Algorithms |
|---|---|
| Smoothing | Gaussian, Recursive Gaussian (Deriche), Median, Bilateral |
| Diffusion | Anisotropic Diffusion (Perona–Malik) |
| Edge Detection | Gradient Magnitude, Laplacian, Sobel, Canny, Laplacian of Gaussian |
| Vesselness | Frangi (Hessian-based) |
| Morphology | Grayscale Erosion, Grayscale Dilation |
| Bias Correction | N4 Bias Field Correction (B-spline fitting) |
| Resampling | Downsample, Resample, Multi-Resolution Pyramid |
| Distance Transform | Euclidean Distance Transform (Meijster et al. 2000) |

**Segmentation**

| Category | Algorithms |
|---|---|
| Thresholding | Otsu, Multi-Otsu, Li, Yen, Kapur, Triangle |
| Binary Morphology | Erosion, Dilation, Opening, Closing, Skeletonization, Fill Holes, Morphological Gradient |
| Labeling | Connected Components (Hoshen–Kopelman) |
| Region Growing | Connected threshold, Confidence connected, Neighborhood connected |
| Clustering | K-Means |
| Watershed | Marker-controlled watershed |
| Level Sets | Chan–Vese, Geodesic Active Contour, Shape Detection, Threshold Level Set, Laplacian Level Set |

**Statistics & Normalization**

| Category | Functions |
|---|---|
| Descriptive | Min, Max, Mean, Variance, Percentile (masked support) |
| Comparison | Dice, Hausdorff Distance, Mean Surface Distance, PSNR, SSIM |
| Normalization | Min-Max, Z-Score, Histogram Matching, Nyúl–Udupa, White Stripe (Shinohara 2014) |
| Noise | MAD-based noise estimation |

### I/O (`ritk-io`)

| Format | Read | Write |
|---|---|---|
| DICOM (series) | ✓ | ✓ |
| NIfTI (.nii/.nii.gz) | ✓ | ✓ |
| MetaImage (.mha/.mhd) | ✓ | ✓ |
| NRRD | ✓ | ✓ |
| PNG | ✓ | ✓ |
| TIFF / BigTIFF | ✓ | ✓ |
| MGH / MGZ (FreeSurfer) | ✓ | ✓ |
| VTK legacy structured points (`.vtk`) | ✓ | ✓ |
| JPEG (`.jpg`, `.jpeg`) | ✓ | ✓* |

*JPEG write support is limited to 2-D grayscale images represented in RITK as shape `[1, height, width]`.

`ritk-dicom` owns DICOM transfer-syntax classification and native pixel-codec
primitives. Native Rust decode covers uncompressed little-endian pixels, RLE
Lossless, grayscale JPEG Baseline/Extended/Lossless, grayscale JPEG-LS, and
JPEG 2000 fragments. Pure-Rust pixel decode additionally covers JPEG XL through
`jxl-oxide`. `dicom-rs` remains the dataset, metadata, and external-codec
adapter; no supported DICOM pixel path requires a C or C++ codec library.

#### DICOM de-identification and export verification

`ritk-io` ships a PS 3.15 Annex E de-identification toolset with an
export-time metadata integrity gate — the piece that stops corrupt or leaking
metadata from silently shipping to a destination PACS:

- `AnonymizationProfile` — `Basic`, `BasicReplaceUids`, `Aggressive`, and
  `Enhanced` profiles covering every Annex E Table E.1-1 attribute, with
  deterministic SHA-256 UID remapping (referentially consistent within a
  batch, irreversible without the salt).
- `anonymize_dicom_file` / `anonymize_dicom_directory` — single-object and
  directory-batch de-identification with per-run statistics and optional
  pixel-data / private-tag cleaning.
- `anonymize::verify` — the export gate:
  - re-parse check (the exported file must open as a conformant Part 10 object),
  - UID presence, DICOM-conformant format, and cross-file Study/Series
    consistency with unique SOPInstanceUIDs,
  - geometry coherence (`Rows × Columns × Samples × BytesPerPixel` vs.
    PixelData length),
  - prohibited-value leak scan for the exact identifiers the pipeline scrubbed.
- `anonymize_dicom_file_verified` / `anonymize_dicom_directory_verified` —
  run anonymization and the export gate as one operation, failing closed on
  any defect.

See `crates/ritk-io/examples/anonymize_pacs_export.rs` for the complete
PACS-bound export pipeline.


### Registration (`ritk-registration`)

**Metrics** — MSE, Mutual Information (Standard / Mattes / NMI), NCC, LNCC, Correlation Ratio, DL losses.

**Optimizers** — Gradient Descent, Adam, Momentum, CMA-ES.

**Regularization** — Bending Energy, Curvature, Diffusion, Elastic, Total Variation.

**Registration Algorithms**

| Algorithm | Category |
|---|---|
| Kabsch SVD | Classical rigid alignment |
| MI-based rigid/affine | Classical iterative |
| Thirion Demons | Deformable |
| Diffeomorphic Demons | Deformable |
| Symmetric Demons | Deformable |
| Greedy SyN | Diffeomorphic |
| Multi-Resolution SyN | Diffeomorphic |
| BSpline SyN | Diffeomorphic |
| BSpline FFD | Deformable |
| LDDMM | Diffeomorphic |
| Groupwise Atlas | Template building (iterative SyN) |
| Joint Label Fusion | Multi-atlas segmentation (Wang 2013) |
| Majority Voting | Multi-atlas label fusion baseline |

### Deep-Learning Models (`ritk-model`)

| Model | Description |
|---|---|
| TransMorph | Transformer-based deformable registration |
| SSMMorph | Statistical shape model registration |

### Python Bindings (`ritk-python`)

PyO3 + maturin package exposing:

- Filters including Gaussian, median, bilateral, Canny, Frangi, and N4
- Segmentation spanning Otsu methods, morphology, connected components,
  watershed, and level sets
- Classical and deformable registration, atlas construction, and label fusion
- Descriptive statistics, similarity metrics, noise estimation, and
  normalization
- Validated NumPy ↔ `Image` conversion at the PyO3 ownership boundary
- Composite transform JSON I/O (`read_transform`, `write_transform`)
- Packaged `.pyi` type stubs and `py.typed`
- Format I/O for all supported formats

### CLI (`ritk-cli`)

```
ritk convert   <input> <output>          # Format conversion
ritk viewer    <input> [opts]            # Inspect a DICOM study using the viewer core
ritk filter    <input> <output> [opts]   # Apply filters
ritk register  <fixed> <moving> [opts]   # Run registration
ritk segment   <input> <output> [opts]   # Run segmentation
ritk stats     --input <path> [opts]     # Summary and comparison metrics
ritk resample  <input> <output> [opts]   # Resample to a new voxel spacing
ritk normalize <input> <output> [opts]   # Normalize intensities (histogram-match, nyul, zscore, minmax, white-stripe)
ritk dwi       <subcommand> [opts]       # Diffusion-weighted image processing
ritk tract     <subcommand> [opts]       # Streamline tractography
ritk parcellate <subcommand> [opts]      # Label a brain by anatomical region
```

`ritk dwi tensor` fits one diffusion tensor per voxel from a DWI series and its
FSL sidecars, and writes the scalar maps requested:

```
ritk dwi tensor --dwi sub-01_dwi.nii.gz \
                --bval sub-01_dwi.bval --bvec sub-01_dwi.bvec \
                --fa fa.nii.gz --md md.nii.gz --ad ad.nii.gz --rd rd.nii.gz
```

Voxels below `--background-fraction` of the b = 0 signal's upper percentile are
not fitted, because a tensor fitted to noise is strongly anisotropic and would
otherwise trace a bright rim around the skull. Fits outside the physical
diffusivity bounds are rejected rather than written; see
`ritk_diffusion::maps::DiffusionMapsConfig` for the derivations.

`ritk tract dti` fits the same field and tracks streamlines through it, writing
MRtrix `.tck` in the image's physical frame:

```
ritk tract dti --dwi sub-01_dwi.nii.gz \
               --bval sub-01_dwi.bval --bvec sub-01_dwi.bvec \
               --output tracks.tck
```

Streamlines are short by the standards of a tuned pipeline — on a single-shell
b = 700 acquisition the median track runs about 16 mm against an anatomical
30–150 mm. That is the data and the nearest-neighbour direction lookup, not a
threshold to loosen; the command reports why each track stopped so the two can
be told apart.

`ritk parcellate atlas` labels a subject by registering one or more labelled
atlases onto it and fusing their votes, which is what turns a tractogram into a
connectome — the streamline endpoints need regions to land in:

```
ritk parcellate atlas --subject sub-01_T1w.nii.gz \
                      --atlas-intensity atlas1.nii.gz --atlas-labels atlas1_dseg.nii.gz \
                      --atlas-intensity atlas2.nii.gz --atlas-labels atlas2_dseg.nii.gz \
                      --output sub-01_dseg.nii.gz --agreement agreement.nii.gz

ritk tract connectome --tractogram tracks.tck --labels sub-01_dseg.nii.gz \
                      --output matrix.json --measures measures.json
```

Every atlas must already lie on the subject's grid; a registration recovers a
deformation, never a resampling, and the command rejects a mismatch rather than
producing labels for a differently sized brain. `--agreement` writes the
fraction of atlases that voted for each winning label, which is low exactly at
the parcel boundaries where streamline endpoints land — so it is the map to
consult before trusting an edge weight.

Current `ritk segment --method` coverage includes:

- Thresholding: `otsu`, `multi-otsu`, `li`, `yen`, `kapur`, `triangle`
- Region / labeling: `connected-threshold`, `connected-components`
- Morphology: `fill-holes`, `morphological-gradient`, `skeletonization`
- Region growing: `confidence-connected`, `neighborhood-connected`
- Clustering / topology: `kmeans`, `watershed`, `distance-transform`
- Level sets: `chan-vese`, `geodesic-active-contour`, `shape-detection`, `threshold-level-set`, `laplacian-level-set`

Selected method-specific options:

- `connected-components`: `--connectivity`
- `chan-vese`: `--mu`, `--nu`, `--lambda1`, `--lambda2`, `--epsilon`
- `geodesic-active-contour`: `--initial-phi`, `--propagation-weight`, `--curvature-weight`, `--advection-weight`, `--edge-k`, `--sigma`, `--dt`, `--level-set-max-iterations`
- `shape-detection`: `--initial-phi`, `--propagation-weight`, `--curvature-weight`, `--advection-weight`, `--edge-k`, `--sigma`, `--dt`, `--level-set-max-iterations`, `--tolerance`
- `threshold-level-set`: `--initial-phi`, `--lower-threshold`, `--upper-threshold`, `--propagation-weight`, `--curvature-weight`, `--dt`, `--level-set-max-iterations`, `--tolerance`
- `confidence-connected`: `--seed`, `--multiplier`, `--max-iterations`
- `neighborhood-connected`: `--seed`, `--lower`, `--upper`, `--neighborhood-radius`

## Usage Example

```rust,no_run
use ritk_io::{read_image_native, write_image_native};

fn main() -> anyhow::Result<()> {
    let image = read_image_native("input.nii.gz")?;
    write_image_native("roundtrip.nrrd", &image)?;
    Ok(())
}
```

## Dependencies

First-party Atlas stack providers:

| Crate | Role |
|---|---|
| `coeus-core` / `-tensor` / `-ops` / `-leto` / `-autograd` / `-nn` / `-optim` | Tensor, backend, and autodiff contracts |
| `leto` / `leto-ops` | Array storage, linear algebra, and numerical operations |
| `eunomia` | Scalar and numeric trait vocabulary |
| `aequitas` | Statistical distributions and estimators |
| `apollo-fft` / `apollo-sht` | FFT and spherical-harmonic transforms |
| `moirai` | CPU parallelism and task execution |
| `mnemosyne` | Workspace allocator |
| `consus-hdf5` / `-core` / `-compression` / `-io` / `-onnx` | Pure-Rust HDF5 (MINC), compression, and ONNX parsing |
| `gaia` | Polyline and mesh geometry for tractography |
| `iris` | Normalized color laws and lookup-table construction |

Third-party:

| Crate | Role |
|---|---|
| `dicom` (`dicom-rs`) | DICOM dataset, metadata, and external-codec adapter |
| `tiff` / `image` | Pixel codecs behind `ritk-tiff` and `ritk-png` |
| `jpeg-decoder` / `zune-jpegxl` | JPEG and JPEG XL pixel decode |
| `wgpu` / `eframe` / `egui` | Viewer rendering and graphics interop |
| `onnx-ir` | ONNX graph import for DL registration |
| `pyo3` / `numpy` | Python bindings |
| `clap` | CLI argument parsing |
| `serde` / `serde_json` | Serialization (transform I/O) |
| `anyhow` / `thiserror` | Error handling |
| `tracing` | Structured logging |

Header, geometry, and metadata parsing is first-party throughout: `ritk-nifti`,
`ritk-nrrd`, `ritk-metaimage`, `ritk-mgh`, `ritk-analyze`, `ritk-minc`,
`ritk-vtk`, `ritk-mif`, and `ritk-codecs` own their byte-level readers and
writers. No supported path requires a C or C++ library.

## Building

```bash
# Build all crates (release)
cargo build --release

# Run all tests through the committed nextest profile
cargo nextest run --workspace

# Build Python extension / install into current environment
cd crates/ritk-python && maturin develop --release

# Install CLI
cargo install --path crates/ritk-cli
```

Hosted workflows check out RITK at `ritk/` and let Cargo resolve the Atlas
providers directly from their `git + version` sources. RITK declares no sibling
path dependencies and no `[patch]` sections, so provider URLs and revisions have
a single home in `Cargo.toml` and `Cargo.lock` with no second RITK-owned list.

### Rust package distribution

RITK publishes its 29 reusable Rust library packages to crates.io in local
dependency order. Publishing a GitHub Release whose tag is
`crate-<package>-v<version>` triggers the `Crates.io Release` workflow, which
verifies the packaged source and uses crates.io trusted publishing to obtain a
short-lived credential. The matching GitHub Release is the source and artifact
record for that package version.

Nine workspace members carry `publish = false` and are not crates.io packages:
`ritk-cli`, `ritk-snap`, `ritk-python`, `ritk-diffusion`, `ritk-tractography`,
`ritk-connectome`, `ritk-tck`, `ritk-trk`, and `ritk-trx`. Python wheels use
the separate `v<version>` maturin release workflow.

Two provider packages carry a registry name that differs from their import
path, because crates.io reserves `iris` and the name `gaia` belongs to an
unrelated crate. They are published as `iris-viz` and `gaia-mesh`, and the
workspace maps them back with Cargo's `package` key, so every `use iris::…`
and `use gaia::…` in RITK is unchanged. A consumer depending on those
providers directly needs the same mapping.

## Development

### Release history

Release history and per-package changes live in [CHANGELOG.md](CHANGELOG.md).

## Testing

```bash
# Native test suite through the committed time budgets
cargo nextest run --workspace --lib --tests

# Documentation tests
cargo test --workspace --doc

# Focused package test suites
cargo nextest run -p ritk-core
cargo nextest run -p ritk-io
cargo nextest run -p ritk-registration
cargo nextest run -p ritk-model
```

## Future Work

- [ ] Sinc interpolation
- [x] MINC format reader/writer (via [consus](https://github.com/ryancinsight/consus) pure-Rust HDF5)
- [x] Analyze format reader/writer
- [ ] Diffeomorphic Demons exact inverse
- [x] Curvature anisotropic diffusion (Alvarez et al. 1992)
- [x] Sato line / Hessian blob detection (Sato 1998)
- [x] Confidence / neighborhood connected region growing
- [x] Skeletonization (hole filling remains)
- [ ] Longitudinal analysis pipeline
- [ ] WGSL/compute-shader kernels for critical filters

- [x] ONNX model import for DL registration (RITK 0.20.1+, `onnx-ir` parsing, initializers, graph validation)
- [ ] Expand Python bindings to cover model inference
- [ ] Publish to crates.io and PyPI

## License

RITK is dual-licensed under either of the following, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

## Contributing

Contributions are welcome. Requirements:

- Follow the existing deep-hierarchy architecture (DIP, SSOT, SoC, SRP)
- Include tests with analytically derived expected values
- Update documentation alongside implementation changes
- No namespace bleeding or unnecessary wrapper types

## Acknowledgments

- Inspired by [ITK](https://itk.org/) (Insight Segmentation and Registration Toolkit)
- Uses Coeus and Leto for tensor, linear-algebra, and numerical execution
