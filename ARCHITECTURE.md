# RITK Architecture Specification

## Table of Contents

1. [Design Principles](#design-principles)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Module Hierarchy](#module-hierarchy)
4. [Algorithm Specifications](#algorithm-specifications)
5. [Component Consolidation](#component-consolidation)
6. [Testing Architecture](#testing-architecture)

---

## Design Principles

### 1. Single Responsibility Principle (SRP)

> **Theorem 1.1 (SRP Invariant)**: For any module M, the cardinality of its responsibility set |R(M)| = 1.

**Formal Definition**:
```
∀M ∈ Modules : ∃! r : r ∈ Responsibilities ∧ M implements r
```

**Application in RITK**:
- `ritk-core::spatial` - Pure geometric operations only
- `ritk-core::transform` - Coordinate transformations only
- `ritk-registration::metric` - Similarity metrics only
- `ritk-registration::optimizer` - Optimization algorithms only

### 2. Separation of Concerns (SOC)

> **Theorem 2.1 (SOC Partitioning)**: ∀m₁, m₂ ∈ Modules : Concerns(m₁) ∩ Concerns(m₂) = ∅

**Implementation**:
```
ritk-core/
├── spatial/     # Geometric primitives
├── image/       # Image data structures
├── transform/   # Spatial transformations
└── interpolation/ # Sampling algorithms
```

### 3. Single Source of Truth (SSOT)

> **Theorem 3.1 (SSOT Consistency)**: ∀type T, |{Source(T)}| = 1

**Evidence**:
- `Point<D>`: Defined exclusively in `spatial/point.rs`
- `Vector<D>`: Defined exclusively in `spatial/vector.rs`
- `ImageMetadata<D>`: Defined exclusively in `image/metadata.rs`

### 4. Don't Repeat Yourself (DRY)

> **Theorem 4.1 (DRY Factorization)**: ∀f,g ∈ Functions : f ≈ g ⇒ ∃h : f = h ∘ α ∧ g = h ∘ β

**Consolidation Strategy**:
- Shared tensor operations → `burn` framework abstractions
- Common spatial math → `nalgebra` direct usage
- IO patterns → Trait-based abstraction layer

### 5. Dependency Inversion Principle (DIP)

> **Theorem 5.1 (DIP Abstraction)**: High-level modules depend on abstractions, not concretions

```
High-Level (Registration)
    ↓ depends on
Transform Trait (Abstraction)
    ↓ implemented by
Concrete Transforms (Translation, Rigid, Affine, BSpline)
```

### 6. DICOM Backend Boundary

> **Theorem 6.1 (DICOM Backend Isolation)**: DICOM file parsing and pixel-frame decode enter `ritk-io` through a `ritk-dicom` trait boundary.

**Boundary surface**:
- `DicomParseBackend`: parses a Part 10 file into a backend-owned object.
- `PixelDecodeBackend`: decodes one frame from a backend-owned object using `DecodeFrameRequest`.
- `DicomBackend`: combines parse and decode without dynamic dispatch.
- `DicomRsBackend`: current temporary implementation backed by `dicom-rs`.

**Replacement invariant**:
Native codec replacement changes codec internals behind `ritk-codecs` / `NativeCodecBackend`; DICOM readers continue to call `decode_frame_with::<DicomRsBackend>` until the parser backend is replaced.

**Codec ownership invariant**:
`ritk-codecs` owns JPEG, JPEG-LS, JPEG 2000, RLE, PackBits, and native pixel primitive implementations. `ritk-dicom::codec` may re-export those primitives and dispatch by transfer syntax, but must not retain copied codec bodies. Native-owned JPEG syntaxes selected by `TransferSyntaxKind::is_native_jpeg_codec()` route exclusively through `NativeCodecBackend`; external backend fallback is limited to `TransferSyntaxKind::is_external_backend_codec_candidate()`.

> **Theorem 6.2 (JPEG 2000 Backend Substitution)**: JPEG 2000 dependency replacement preserves the DICOM frame-decode contract when the decoded component stream is the same ordered integer sample sequence consumed by the DICOM modality LUT.

**Boundary surface**:
- `ritk-codecs::jpeg_2000` owns `decode_jpeg2000_fragment(fragment, PixelLayout) -> Vec<f32>`.
- Production decode uses `jpeg2k::Image` with the `openjp2` Rust backend, not `openjpeg-sys`.
- `image::extract_pixels` consumes `ImageComponent::data()` as raw `i32` component samples; it validates component count, dimensions, precision, and signedness before applying `output = stored_integer × slope + intercept`.
- `ritk-dicom::NativeCodecBackend` remains the only DICOM transfer-syntax dispatch point for JPEG 2000 Lossless/Lossy, so parser ownership and codec implementation ownership stay separated.

**Proof obligation**:
For any DICOM JPEG 2000 frame `C`, layout `L`, and backend `B`, if `B(C)` yields component planes `S₀..Sₙ` whose concatenated ordered samples equal the ISO 15444-1 decoded sample sequence for `C`, then `decode_jpeg2000_fragment(C,L)[i] = S[i] × L.rescale_slope + L.rescale_intercept`. Backend replacement is therefore behavior-preserving at the DICOM boundary when component metadata validation passes.

> **Theorem 6.3 (JPEG Backend Static Boundary)**: DICOM JPEG dependency replacement preserves frame-decode behavior when each backend yields the same validated raster metadata and integer sample stream.

**Boundary surface**:
- `ritk-codecs::jpeg` owns `decode_jpeg_fragment(fragment, PixelLayout) -> Vec<f32>`.
- `ritk-codecs::jpeg::backend::JpegDecodeBackend` is sealed and uses static dispatch; there is no `dyn` codec dispatch in the DICOM JPEG path.
- `JpegDecoderCrate` is the current ZST implementation backed by `jpeg-decoder`.
- `JpegPixelFormat::L16` samples use the backend's native-endian byte contract; conversion to signed or unsigned DICOM stored integers happens only after `PixelLayout` validation.
- `JpegPixelFormat::Rgb24` maps to interleaved RGB samples with `samples_per_pixel=3`, `BitsAllocated=8`, and unsigned sample interpretation.
- `PixelLayout` owns integer sample interpretation for all native codecs; `BitsAllocated=8` with `PixelRepresentation=1` maps each byte through `i8`, not `u8`.

**Proof obligation**:
For any DICOM JPEG frame `C`, layout `L`, and backend `B`, if `B(C)` yields dimensions `W,H`, pixel format `F`, and ordered sample bytes `S` equal to the decoded JPEG raster under the backend byte contract, then `decode_jpeg_fragment(C,L)` either rejects `(W,H,F,S)` when it conflicts with `L`, or returns `stored_integer(S[i]) × L.rescale_slope + L.rescale_intercept`. Backend replacement is behavior-preserving when the replacement satisfies the same raster and byte-order contract.

> **Theorem 6.4 (Scalar DICOM Volume Boundary)**: A scalar 3-D DICOM volume loader must reject color sample layouts before tensor construction.

**Boundary surface**:
- `ritk-io::format::dicom::reader::read_slice_pixels` decodes only scalar series slices with `SamplesPerPixel=1`.
- `ritk-io::format::dicom::load_dicom_multiframe` decodes only scalar multiframe objects with `SamplesPerPixel=1`.
- RGB JPEG frames remain decodable through `ritk-codecs` / `ritk-dicom`; scalar `Image<B,3>` loaders do not collapse or drop color channels.

**Proof obligation**:
For scalar tensor shape `[depth, rows, cols]`, each frame contributes exactly `rows × cols` samples. If a DICOM object declares `SamplesPerPixel = k ≠ 1`, a decoded frame contains `rows × cols × k` samples and cannot be represented in the scalar tensor without either channel loss or shape ambiguity. The loader must reject before constructing `Image<B,3>`.

> **Theorem 6.5 (JPEG-LS Lossless Native Boundary)**: JPEG-LS Lossless transfer syntax `.80` must route through RITK-native decode before any external backend fallback.

**Boundary surface**:
- `ritk-codecs::jpeg_ls` owns JPEG-LS marker parsing, run-mode and regular-mode scan decode, and DICOM modality LUT application.
- The SOS header fields are parsed as `NEAR`, `ILV`, and point transform; prediction is the ISO adaptive JPEG-LS predictor, not a DICOM-specific SOS selector.
- JPEG-LS entropy decode implements bit stuffing, not byte stuffing: after an encoded `0xFF` data byte, exactly one stuffed zero bit is discarded and the remaining seven bits of the following byte remain entropy data.
- JPEG-LS scan decode maintains the line-left guard equivalent to CharLS `current_line[-1]`; at column 0, `Rc` is the previous line's guard, not `Rb`.
- `ritk-dicom::DicomRsBackend` delegates `TransferSyntaxKind::JpegLsLossless` to `NativeCodecBackend`; JPEG-LS Near-Lossless remains an external backend candidate.
- DICOM UI padding bytes are stripped by `TransferSyntaxKind::from_uid` before transfer-syntax classification, so padded file-meta UIDs cannot bypass native codec dispatch.

**Proof obligation**:
For any JPEG-LS Lossless frame `C` with `NEAR=0`, `ILV=0`, one component, and layout `L`, native decode reconstructs each stored sample from the ISO 14495-1 run/regular contexts, entropy bit-stuffing rules, and causal line guards, then returns `stored_integer × L.rescale_slope + L.rescale_intercept`. A third-party lossless encoder fixture is admissible only when the same encoded bytes self-decode to the asserted source samples under the reference implementation.

**Structural invariant**:
- `ritk-codecs::jpeg_ls::marker` owns JPEG-LS marker constants.
- `ritk-codecs::jpeg_ls::parser` owns marker traversal and scan-data discovery.
- `ritk-codecs::jpeg_ls::decoder` owns header-derived decoder state, scan precondition validation, and scan-to-native-byte conversion.
- `ritk-codecs::jpeg_ls::scan`, `context`, and `bitstream` remain the ISO scan, context, and entropy subdomains.
- `ritk-codecs::jpeg_ls::tests` partitions conformance, parser, and decoder-state tests by contract family.

### 7. NIfTI Spatial Boundary

> **Theorem 7.1 (NIfTI Axis-Affine Consistency)**: NIfTI voxel payload axis conversion and affine metadata conversion must apply the same file-axis to internal-axis permutation.

**Boundary surface**:
- `crates/ritk-nifti/src/spatial.rs` owns RAS↔LPS row conversion and NIfTI `[x,y,z]`↔RITK `[depth,row,col]` affine-column mapping.
- Reader invariant: after NIfTI file data `[x,y,z]` becomes RITK tensor data `[depth,row,col]`, internal metadata columns are derived from file affine columns `[z,y,x]`.
- Writer invariant: NIfTI sform columns are emitted as `[internal_col, internal_row, internal_depth]`, and `pixdim[1..=3]` is `[dx,dy,dz] = [spacing[2], spacing[1], spacing[0]]`.
- `ritk-io::format::nifti` is a facade re-export; it must not contain a parallel NIfTI implementation.

**Replacement invariant**:
NIfTI parser/writer dependency changes stay behind `ritk-nifti`; callers in `ritk-io`, CLI, and viewer code consume the same authoritative API.

### 8. NRRD Spatial Boundary

> **Theorem 8.1 (NRRD Payload-Affine Axis Consistency)**: NRRD raw payload order and spatial metadata conversion must apply the same file-axis to internal-axis mapping.

**Boundary surface**:
- `crates/ritk-nrrd/src/spatial.rs` owns NRRD `[x,y,z]` file-axis ↔ RITK `[depth,row,col]` spatial metadata conversion.
- Reader invariant: NRRD raw payload bytes are X-fastest, which is identical to RITK `[depth,row,col]` flat order when shaped as `[nz,ny,nx]`; no tensor permutation is applied.
- Reader metadata invariant: `space directions` vectors `[x,y,z]` become internal metadata columns `[depth,row,col] = [z,y,x]`; scalar `spacings` follow the same reorder with axis-aligned directions.
- Writer invariant: RITK ZYX flat payload data is emitted directly, and NRRD `space directions` are generated from internal columns `[col,row,depth]`.
- `ritk-io::format::nrrd` is a facade re-export; it must not contain a parallel NRRD implementation.

**Replacement invariant**:
NRRD parser/writer dependency changes stay behind `ritk-nrrd`; callers in `ritk-io`, CLI, and viewer code consume the same authoritative API.

### 9. MetaImage Spatial Boundary

> **Theorem 9.1 (MetaImage Payload-Affine Axis Consistency)**: MetaImage raw payload order and spatial metadata conversion must apply the same file-axis to internal-axis mapping.

**Boundary surface**:
- `crates/ritk-metaimage/src/spatial.rs` owns MetaImage `[x,y,z]` file-axis ↔ RITK `[depth,row,col]` spatial metadata conversion.
- Reader invariant: MetaImage raw payload bytes are X-fastest, which is identical to RITK `[depth,row,col]` flat order when shaped as `[nz,ny,nx]`; no tensor permutation is applied.
- Reader metadata invariant: `ElementSpacing` values `[x,y,z]` become internal spacing `[depth,row,col] = [z,y,x]`, and `TransformMatrix` file columns `[x,y,z]` become internal direction columns `[col,row,depth]`.
- Writer invariant: RITK ZYX flat payload data is emitted directly, `ElementSpacing` is emitted as `[spacing[col], spacing[row], spacing[depth]]`, and `TransformMatrix` file columns are generated from internal columns `[col,row,depth]`.
- `ritk-io::format::metaimage` is a facade re-export; it must not contain a parallel MetaImage implementation.

**Replacement invariant**:
MetaImage parser/writer dependency changes stay behind `ritk-metaimage`; callers in `ritk-io`, CLI, and viewer code consume the same authoritative API.

### 10. PNG Format Boundary

> **Theorem 10.1 (PNG Series Ownership)**: PNG single-slice and directory-series parsing have exactly one implementation body owned by `ritk-png`.

**Boundary surface**:
- `ritk-png` owns `read_png_to_image`, `read_png_series`, `PngReader<B>`, and `PngSeriesReader<B>`.
- `ritk-png` owns `read_png_color_to_volume`, `read_png_color_series`, `PngColorReader<B>`, and `PngColorSeriesReader<B>`.
- Reader invariant: grayscale pixels decode into `Image<B, 3>` with tensor shape `[1, height, width]` for a single PNG and `[slice_count, height, width]` for a series.
- RGB reader invariant: decoded `Rgb8` pixels decode into `RgbVolume<B>` with tensor shape `[1, height, width, 3]` for a single PNG and `[slice_count, height, width, 3]` for a series.
- Metadata invariant: PNG carries no physical-space metadata, so origin is `[0,0,0]`, spacing is `[1,1,1]`, and direction is identity.
- Series invariant: directory slices are ordered by deterministic natural filename order and dimension mismatches are rejected before tensor construction.
- `ritk-io::format::png` is a facade re-export plus local `ImageReader` adapters only.

### 11. JPEG Format Boundary

> **Theorem 11.1 (JPEG 2D Ownership)**: JPEG grayscale and RGB file parsing have exactly one implementation body owned by `ritk-jpeg`.

**Boundary surface**:
- `ritk-jpeg` owns `read_jpeg`, `write_jpeg`, `JpegReader<B>`, and `JpegWriter<B>`.
- `ritk-jpeg` owns `read_jpeg_color_to_volume` and `JpegColorReader<B>`.
- Reader invariant: decoded JPEG Luma8 pixels become `Image<B, 3>` with tensor shape `[1, height, width]`.
- RGB reader invariant: decoded JPEG `Rgb8` pixels become `RgbVolume<B>` with tensor shape `[1, height, width, 3]`.
- Writer invariant: input `Image<B, 3>` must have `nz == 1`; values are rounded, clamped to `[0,255]`, and encoded as 8-bit grayscale.
- Metadata invariant: JPEG carries no physical-space metadata, so origin is `[0,0,0]`, spacing is `[1,1,1]`, and direction is identity.
- `ritk-io::format::jpeg` is a facade re-export plus local `ImageReader` / `ImageWriter` adapters only.

### 12. TIFF Format Boundary

> **Theorem 12.1 (TIFF Stack Ownership)**: TIFF / BigTIFF image-stack parsing and writing have exactly one implementation body owned by `ritk-tiff`.

**Boundary surface**:
- `ritk-tiff` owns `read_tiff`, `write_tiff`, `TiffReader<B>`, and `TiffWriter`.
- `ritk-tiff` owns `read_tiff_color_to_volume` and `TiffColorReader<B>`.
- Reader invariant: TIFF pages decode into `Image<B, 3>` with tensor shape `[page_count, height, width]`; a single-page TIFF has depth 1.
- RGB reader invariant: TIFF RGB pages decode into `RgbVolume<B>` with tensor shape `[page_count, height, width, 3]`.
- Writer invariant: `Image<B, 3>` is emitted as a page stack with one page per depth slice.
- `ritk-io::format::tiff` is a facade re-export plus local `ImageReader` / `ImageWriter` adapters only.

### 13. MINC Format Boundary

> **Theorem 13.1 (MINC2 HDF5 Ownership)**: MINC2 HDF5 parsing and writing have exactly one implementation body owned by `ritk-minc`.

**Boundary surface**:
- `ritk-minc` owns `read_minc`, `write_minc`, `MincReader<B>`, and `MincWriter`.
- Reader invariant: MINC2 dimension metadata, `dimorder`, voxel datatype conversion, and spatial metadata derivation are isolated in `ritk-minc`.
- Writer invariant: RITK tensor data is emitted as contiguous little-endian `f32` voxel bytes in the MINC2 HDF5 layout.
- `ritk-io::format::minc` is a facade re-export plus local `ImageReader` / `ImageWriter` adapters only.

### 14. Format Facade Monomorphization Boundary

> **Theorem 14.1 (Single Implementation Ownership)**: A format with a dedicated crate has exactly one parser/writer implementation body; `ritk-io` may expose only static re-exports and trait adapters.

**Boundary surface**:
- `ritk-analyze`, `ritk-jpeg`, `ritk-metaimage`, `ritk-mgh`, `ritk-minc`, `ritk-nifti`, `ritk-nrrd`, `ritk-png`, `ritk-tiff`, and `ritk-vtk` own their format parsers and writers.
- `ritk-io::format::*` modules for those crates are facade boundaries. They re-export the authoritative functions and define only local `ImageReader` / `ImageWriter` adapters when orphan rules require those impls to live in `ritk-io`.
- Adapter types remain generic over `B: Backend`; calls monomorphize per backend and do not use dynamic dispatch in throughput paths.
- Copied reader/writer files under `ritk-io` for dedicated-crate formats are prohibited.

**Verification invariant**:
Implementation tests live with the owning format crate. `ritk-io` tests only facade-level behavior and trait-adapter wiring.

**MGH / MGZ structural invariant**:
- `ritk-mgh::reader` owns path handling, gzip selection, header decode, scalar voxel conversion, and `MghReader` delegation.
- `ritk-mgh::writer` owns path handling, gzip emission, header encode, f32 voxel byte emission, and `MghWriter` delegation.
- `ritk-mgh::binary` owns big-endian primitive I/O.
- `ritk-mgh::types` owns MGH scalar type byte-width validation.
- `ritk-mgh::spatial` owns the inverse RAS transforms `origin = c_ras - Mdc*D*h` and `c_ras = origin + Mdc*D*h`.
- Reader/writer tests are partitioned by contract family; crafted binary fixtures and image construction live in crate-local `test_support`.

### 15. PET/CT Fusion Display Boundary

> **Theorem 15.1 (PET Display Value Consistency)**: A fused PET/CT renderer must window PET samples in SUVbw display units, not raw activity concentration units, when PET acquisition metadata is available.

**Boundary surface**:
- `ritk-snap::render::fusion::render_fused_slice` is the SSOT for primary/secondary fused slice composition.
- `ritk-snap::dicom::pet::PetAcquisitionParams` is the SSOT that maps a loaded PT volume to SUVbw parameters.
- The fusion renderer applies a per-volume display transform before window-level mapping: PT with complete PET metadata maps `Bq/mL -> SUVbw`; all other volumes use raw modality values.
- `ritk-snap::dicom::hanging_protocol::select_hanging_protocol` selects the PT SUV whole-body default window (`center=3`, `width=6`).

**Proof obligation**:
For any PET voxel value `p` in Bq/mL, patient mass `m_kg`, injected dose `d_bq`, and decay factor `k` derived from acquisition timing, `render_fused_slice` maps `p` to `p * (m_kg * 1000.0) / (d_bq * k)` before applying the PT SUV window and colormap. With secondary alpha `1.0`, the fused pixel equals the PET colormap output for that SUV value; with incomplete PET metadata, the renderer preserves the prior raw-value contract.

### 16. Color Volume Boundary

> **Theorem 16.1 (Color Volume Shape Separation)**: Multi-component image volumes must use a channel-explicit tensor boundary and must not enter scalar `Image<B,3>` loaders.

**Boundary surface**:
- `ritk-core::image::ColorVolume<B, C>` is the SSOT for channel-explicit 3-D volumes, backed by tensor shape `[depth, rows, cols, C]`.
- `ritk-core::image::RgbVolume<B>` is the `C = 3` specialization for interleaved RGB volume data.
- `ritk-io::format::dicom::read_dicom_color_series` loads validated interleaved RGB DICOM series into `RgbVolume<B>` while preserving spatial metadata from the scalar DICOM series scanner.
- `ritk-io::format::dicom::read_dicom_color_multiframe` loads validated interleaved RGB DICOM multiframe objects into `RgbVolume<B>` while preserving multiframe origin, spacing, and direction metadata.
- `ritk-png::read_png_color_to_volume` and `ritk-png::read_png_color_series` load only decoded `Rgb8` PNG inputs into `RgbVolume<B>` with default PNG spatial metadata.
- `ritk-jpeg::read_jpeg_color_to_volume` loads only decoded `Rgb8` JPEG inputs into `RgbVolume<B>` with default JPEG spatial metadata.
- `ritk-tiff::read_tiff_color_to_volume` loads only TIFF `ColorType::RGB(_)` page stacks into `RgbVolume<B>` with default TIFF spatial metadata.
- Scalar DICOM series and multiframe loaders remain constrained to `SamplesPerPixel = 1`.

**Proof obligation**:
For any RGB DICOM frame stack with depth `d`, rows `r`, columns `c`, and interleaved samples `S`, the DICOM color loaders construct exactly one tensor with shape `[d,r,c,3]` and element order `S[(((z*r + y)*c + x)*3 + k)]`. If declared metadata is not `SamplesPerPixel=3`, `PhotometricInterpretation=RGB`, `PlanarConfiguration=0`, unsigned 8-bit storage, or consistent spatial dimensions, the loader rejects before constructing `RgbVolume<B>`.

For any RGB PNG stack with depth `d`, height `h`, width `w`, and decoded interleaved RGB bytes `S`, the PNG color loaders construct exactly one tensor with shape `[d,h,w,3]` and element order `S[(((z*h + y)*w + x)*3 + k)]`. If any slice decodes as a non-`Rgb8` color type or has dimensions different from the first slice, the loader rejects before constructing `RgbVolume<B>`.

For any RGB JPEG decode result with height `h`, width `w`, and interleaved RGB bytes `S`, the JPEG color loader constructs exactly one tensor with shape `[1,h,w,3]` and element order `S[((y*w + x)*3 + k)]`. Because JPEG is lossy, the preservation contract is over the decoded raster `S`, not the pre-encoding source raster. If the decoded color type is not `Rgb8`, the loader rejects before constructing `RgbVolume<B>`.

For any RGB TIFF page stack with depth `d`, height `h`, width `w`, and decoded interleaved samples `S`, the TIFF color loader constructs exactly one tensor with shape `[d,h,w,3]` and element order `S[(((z*h + y)*w + x)*3 + k)]`. If any page is not `ColorType::RGB(_)`, has dimensions different from the first page, or decodes to a sample count different from `h*w*3`, the loader rejects before constructing `RgbVolume<B>`.

### 17. Image Comparison Metrics Boundary

> **Theorem 17.1 (Metric Family Separation)**: Image comparison metrics with different mathematical domains must live in separate leaf modules while preserving one public metric API surface.

**Boundary surface**:
- `ritk-core::statistics::image_comparison::overlap` owns Dice overlap computation.
- `ritk-core::statistics::image_comparison::surface` owns boundary extraction, distance primitives, Hausdorff distance, and mean surface distance.
- `ritk-core::statistics::image_comparison::quality` owns PSNR and global SSIM.
- `ritk-core::statistics::image_comparison::tests` owns value-semantic tests partitioned by the same metric families.
- `ritk-core::statistics` re-exports `dice_coefficient`, `hausdorff_distance`, `mean_surface_distance`, `psnr`, and `ssim`; caller-visible paths are unchanged.

**Proof obligation**:
For every public metric function `f` moved from the flat module, the new module tree exports exactly the same symbol, signature, and return contract. Since each function body was moved without semantic dependency changes and the focused `statistics::image_comparison` test suite validates analytical values for overlap, surface distance, PSNR, and SSIM, the split is behavior-preserving.

---

## Theoretical Foundations

### Transform Theory

#### Theorem T.1 (Transform Composition)
Given transforms T₁, T₂ ∈ Transform Space, their composition T₂ ∘ T₁ forms a valid transform.

**Proof**:
```
∀p ∈ Points, T₁(p) = p' ∈ Points
T₂(p') = p'' ∈ Points
∴ (T₂ ∘ T₁)(p) = p'' ∈ Points
```

#### Algorithm T.1 (Chained Transform)

**Input**: Sequence of transforms [T₁, T₂, ..., Tₙ], point p  
**Output**: Transformed point p'

```
ALGORITHM ChainedTransform:
    p' ← p
    FOR i ← 1 TO n:
        p' ← Tᵢ.transform(p')
    RETURN p'
```

**Complexity**: O(n) where n = number of transforms

### Interpolation Theory

#### Theorem I.1 (Linear Interpolation Continuity)
Given grid G with values V, linear interpolation Iₗ is C⁰ continuous.

**Proof Sketch**:
At grid boundaries, weights sum to 1:
```
∀x ∈ [x₀, x₁]: w₀(x) + w₁(x) = 1
where w₀(x) = (x₁ - x) / (x₁ - x₀)
      w₁(x) = (x - x₀) / (x₁ - x₀)
```

#### Algorithm I.1 (Trilinear Interpolation)

**Input**: Volume V[Z][Y][X], coordinate (z, y, x)  
**Output**: Interpolated value v

```
ALGORITHM TrilinearInterpolate:
    // Floor coordinates
    z₀ ← ⌊z⌋, y₀ ← ⌊y⌋, x₀ ← ⌊x⌋
    z₁ ← min(z₀ + 1, Z - 1), etc.
    
    // Weights
    wz ← z - z₀, wy ← y - y₀, wx ← x - x₀
    
    // Interpolate along X
    FOR k ∈ {0, 1}:
        FOR j ∈ {0, 1}:
            c₀₀ ← V[zₖ][y
