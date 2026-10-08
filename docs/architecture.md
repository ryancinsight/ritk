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
- Shared tensor operations → `coeus` framework abstractions (via `coeus-tensor`, `coeus-ops`, `coeus-leto`)
- Common spatial math → `leto`/`gaia` type system
- IO patterns → Trait-based abstraction layer (ritk-io)

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
- Production decode is the RITK-native Rust SIZ/COD/QCD, packet, EBCOT, wavelet, and pixel-extraction path; `jpeg2k`, `openjp2`, and `openjpeg-sys` are not runtime or workspace dependencies.
- The current pixel extractor accepts one grayscale LRCP component with default precinct and code-block styles, 64×64 nominal code-blocks, inline packet headers, one tile-part per tile, and no progression/tile overrides or MCT. Unsupported profiles are rejected before output allocation rather than replaying one component's packets or accepting an incomplete packet sequence.
- Main and tile-header marker extents, terminal EOC, `Psot = 0` extent, and coverage of every declared tile are preflighted structurally. The boundary permits only a required one-byte DICOM zero pad after EOC. Packet and tier-1 decode require every expected LRCP header, a nonempty body for every included code-block, a valid coding-pass budget, and bounded MQ terminal fill before integer samples cross the DICOM boundary as `output = stored_integer × slope + intercept`.
- `ritk-dicom::NativeCodecBackend` remains the only DICOM transfer-syntax dispatch point for JPEG 2000 Lossless/Lossy, so parser ownership and codec implementation ownership stay separated.

**Proof obligation**:
For any supported grayscale DICOM JPEG 2000 frame `C` and layout `L`, if native packet and inverse-transform decoding yields the ISO 15444-1 integer sample sequence `S`, then `decode_jpeg2000_fragment(C,L)[i] = S[i] × L.rescale_slope + L.rescale_intercept`. Unsupported component traversal is a typed error, never a successful approximation.

> **Theorem 6.3 (JPEG Provider Boundary)**: DICOM JPEG decoding preserves the provider's validated encoded-grid raster while RITK alone interprets DICOM layout and modality metadata.

**Boundary surface**:
- `ritk-codecs::jpeg` owns `decode_jpeg_fragment(fragment, PixelLayout) -> Vec<f32>`.
- `consus-raster` owns bounded JPEG parsing and returns encoded-grid dimensions, a pixel format, and packed sample bytes without applying EXIF presentation orientation.
- `PixelFormat::GrayWide` samples use the provider's native-endian byte contract; conversion to signed or unsigned DICOM stored integers happens only after `PixelLayout` validation.
- `PixelFormat::Rgb` maps to interleaved RGB samples with `samples_per_pixel=3`, `BitsAllocated=8`, and unsigned sample interpretation.
- A single zero byte used to make an odd-length DICOM item value even is removed only when it follows terminal JPEG EOI; other trailing data is rejected by the provider.
- `PixelLayout` owns integer sample interpretation for all native codecs; `BitsAllocated=8` with `PixelRepresentation=1` maps each byte through `i8`, not `u8`.

**Proof obligation**:
For any DICOM JPEG frame `C` and layout `L`, if bounded provider decode yields dimensions `W,H`, pixel format `F`, and ordered sample bytes `S`, then `decode_jpeg_fragment(C,L)` either rejects `(W,H,F,S)` when it conflicts with `L`, or returns `stored_integer(S[i]) × L.rescale_slope + L.rescale_intercept`. For lossless 8 through 16-bit grayscale frames, `S` retains the exact encoded integer samples before signed interpretation and rescale.

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
- `ritk-png` owns `write_png`, `write_png_volume`, `encode_png_slice`, and `PngWriter<B>`.
- Reader invariant: grayscale pixels decode into `Image<B, 3>` with tensor shape `[1, height, width]` for a single PNG and `[slice_count, height, width]` for a series.
- RGB reader invariant: decoded `Rgb8` pixels decode into `RgbVolume<B>` with tensor shape `[1, height, width, 3]` for a single PNG and `[slice_count, height, width, 3]` for a series.
- Writer invariant: input `Image<B, 3>` must have `nz == 1` and is windowed from its own `[min, max]` onto 8-bit grayscale; the file therefore records rank order and shape, not the source scale. A volume-shaped input is rejected rather than truncated to its first slice.
- Metadata invariant: PNG carries no physical-space metadata, so origin is `[0,0,0]`, spacing is `[1,1,1]`, and direction is identity.
- Series invariant: directory slices are ordered by deterministic natural filename order and dimension mismatches are rejected before tensor construction.
- `ritk-io::format::png` is a facade re-export plus local `ImageReader` / `ImageWriter` adapters only.

### 11. JPEG Format Boundary

> **Theorem 11.1 (JPEG 2D Ownership)**: JPEG raster coding has exactly one implementation body in `consus-raster`; RITK owns only file policy and conversion into RITK image types.

**Boundary surface**:
- `consus-raster` owns bounded encoded-grid JPEG decode and grayscale encode.
- `ritk-jpeg` owns `read_jpeg`, `write_jpeg`, `JpegReader<B>`, and `JpegWriter<B>`.
- `ritk-jpeg` owns `read_jpeg_color_to_volume` and `JpegColorReader<B>`.
- `ritk-codecs` owns DICOM layout validation, signed sample interpretation, and modality rescale after bounded provider decode.
- Reader invariant: decoded grayscale JPEG pixels become Luma8-valued `Image<B, 3>` with tensor shape `[1, height, width]`; wide samples scale with nearest-integer rounding and RGB pixels convert to CIE luminance.
- RGB reader invariant: only provider `Rgb` pixels become `RgbVolume<B>` with tensor shape `[1, height, width, 3]`; grayscale input is rejected.
- Writer invariant: input `Image<B, 3>` must have `nz == 1`; values are rounded, clamped to `[0,255]`, and encoded as 8-bit grayscale.
- Metadata invariant: JPEG carries no physical-space metadata, so origin is `[0,0,0]`, spacing is `[1,1,1]`, and direction is identity.
- Orientation invariant: readers preserve encoded-grid orientation and do not apply EXIF display transforms to clinical image coordinates.
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
- `ritk-analyze`, `ritk-jpeg`, `ritk-metaimage`, `ritk-mgh`, `ritk-mif`, `ritk-minc`, `ritk-nifti`, `ritk-nrrd`, `ritk-png`, `ritk-tiff`, and `ritk-vtk` own their format parsers and writers.
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

**MIF structural invariant**:
- `ritk-mif::header` and `ritk-mif::decode` own text-header parsing and the `.mif.dat` detached-payload resolution; `ritk-mif::reader` owns `read_mif` / `read_mif_series` / `MifReader`, and `ritk-mif::writer` owns `write_mif` / `write_mif_series` / `MifWriter`.
- Spatial invariant: the `transform` 4×4 affine maps voxel `[x,y,z]` to scanner millimetres. The reader decomposes it into RITK origin, spacing, and direction; the writer reorders columns from internal `[depth, row, col]` to file `[x, y, z]`. A header with no `transform` is axis-aligned identity and its `vox:` triple — file `[x, y, z]` — must be reversed into `[Δdepth, Δrow, Δcol]`.
- `ritk-io::format::mif` is a facade re-export plus local `ImageReader` / `ImageWriter` adapters only.

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
- `ritk-io::format::dicom::load_color_volume_flat` loads validated interleaved RGB DICOM series into a channel-explicit flat buffer while preserving spatial metadata from the scalar DICOM series scanner.
- `ritk-io::format::dicom::load_color_multiframe_flat` loads validated interleaved RGB DICOM multiframe objects into `ColorMultiFrameVolume` while preserving multiframe origin, spacing, and direction metadata; its byte-payload counterpart serves browser or dropped-file hosts.
- `ritk-png::read_png_color_to_volume` and `ritk-png::read_png_color_series` load only decoded `Rgb8` PNG inputs into `RgbVolume<B>` with default PNG spatial metadata.
- `ritk-jpeg::read_jpeg_color_to_volume` loads only provider `Rgb` JPEG outputs into `RgbVolume<B>` with default JPEG spatial metadata.
- `ritk-tiff::read_tiff_color_to_volume` loads only TIFF `ColorType::RGB(_)` page stacks into `RgbVolume<B>` with default TIFF spatial metadata.
- Scalar DICOM series and multiframe loaders remain constrained to `SamplesPerPixel = 1`.

**Proof obligation**:
For any RGB DICOM frame stack with depth `d`, rows `r`, columns `c`, and interleaved samples `S`, the DICOM color loaders construct exactly one tensor with shape `[d,r,c,3]` and element order `S[(((z*r + y)*c + x)*3 + k)]`. If declared metadata is not `SamplesPerPixel=3`, `PhotometricInterpretation=RGB`, `PlanarConfiguration=0`, unsigned 8-bit storage, or consistent spatial dimensions, the loader rejects before constructing `RgbVolume<B>`.

For any RGB PNG stack with depth `d`, height `h`, width `w`, and decoded interleaved RGB bytes `S`, the PNG color loaders construct exactly one tensor with shape `[d,h,w,3]` and element order `S[(((z*h + y)*w + x)*3 + k)]`. If any slice decodes as a non-`Rgb8` color type or has dimensions different from the first slice, the loader rejects before constructing `RgbVolume<B>`.

For any RGB JPEG decode result with height `h`, width `w`, and interleaved RGB bytes `S`, the JPEG color loader constructs exactly one tensor with shape `[1,h,w,3]` and element order `S[((y*w + x)*3 + k)]`. Because JPEG is lossy, the preservation contract is over the decoded raster `S`, not the pre-encoding source raster. If the provider format is not `Rgb`, the loader rejects before constructing `RgbVolume<B>`.

For any RGB TIFF page stack with depth `d`, height `h`, width `w`, and decoded interleaved samples `S`, the TIFF color loader constructs exactly one tensor with shape `[d,h,w,3]` and element order `S[(((z*h + y)*w + x)*3 + k)]`. If any page is not `ColorType::RGB(_)`, has dimensions different from the first page, or decodes to a sample count different from `h*w*3`, the loader rejects before constructing `RgbVolume<B>`.

### 17. Render Pipeline Scratch-Buffer Boundary

> **Theorem 17.1 (Render-Path Zero-Allocation After Warm-Up)**: Once `RenderBufferPool` scratch buffers have reached peak observed dimension, all subsequent dirty-texture rebuilds for equal or smaller dimensions incur zero heap allocations.

**Boundary surface**:
- `ritk-snap::render::buffer_pool::RenderBufferPool` owns three monotone-capacity scratch buffers: `pixel_f32: Vec<f32>`, `rgba_u8: Vec<u8>`, `color32: Vec<egui::Color32>`.
- `SliceRenderer::render_with_scratch` writes f32 extraction and u8 RGBA encoding into `pool.pixel_f32` and `pool.rgba_u8`, replacing the former per-rebuild `Vec<f32>` and `Vec<u8>` allocations.
- `render_mip_axial_with_scratch` / `render_vr_axial_with_scratch` write RGBA into `pool.rgba_u8`.
- `apply_to_image_into` writes viewport orientation transform output (flip/rotate) into `pool.color32`, replacing the former multi-step `Vec<Color32>` allocations (one per flip/rotate step).
- `SnapApp::render_buffer_pool` is the single owner; it is threaded through as `&mut RenderBufferPool`.

**Capacity invariant**: `Vec::capacity` is monotone non-decreasing for each scratch field. `resize_u8`, `resize_color32`, and direct `pixel_f32.resize` extend when needed and reuse without shrinking. New elements are zero-initialized (`0_u8`, `Color32::BLACK`, `0.0_f32`).

**Allocation profile per dirty-texture rebuild (after warm-up)**:

| Operation | Before pool | After pool | Remaining |
|---|---|---|---|
| `extract_slice` → f32 | 1 × `Vec<f32>` alloc | 0 (reused) | — |
| RGBA encoding → u8 | 1 × `Vec<u8>` alloc | 0 (reused) | — |
| Orientation transform | N × `Vec<Color32>` allocs | 0 (reused) | — |
| `ColorImage::from_rgba_unmultiplied` | 1 × `Vec<Color32>` alloc | same | 1 (egui API) |
| Texture name string | 1 × `String` alloc (format!) | 0 (static str) | — |

**Deferred**: `egui::ColorImage::from_rgba_unmultiplied` constructs a `Vec<Color32>` internally. Eliminating this requires an egui API addition for in-place texture pixel update. Tracked as GAP-258-PERF-03 (Low, blocked on upstream).

**Proof obligation**: For all valid (`img`, `transform`) inputs, `apply_to_image_into(pool, img, transform).pixels == apply_to_image(img, transform).pixels`. The differential test suite in `view_transform/tests.rs` verifies pixel identity across all 16 `(flip_h, flip_v, rotation)` combinations for both rectangular and square images.

### 18. Image Comparison Metrics Boundary

> **Theorem 18.1 (Metric Family Separation)**: Image comparison metrics with different mathematical domains must live in separate leaf modules while preserving one public metric API surface.

**Boundary surface**:
- `ritk-core::statistics::image_comparison::overlap` owns Dice overlap computation.
- `ritk-core::statistics::image_comparison::surface` owns boundary extraction, distance primitives, Hausdorff distance, and mean surface distance.
- `ritk-core::statistics::image_comparison::quality` owns PSNR and global SSIM.
- `ritk-core::statistics::image_comparison::tests` owns value-semantic tests partitioned by the same metric families.
- `ritk-core::statistics` re-exports `dice_coefficient`, `hausdorff_distance`, `mean_surface_distance`, `psnr`, and `ssim`; caller-visible paths are unchanged.

**Proof obligation**:
For every public metric function `f` moved from the flat module, the new module tree exports exactly the same symbol, signature, and return contract. Since each function body was moved without semantic dependency changes and the focused `statistics::image_comparison` test suite validates analytical values for overlap, surface distance, PSNR, and SSIM, the split is behavior-preserving.

---

### 19. Gaia Meshing Boundary

> **Theorem 19.1 (Gaia SSOT)**: All surface mesh generation, Boolean CSG operations, watertight analysis, and welded mesh I/O must be delegated to `gaia::IndexedMesh<f64>`. No RITK crate may implement competing mesh topology, vertex deduplication, or triangle-soup-to-indexed-mesh conversion.

**Repository layout**:
- `gaia/` is cloned at `D:\ritk\gaia` as a separate git repository tracked independently. It is excluded from the ritk `.gitignore` (`/gaia`).
- The workspace `Cargo.toml` references it via `gaia = { path = "gaia", default-features = false }`.

**Boundary surface**:
- `ritk-core::filter::surface::Mesh` is `pub type Mesh = gaia::IndexedMesh<f64>` — the SSOT type alias for all surface-extraction output.
- `ritk-core::filter::surface::MeshBuilder` re-exports `gaia::MeshBuilder` — the SSOT builder for low-level triangle construction.
- `ritk-vtk::domain::mesh_bridge::indexed_mesh_to_poly` converts `IndexedMesh<f64>` → `VtkPolyData` for VTK interchange (preserving welded topology, narrowing coordinates to f32).
- `ritk-vtk::domain::mesh_bridge::poly_to_indexed_mesh` converts `VtkPolyData` → `IndexedMesh<f64>` (applying VertexPool welding, promoting coordinates to f64; only triangular cells are mapped).
- `ritk-vtk::io::mesh_indexed` owns gaia-native file I/O: `read_stl_indexed`, `read_obj_indexed`, `read_ply_indexed`, `write_indexed_stl_binary`, `write_indexed_stl_ascii`, `write_indexed_obj`, `write_indexed_ply`, `write_indexed_glb`.
- `ritk-vtk::io::{stl,obj,ply,gltf}` own `VtkPolyData`-based I/O for VTK pipeline interchange (triangle soup, no welding); these are retained for Paraview/ITK-SNAP round-trip compatibility.

**Invariants**:
1. Every surface mesh produced by `MarchingCubesFilter`, CSG operations, or primitive generation is an `IndexedMesh<f64>` with VertexPool-welded vertices.
2. Writing an `IndexedMesh` to any supported format (STL, OBJ, PLY, GLB) must use `ritk-vtk::io::mesh_indexed`; writing a `VtkPolyData` to a format uses the corresponding `ritk-vtk::io::{stl,obj,ply}` function.
3. The `VtkPolyData` representation is permissible only at VTK interchange boundaries; internal algorithm state must use `IndexedMesh<f64>`.

**Proof obligation**:
For `poly_to_indexed_mesh(indexed_mesh_to_poly(M))`, vertex count ≤ M.vertex_count() because narrowing f64→f32 cannot introduce new distinct vertex positions, and welding can only merge. Face count is invariant because each triangle is mapped to exactly one face.

### 20. Stored-Volume Conversion Contract Boundary

> **Theorem 20.1 (Conversion Contract Ownership)**: Exactly one crate owns the stored-volume conversion contract; format crates implement it and never redefine it.

The decision record is [ADR 0054](adr/0054-stored-volume-contract.md). This section specifies the interfaces, behaviors, and edge cases the contract imposes on every format route.

**Boundary surface**:
- `ritk-image-io` owns `StoredVolume`, `StoredSeries`, `SeriesAxis`, `IntensityCalibration`, `ConversionTarget`, `ConversionAdapter`, `ConversionRejection`, `ConversionPrepareError`, `ConversionCapabilityReport`, `ConversionLoss`, `ConversionFeature`, `ConversionLocation`, `FormatMetadataLoss`, `PreparedConversion`, `report_conversion_capabilities`, and `prepare_conversion`. These are the SSOT names for the contract.
- `ritk-image-io` depends inward on `ritk-codecs` (`SampleBuffer`, `SampleType`), `ritk-image` (`ImageMetadata<3>`), and `ritk-spatial` (`CoordinateMap`). It must not depend on any format crate (DIP: conversion policy depends on the `ConversionAdapter` abstraction, which the format crates implement).
- Each format crate owns its header parse/serialize and implements `ConversionTarget` + `ConversionAdapter`; `ritk-io` exposes only facade re-exports and trait adapters (Theorem 14.1).

**Interface contract**:
```
ConversionTarget {
    const FORMAT: &'static str;
    const FEATURES: &'static [ConversionFeature];
}
ConversionAdapter: ConversionTarget {
    type Plan;
    type Rejection: ConversionRejection;          // fn location(&self) -> ConversionLocation
    fn prepare(&self, series: &StoredSeries) -> Result<Self::Plan, Self::Rejection>;
}
prepare_conversion(target, source_format, series, metadata_losses)
    -> Result<PreparedConversion, ConversionPrepareError<Rejection>>
```

**Behavioral invariants**:
1. **Preflight before output.** `prepare_conversion` accepts no destination, so a rejected conversion cannot create or alter an output. Every writer entry point consumes a `PreparedConversion` and opens its destination only after the plan exists.
2. **Capability first, target second.** `report_conversion_capabilities` compares declared feature categories per volume and per series axis; a non-empty loss set rejects before the adapter runs. The adapter then checks the input-dependent values and cross-volume constraints a category report cannot express.
3. **Exact scope.** Every rejection returns the `ConversionLocation` (Series, Volume, or Frame) that violates the target contract.
4. **No silent rescale.** Stored reads retain the source sample representation and calibration; compute-ready values require an explicit calibration operation, never an implicit conversion during a stored read.
5. **Value semantics over presence.** A supported round trip preserves exact stored sample bits, LPS-millimetre geometry, calibration, and axis semantics, or reports typed loss; an adapter never reports success over dropped semantics.

**Edge cases**:
- A series is non-empty by construction; an empty series is unrepresentable rather than a runtime rejection.
- `FEATURES` is a `&'static [ConversionFeature]` resolved at compile time; a new category is a single edit in `ritk-image-io`, and a zero-sized or tag-only adapter keeps `prepare` monomorphized with no dynamic dispatch.
- `metadata_losses` is the caller's channel for source-format fields the shared model cannot carry; a format adapter that cannot retain a field reports it here instead of dropping it.
- A zero multiplicative scale that a target treats as "scaling disabled" (for example NIfTI `scl_slope == 0`) is a rejection, never an encoded no-op.

### 21. NIfTI Stored Conversion Boundary

> **Theorem 21.1 (NIfTI Stored Round-Trip Fidelity)**: A NIfTI document built from a stored series and read back preserves exact sample bits, LPS-millimetre geometry, calibration, and the acquisition axis, or reports scoped typed loss.

**Boundary surface**:
- `ritk-nifti::NiftiDocument` owns single-file transport: `from_bytes`, `read`, `write`, `sample_bytes`, `uncompressed_bytes`, `header`.
- `ritk-nifti::NiftiDocument::from_stored_series` owns the write path through `NiftiStoredSeriesTarget` (`ConversionTarget` + `ConversionAdapter`).
- `ritk-nifti::NiftiDocument::to_stored_series` owns the read path (document → `StoredSeries`).
- `ritk-nifti::spatial` owns RAS↔LPS row conversion and `[x,y,z]`↔`[depth,row,col]` column mapping; read and write share it (SSOT).

**Reader interface** (`to_stored_series`):
- Input a validated `NiftiDocument`; output a `StoredSeries` with one `StoredVolume` per declared volume, or a scoped typed error.
- Shape `dim[1..=3] = [nx, ny, nz]` becomes internal shape `[nz, ny, nx]`.
- `datatype_code` maps through `NiftiDatatype` to `SampleType`; the payload is decoded in the header's byte order, retaining exact bits.
- The active affine (`sform` when `sform_code > 0`, else `qform` when `qform_code > 0`, else the `pixdim` diagonal) converts through `metadata_from_nifti_ras_affine` to LPS origin, spacing, and direction.
- `scl_slope == 0` is `IntensityCalibration::Identity`; otherwise `Linear(LinearCalibration::new(scl_slope, scl_inter))`.
- Rank 3 is `SeriesAxis::SingleVolume`; rank 4 is `SeriesAxis::List`.

**Writer interface** (`from_stored_series`): rejects before allocating on per-volume shape, sample-type, geometry, or calibration mismatch; unsupported sample type; unsupported axis; non-constant or empty per-frame calibration; zero-slope calibration; modality-lookup calibration; a volume count outside the version's `dim[4]` range; and a document above `MAX_DOCUMENT_BYTES`.

**Edge cases**:
- `scl_slope` is a stored zero but a nonzero physical scale (NIfTI reads `scl_slope == 0` as "scaling disabled"): reject, never silently drop.
- Both `qform` and `sform` active with opposite handedness: `sform` is authoritative; the reader must not average or prefer `qform`.
- Spatial `xyzt_units` other than millimetres (metre = 1, micron = 3) require conversion or typed loss; the stored model is LPS-millimetre.
- NIfTI-1 stores `pixdim`, `scl_*`, and `srow_*` as `f32`; a value that underflows to zero after narrowing is rejected by `validate_for_encoding`.
- `vox_offset` beyond the header leaves an extension gap; payload slicing starts at `vox_offset`, not at the header end.
- Rank-4 with `dim[4] == 1` is distinct from rank-3 and is read as a one-volume `List`.
- `u64`/`i64`/`f64` payloads round-trip without narrowing; this is the reason the stored path exists.

### 22. NRRD Stored Conversion Boundary

> **Theorem 22.1 (NRRD Stored Round-Trip Fidelity)**: An NRRD document written from a stored series and read back preserves exact sample bits, LPS-millimetre geometry, the acquisition axis, and any per-slice coordinate map, or reports scoped typed loss.

**Boundary surface**:
- `ritk-nrrd::NrrdDocument` owns in-memory samples plus validated metadata (`new`, `series`, `comments`, `records`); `read_nrrd_document` / `write_nrrd_document` own file transport.
- `ritk-nrrd::reader::stored` owns `read_nrrd_stored` / `read_nrrd_stored_series`; `ritk-nrrd::writer::stored` owns `write_nrrd_stored` / `write_nrrd_stored_series`.
- `ritk-nrrd::spatial` and `ritk-nrrd::coordinate_map` own `[x,y,z]`↔`[depth,row,col]` mapping and the per-slice / fixed-parameter coordinate-map extension.

**Reader behaviors**: all ten fixed-width sample types and standard type aliases; both binary payload byte orders (an explicit `endian` is required for multi-byte binary samples); `raw`, `ascii` (`text`, `txt`), and `gzip` (`gz`) encodings; line skip before byte skip; `byte skip: -1` for raw payloads only; detached data limited to one relative filename (no absolute paths or parent traversal); `space`, `space directions`, and `space origin` normalized to LPS-millimetre; `kinds: list/domain/dwmri`; NRRD0005 measurement frame for nonzero DWI gradients; encoded-byte, decoded-byte, and volume-count ceilings enforced before allocation.

**Writer behaviors**: a trailing contiguous acquisition axis keeps each volume contiguous; little-endian payload; NRRD0004 for non-DWI series and NRRD0005 with an identity measurement frame for DWI; exact samples streamed through a buffered writer; non-identity calibration rejected before the output opens.

**Edge cases**:
- Per-axis `units` without a directions or spacings source are rejected; units alone do not define a grid.
- Unknown `space`, anonymous `space dimension` frames, and units without a known millimetre conversion are rejected.
- `axismins`, `axismaxs`, and `centerings` aliases are canonicalized before conflict checks.
- A measurement frame without DWMRI acquisition metadata, axis support bounds, or cell/node centering is rejected before the payload is read.
- Non-identity calibration is rejected by the stored writer (NRRD has no standard modality-calibration field).
- ASCII payloads do not carry binary float bits; a float sample written as ASCII is a lossy path and must be reported.

**Convergence step**: `ritk-nrrd` currently validates inside `NrrdDocument::new` / `write_to` and does not expose `ConversionTarget` + `ConversionAdapter`. Wrapping that validation as `NrrdStoredSeriesTarget` and routing it through `prepare_conversion` gives NRRD and NIfTI one shared preflight entry point (RITK-IMAGE-CONVERSION-ADAPTERS-001).

### 23. Analyze Stored Conversion Boundary

> **Theorem 23.1 (Analyze Pair Atomicity)**: An Analyze write produces a consistent `.hdr`/`.img` pair or leaves both destinations unchanged.

**Boundary surface**:
- `ritk-analyze::reader` owns the 348-byte header parse and `.img` streaming decode; `ritk-analyze::writer` owns header encode and payload emission; `ritk-analyze::codec` owns the `DT_*` constants and little-endian primitives (SSOT for the pair's byte layout).

**Reader behaviors**: little-endian only; big-endian and paired NIfTI-1 (`ni1\0`) rejected; `dim[0] ∈ {3,4}` with `dim[4] == 1`; datatype `2/4/8/16/64`; `bitpix` must match the datatype; `pixdim[1..=3] ≤ 0` falls back to unit spacing; `funused1` (`0 → 1`) is the scale; `vox_offset` must be a whole byte count; origin reconstructed from the `originator` voxel coordinates times spacing; the `.img` length must equal `vox_offset + voxel_count × width`.

**Writer behaviors**: `f32` only (`DT_FLOAT`); dimensions at most `i16::MAX`; positive finite `f32` spacing; origin rounded to an `i16` voxel coordinate; identity direction implied; the header is published only after the full payload is written.

**Edge cases**:
- Analyze has no direction field: any non-identity direction is unrepresentable and must be rejected on write rather than silently dropped.
- The `originator` field is unreliable across writers and rounds the origin to voxel coordinates; a non-integer-representable origin is rejected or reported as loss.
- Pair atomicity: a failure after the `.img` is written but before the `.hdr` leaves an orphan `.img`; the writer stages both or documents the recovery.
- Datatype `2/4/8/16/64` only; `u16`, `u32`, `i64`, and `u64` are unrepresentable.
- The current reader decodes to `f32` and applies `funused1`, discarding the stored type and separating values from calibration. Completing the capability requires a stored read that returns the source `SampleType` and carries `funused1` as `LinearCalibration`.

**Gap**: no stored read/write and no adapter yet (RITK-ANALYZE-CONVERSION-001, RITK-ANALYZE-CONVERSION-INVENTORY-001).

### 24. DICOM Stored Conversion Boundary

> **Theorem 24.1 (DICOM Stored Import Fidelity)**: Supported DICOM instances import into stored samples without scaling or narrowing voxels.

> **Theorem 24.2 (DICOM Metadata Accounting)**: Every parsed DICOM element is interpreted, retained opaquely, or recorded as a scoped loss; none is discarded without a record.

**Boundary surface**:
- `ritk-io::format::dicom::reader` owns series assembly and slice pixel decode; `ritk-dicom` owns Part 10 parsing (`DicomParseBackend`), transfer-syntax dispatch (`NativeCodecBackend`), and pixel-layout interpretation (`PixelLayout`).
- `ritk-codecs` owns the encapsulated fragment decoders (JPEG, JPEG-LS, JPEG 2000, RLE, PackBits) and native pixel primitives.
- Geometry derives from `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, and slice spacing; calibration derives from `RescaleSlope` / `RescaleIntercept`.

**Reader behaviors (initial path)**: monochrome uncompressed instances with identity calibration; validate geometry and pixel layout; preserve the source metadata inventory; return exact samples; unsupported encoding or calibration fails before a series escapes (RITK-DICOM-STORED-IMPORT-001).

**Metadata inventory (ADR 0055)**: `ritk-io::format::dicom` owns the source-owned retention inventory. `DicomPreservationSet` carries interpreted nodes, opaquely retained elements, and `DicomRetentionLoss { tag, reason }` records; `DicomRetentionReason` is `#[non_exhaustive]` and names the three cases where retention is impossible — `SequenceItemsUnavailable` (an SQ element exposed no items and could not be re-encoded), `ValueBytesUnavailable` (the value could not be re-encoded to bytes), and `NestingDepthExceeded` (recorded at the boundary element when recursion would pass `MAX_RETAINED_SEQUENCE_DEPTH`). The inventory travels on the reader's own metadata (`DicomReadMetadata::preservation`, `DicomSliceMetadata::preservation`) and is never derived from a destination. `inventory::dicom_metadata_losses` is the single projection into the shared `FormatMetadataLoss` vocabulary that `prepare_conversion` consumes, so a conversion preflight rejects a source whose metadata was not fully retained. Losses are scoped per slice, so each one carries its exact `ConversionLocation::Frame`.

**Edge cases**:
- `BitsAllocated`, `BitsStored`, `HighBit`, and `PixelRepresentation` determine the stored integer interpretation; `BitsAllocated=8, PixelRepresentation=1` maps through `i8`.
- `SamplesPerPixel ≠ 1` (color) cannot enter a scalar `StoredVolume`; reject or route to a color path (Theorem 6.4).
- Mixed slice geometry in a series (inconsistent spacing, orientation, or frame of reference) is rejected, not averaged.
- `RescaleSlope` / `RescaleIntercept` belong in `IntensityCalibration`, never baked into samples.
- Gantry tilt and non-orthogonal slice ordering require a per-slice coordinate map (`CoordinateMap::SliceSeries`), not a single affine.
- Encapsulated transfer syntaxes decode through `ritk-codecs`; the initial import path is uncompressed-only, with encapsulated decode a later increment.
- Nesting past the retention bound is a recorded loss, not a silent truncation; a sequence that cannot be walked is first attempted as opaque bytes (Theorem 24.2).

**Gap**: no stored import and no adapter yet. The metadata inventory (RITK-DICOM-METADATA-INVENTORY-001) and the object-pixel preflight (RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001) are complete, so only the stored read/write, its `ConversionTarget`/`ConversionAdapter`, and its calibration/unit mapping remain.

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
