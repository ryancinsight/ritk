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
Native JPEG replacement changes codec internals behind `ritk-codecs` / `NativeCodecBackend`; DICOM readers continue to call `decode_frame_with::<DicomRsBackend>` until the parser backend is replaced.

**Codec ownership invariant**:
`ritk-codecs` owns JPEG, JPEG-LS, JPEG 2000, RLE, PackBits, and native pixel primitive implementations. `ritk-dicom::codec` may re-export those primitives and dispatch by transfer syntax, but must not retain copied codec bodies. Native-owned JPEG syntaxes selected by `TransferSyntaxKind::is_native_jpeg_codec()` route exclusively through `NativeCodecBackend`; external backend fallback is limited to `TransferSyntaxKind::is_external_backend_codec_candidate()`.

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

### 10. Format Facade Monomorphization Boundary

> **Theorem 10.1 (Single Implementation Ownership)**: A format with a dedicated crate has exactly one parser/writer implementation body; `ritk-io` may expose only static re-exports and trait adapters.

**Boundary surface**:
- `ritk-analyze`, `ritk-metaimage`, `ritk-mgh`, `ritk-nifti`, `ritk-nrrd`, and `ritk-vtk` own their format parsers and writers.
- `ritk-io::format::*` modules for those crates are facade boundaries. They re-export the authoritative functions and define only local `ImageReader` / `ImageWriter` adapters when orphan rules require those impls to live in `ritk-io`.
- Adapter types remain generic over `B: Backend`; calls monomorphize per backend and do not use dynamic dispatch in throughput paths.
- Copied reader/writer files under `ritk-io` for dedicated-crate formats are prohibited.

**Verification invariant**:
Implementation tests live with the owning format crate. `ritk-io` tests only facade-level behavior and trait-adapter wiring.

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
