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