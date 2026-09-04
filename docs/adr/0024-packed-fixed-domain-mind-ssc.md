# ADR 0024: Packed fixed-domain MIND-SSC similarity

- **Status:** Accepted
- **Board item:** `RITK-MIND-REGISTRATION-2026-09-04`
- **Class:** [minor] [arch]
- **Date:** 2026-09-04
- **Revision 2026-09-04:** Replace the initial dense moving-descriptor plan
  with selected fixed centers and on-demand moving patch evaluation. A dense
  moving descriptor volume per pose exceeds the rigid-search runtime bound.

## Context

CT and MR soft tissue can have unrelated intensity scales while preserving
local anatomical self-similarity. Mutual information captures global intensity
co-occurrence, and normalized gradient fields capture edge orientation, but the
registration crate lacks the local modality-independent neighbourhood
descriptor (MIND) and self-similarity context (SSC) representation described by
Heinrich et al. [1,2].

A dense 12-channel descriptor is unsuitable for repeated rigid-pose scoring.
At 29×512×512 voxels, rebuilding moving descriptors would evaluate about 273
million descriptor-filter terms per pose and retain a large transform-dependent
buffer. The classical optimizer needs a fixed sample population and a fixed
denominator so loss of field of view cannot improve a candidate by deleting
samples.

The classical rigid-search affine uses physical `[z,y,x]` millimetres. Native
`Image` world points and `AtlasAffineTransform` use metadata `[x,y,z]`. That
bridge belongs to RITK, not each consumer.

## Decision

`metric::mind` owns `MindSscConfig`, `MindSscSampling`, `MindSscFixedPrep`,
`MindSscMemoryUsage`, `MindSscError`, and `mind_ssc_value`. Preparation retains
one C-order index and one packed `u64` descriptor per selected fixed center,
plus an optional selected weight. Evaluation maps each fixed support point
through the candidate transform and trilinearly samples the moving image on
demand. It builds no resampled volume or dense moving descriptor field.

For center \(x\), patch offsets \(p\in[-r,r]^3\), and the six axial neighbour
offsets \(n_j\in\{\pm d_z,\pm d_y,\pm d_x\}\), the 12 SSC distances compare
cross-axis neighbours separated by \(\sqrt{2}\) neighbour steps:

\[
D_j(x)=\sum_p\left(I(x+n_{a_j}+p)-I(x+n_{b_j}+p)\right)^2.
\]

With \(V(x)=\frac1{12}\sum_jD_j(x)\) and
\(D_{\min}(x)=\min_jD_j(x)\), each response is

\[
S_j(x)=\exp\left(-\frac{D_j(x)-D_{\min}(x)}{V(x)}\right).
\]

When \(V=0\), all responses are one. Otherwise each response is rounded to an
integer level in 0–5 and encoded as five unary bits. Twelve components occupy
the low 60 bits of one `u64`; XOR plus population count is exactly the L1
distance between quantized component levels. The similarity is

\[
1-\frac{\sum_{x\in\Omega_K}w_x\,
\operatorname{popcount}(q_F(x)\oplus q_M(Tx))}
{60\sum_{x\in\Omega_K}w_x}.
\]

The fixed domain contains only masked centers whose full patch-and-neighbour
support lies inside the fixed image. Moving support is sampled over continuous
C-order `[z,y,x]` indices. ITK's image field is the half-voxel interval
`[-0.5,size-0.5)` on each axis. Samples inside that field use replicate-edge
trilinear interpolation; a point outside it contributes explicit zero
background. It is never removed from the selected population or
denominator.

Sampling is explicit and validated:

- `dense()` selects every eligible center for small inputs.
- `try_indices` accepts caller-controlled C-order indices.
- deterministic stratified sampling uses all centers when the eligible count
  is at most 8,192; otherwise it recursively bisects the longest physical
  extent, assigns population-proportional quotas, and selects the lowest
  fixed-seed (`MIND-SSC`) hash ranks without replacement in each stratum.

Per-center normalized Hamming loss is in `[0,1]`. For uniform sampling,
Hoeffding's inequality gives

\[
K\ge\frac{\ln(2/0.01)}{2(0.02)^2}=6622.9,
\]

so the default cap is the next power of two, 8,192. The probability statement
applies to the random-ranking sampling design. Fixing and documenting one seed
provides repeatability; it does not itself establish the error bound for that
realization or population, clinical, or FDA validation. Spatial stratification
is a coverage constraint, not a replacement for external validation.

`classical::rigid_physical_affine_to_native` converts the already-physical
classical affine. With the axis-reversal permutation \(P\), it computes linear
part \(PMP\) and translation \(Pt\). It does not call
`index_affine_to_physical`, which has a different index-space contract.

## Memory and runtime model

For \(K\le8192\), persistent heap payload is `K*(sizeof(usize)+8)` bytes,
plus `4*K` only when weights are supplied: 128 KiB unweighted or 160 KiB
weighted on a 64-bit target at the default cap. Preparation scratch is bounded
by the selected heaps and stratum tables and is released after construction.
Each pose uses six patch values and 12 distance accumulators (72 bytes) per
center, reused immediately, plus constant-size transform state. Runtime is
`O(K * 12 * patch_volume)` scalar work and 162 moving trilinear samples per
center for the default 3×3×3 patch; it does not scale with total fixed voxels
after center selection.

## Alternatives rejected

- Dense 12-channel descriptors: rejected because they require 48 bytes per
  voxel before allocator overhead and encourage rebuilding the moving field.
- One packed moving descriptor volume per pose: rejected because resampling
  changes each patch support and the full-volume rebuild violates the pose
  budget.
- Random default samples: rejected because regression and regulated validation
  require reproducible objectives.
- Reusing `index_affine_to_physical`: rejected because the rigid-search result
  is already physical; applying image geometry again is dimensionally wrong.

## Verification and limits

Hand-derived fixtures pin the complete packed descriptor word, normalization,
and quantization independently of the implementation. Property tests cover
positive affine-intensity invariance. Value tests cover constant images,
one-shot/prepared parity, fixed denominators under field-of-view loss, malformed
inputs, deterministic repeats, stratum coverage, and dense parity when all
eligible centers fit. A differential physical-frame test uses anisotropic
spacing, nonzero origins, and rotated direction cosines. A manufactured rigid
translation verifies objective recovery. The RIRE example reports, but does
not promote, its Patient 001 score because raw masked CT/MR agreement is not a
cross-modality validation oracle. The benchmark records preparation memory and
repeated-pose runtime; these establish implementation behavior, not clinical
registration accuracy.

## References

1. Heinrich MP, et al. “MIND: Modality independent neighbourhood descriptor
   for multi-modal deformable registration.” *Medical Image Analysis* 16(7),
   2012, section 3, equations 1–4. DOI:
   <https://doi.org/10.1016/j.media.2012.05.008>.
2. Heinrich MP, et al. “Towards realtime multimodal fusion for image-guided
   interventions using self-similarities.” *MICCAI 2013*, pages 187–194,
   section 3.1. DOI: <https://doi.org/10.1007/978-3-642-40811-3_24>.
