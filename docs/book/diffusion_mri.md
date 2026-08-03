# Diffusion MRI Acquisition and Q-ball ODFs

Diffusion-weighted MRI measures how water self-diffusion varies with gradient
direction and strength. RITK separates three concerns:

| Concern | Owner | Chapter |
|---|---|---|
| Validated gradient metadata (b-values, directions, frames, codecs, reorientation) | `ritk-diffusion-scheme` | [Diffusion Gradient Schemes](diffusion_scheme.md) |
| Orientation models derived from voxel signals | `ritk-diffusion` | This chapter (Q-ball ODFs; DTI, DKI, CSD, and NODDI follow the same pipeline pattern) |
| Streamline integration over a local orientation field | `ritk-tractography` | [Deterministic Streamline Tractography](tractography.md) |

This chapter focuses on the physics that relates gradient pulses to signal
attenuation — the bridge between the validated metadata in the scheme chapter
and the model-fitting algorithms documented in the [Diffusion
Models](ritk_diffusion.md) chapter.

## From gradient pulse to b-value

For the idealized Stejskal–Tanner pulsed-gradient experiment, the diffusion
weighting is

\[
b = \gamma^2 G^2 \delta^2\left(\Delta-\frac{\delta}{3}\right),
\]

where \(\gamma\) is the gyromagnetic ratio, \(G\) the gradient amplitude,
\(\delta\) the pulse duration, and \(\Delta\) the pulse separation. The measured
signal for a Gaussian diffusion tensor \(D\) is

\[
S(b,\mathbf g)=S_0\exp\!\left(-b\,\mathbf g^T D\mathbf g\right).
\]

Thus \(b\) has dimensions of time per area, conventionally s/mm²; it is not a
diffusivity. `DiffusionWeighting` stores the canonical SI value (s/m²)
internally through the Aequitas quantity system and exposes the scanner-facing
s/mm² value at the API boundary. See [the scheme
chapter](diffusion_scheme.md#diffusion-weighting) for the type-level contract
and Stejskal and Tanner,
[“Spin Diffusion Measurements”](https://doi.org/10.1063/1.1695690).

## Relationship to the gradient scheme

The [Diffusion Gradient Schemes](diffusion_scheme.md) chapter establishes
four guarantees that every model fitting depends on: validated directions,
unambiguous coordinate frames, scheme-level reorientation, and shell
grouping. The [Diffusion Models](ritk_diffusion.md) chapter documents how
all five implemented models (DTI, DKI, ODF, CSD, NODDI) consume these
guarantees — this chapter focuses on the physics shared by all of them.

## From scheme to signal to model

The pipeline established by the three chapters in this part is:

1. **Validated scheme** — the [Diffusion Gradient Schemes](diffusion_scheme.md)
   chapter ensures every direction carries a declared frame, a physically
   typed weighting, and a validated unit-vector contract.
2. **Orientation model** — the [Diffusion Models](ritk_diffusion.md) chapter
   documents the transformation from validated scheme + voxel signals →
   fitted field. Five models are implemented: DTI (log-linear tensor),
   DKI (nonlinear kurtosis), ODF (analytical Q-ball), CSD (non-negative
   fODF), and NODDI (3-compartment biophysical).
3. **Streamline geometry** — the [Deterministic Streamline
   Tractography](tractography.md) chapter integrates the fitted orientation
   field into Gaia polyline curves.

The [signal-to-streamlines example](examples/diffusion_tractography.md)
closes the loop: known tensor → synthetic signals → model fit → direction
field → streamlines.
