# Example: A Real Subject

The [previous example](diffusion_tractography.md) demonstrates the method on a
synthetic single-fibre phantom, where the tensor that generated the signal is
known and the recovered ODF peak can be asserted at exactly zero degrees of
error. That is a statement about correctness.

This one is a different claim. It runs the same estimators over a real
acquisition, where no ground truth exists, and shows what they produce. The
question it answers is not "is the method right" but "does the method, applied
to scanner data, recover anatomy".

![Fractional anisotropy and deterministic streamlines from a real diffusion acquisition](../figures/brain_tractography.svg)

## The data

OpenNeuro [`ds002087`](https://openneuro.org/datasets/ds002087) sub-01, released
under CC0. 104 × 104 × 72 voxels at 2 mm isotropic, 99 volumes at \(b = 0\) and
\(b = 700\) s/mm², with FSL `bval`/`bvec` sidecars.

The dataset is not committed — only its provenance is. Fetch it, then
regenerate the figure:

```bash
bash test_data/diffusion/download.sh
cargo run --release -p ritk-diffusion --example book_brain_tractography
```

The script clones the OpenNeuro DataLad repositories for the sidecars and pulls
the DWI volume from the public S3 bucket. The clones carry git-annex *pointer*
files rather than imaging data — a `.nii.gz` there is a few hundred bytes naming
a key — so the volume is fetched directly instead of requiring git-annex. Re-runs
skip files already present, judged by size rather than existence, since an
existence check would happily accept the pointer.

The example exits without writing when the data is absent, so it stays runnable
where the dataset is not present.

## Reading the panels

Both panels are **directionally encoded**: red is left-right, green
anterior-posterior, blue superior-inferior, following the standard diffusion
convention. Colour comes from the absolute components of the local orientation —
absolute because an eigenvector has no sign, so a fibre running left-to-right
must not change colour against one running right-to-left.

That mapping is derived per volume rather than assumed. `resolve_colour_channels`
reads the image's own direction cosines and works out which anatomical axis each
gradient component actually points along, because the two orders involved are
reversed: FSL `bvec` components are in image-axis order `(i, j, k)`, while RITK
stores direction cosines as `[depth, row, column]` against LPS rows. It fails
rather than guesses when an axis has no dominant anatomical direction, which is
what an obliquely acquired volume looks like — colours would then name
directions the data does not have. For this acquisition the sform is
axis-aligned to within a few degrees, so the resolved channels are the identity.

**Panel 1 — fractional anisotropy.** One tensor is fitted per voxel from all 99
volumes; FA is the rotationally invariant measure of how directional that
tensor is, and here it sets the brightness while orientation sets the hue. The
anatomy is the check, because it is not something a bug produces:

- the **corpus callosum** reads as a bright arc, genu anteriorly and splenium
  posteriorly, which is the most coherent white matter in the brain;
- the **ventricles** are dark, as they must be — cerebrospinal fluid diffuses
  isotropically, so its FA is near zero;
- **cortical grey matter** is dark at the periphery while the interior white
  matter is bright;
- the **internal capsule** is visible lateral to the ventricles.

**Panel 2 — streamlines.** Deterministic tracking follows the principal
eigenvector through the fitted field, seeded in the most anisotropic voxels.
Each segment is coloured by its own local direction rather than each track by a
single colour, since a streamline changes orientation along its length and one
stroke colour would average that away.
Tracking is three-dimensional over a 29-slice slab; the tracks are projected
onto the rendered plane for display. Their agreement with the bright structure
underneath is the check — tracks straying onto dark tissue would mean the
direction field and the FA map disagree.

Streamlines are `gaia::Polyline` values, not a RITK-local curve type, per
[ADR 0036](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0036-neuroimaging-and-mr-ownership.md).

## Two choices the figure depends on

Both exist because a plausible-looking map is easy to produce by accident.

**A background mask.** Outside the head the signal is noise, and a tensor
fitted to noise is strongly anisotropic. Without masking, a bright rim traces
the skull and dominates the FA range. Voxels below 12 % of the \(b = 0\)
signal's 98th percentile are not fitted.

**Rejecting degenerate fits, not dim voxels.** Some voxels pass the intensity
mask yet still yield a collapsed, rank-one tensor: one large eigenvalue with the
other two near zero. Such a tensor is positive-definite, so a sign check accepts
it, and its FA approaches 1. Fits are therefore rejected on physics — a smallest
eigenvalue below 10⁻⁵ mm²/s, or a largest above free water at 3.2 × 10⁻³ mm²/s,
is not a measurement.

## What this figure does not claim

Peak FA in the rendered slice is **0.972**. Coherent white matter measures
roughly 0.85–0.90, so the highest values here are still fit artefacts rather
than tissue — residual partial-volume voxels at boundaries, which single-shell
DTI at \(b = 700\) does not resolve. The remedy is a better fit or proper brain
extraction, not a tighter constant: tightening the eigenvalue floor until the
peak looks physiological would tune a threshold to move a number rather than fix
the estimate.

No clinical or anatomical conclusion should be drawn from one subject, one
slice, and an uncorrected acquisition. Motion, eddy-current, and susceptibility
correction are all available in RITK and none is applied here — the figure shows
the estimators, not a pipeline.
