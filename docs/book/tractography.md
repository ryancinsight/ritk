# Deterministic Streamline Tractography

`ritk-tractography` integrates a local orientation field into curves, producing
Gaia polyline geometry. It does not directly reconstruct axons and it does not
resolve the biological cause of a local orientation. The current algorithm is a
deterministic Euler baseline — explicit stepping with direction continuity,
turn-angle gating, and step-count bounding — designed for reproducible
examples and baseline algorithms.

This chapter documents the integration algorithm, the direction-field helpers
that bridge diffusion models to tractography, and the format export methods
that write streamlines for validation against reference toolchains.

## Integration rule

For a physical seed point `x₀`, step size `h > 0`, and local unit orientation
`v(x)`, one forward step is

```text
xₖ₊₁ = xₖ + h · v(xₖ)
```

At each sample the integrator validates the returned direction:
non-finite components or a norm that deviates from unity beyond `10⁻⁶`
produce a typed `InvalidDirection` error rather than a silently dropped
streamline.

### Sign continuity

Diffusion orientations are antipodally symmetric: `v` and `−v` describe the
same physical axis. An ODF peak extractor,
spherical harmonic evaluator, or eigendecomposition may return either sign
arbitrarily. At each step the integrator computes the dot product between
the current direction and the returned direction. When the dot product is
negative, the direction is flipped so the streamline continues forward
rather than reversing:

~~~rust,ignore
if current_direction.dot(&next_direction) < 0.0 {
    next_direction = -next_direction;
}
~~~

This prevents arbitrary sign choices from making a streamline reverse
direction mid-integration.

### Bidirectional tracking

`TrackingDirection::Bidirectional` integrates both signs from the seed —
one half along `+v` and one along `-v` — reverses the backward half, and
joins the two halves with the seed appearing exactly once. The joined
polyline runs from the backward termination point through the seed to the
forward termination point.

`TrackingDirection::Forward` integrates only along the initial direction
returned at the seed. The backward termination reason is `None` in the
output.

## Configuration

`TractographyConfig` carries four validated parameters:

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `step_size` | `f64` | 0.5 mm | Finite, `> 0` |
| `max_steps` | `usize` | 1 000 | Nonzero, seed + steps fits in `usize` |
| `max_turn_degrees` | `f64` | 60° | Finite, `∈ [0, 180]` |
| `tracking_direction` | `TrackingDirection` | `Bidirectional` | — |

Construction validates every parameter and returns `InvalidStepSize`,
`InvalidMaxSteps`, or `InvalidTurnLimit` for out-of-range values:

```rust,ignore
use ritk_tractography::{TractographyConfig, TrackingDirection};

let config = TractographyConfig::new(0.5, 1_000, 45.0, TrackingDirection::Bidirectional)?;
```

The turn-angle cosine limit is precomputed once at construction:
`cos(max_turn_degrees.to_radians())`. A turn-angle check fails when the
dot product between consecutive (sign-aligned) directions falls below this
threshold.

## Termination is part of the result

Every half-streamline records exactly one `TerminationReason`:

| Variant | Condition | The out-of-domain point |
|---|---|---|
| `FieldBoundary` | Direction field returns `None` at the proposal | Not appended |
| `TurningAngle` | Sign-aligned direction dot product `< cos(θ_max)` | Appended (it was reached) |
| `StepLimit` | `max_steps` accepted steps exhausted | N/A — last step was accepted |

The distinction between `FieldBoundary` and `TurningAngle` matters for
interpretation: a streamline that stops because the field has no trackable
orientation reached an anatomical boundary; one that stops because the
turn angle exceeded the limit may have encountered noise, a crossing-fibre
region, or a genuine curvature that the configuration rejected.

A seed at which the field returns `None` at the first query is a normal
untrackable seed — it produces no streamline and is not an error.

## Direction fields

A direction field is any closure `Fn(&Point<3>) -> Option<Vector<3>>`.
`ritk-tractography` provides five pre-built helpers that bridge diffusion
models to the integration algorithm, and the caller can supply a custom
closure for ad-hoc or synthetic fields.

### Single-voxel fields

`dti_pev_direction_field` and `fod_peak_direction_field` extract one
direction from a single-voxel fit and return the same direction at every
query point. They are useful for bootstrapping and unit testing:

```rust,ignore
use ritk_tractography::{euler_tractography, dti_pev_direction_field};

let tensor = ritk_diffusion::dti::estimate_dti(&scheme, &signals, config)?;
let result = euler_tractography(&seeds, track_config, dti_pev_direction_field(&tensor))?;
```

`dti_pev_direction_field` returns `None` when the fitted tensor has
near-zero FA (`< 10⁻¹⁰`), i.e. it is isotropic and the PEV is not
trackable.

`fod_peak_direction_field` pre-extracts the strongest fODF peak via
`FodField::find_peaks(50, 100, 0.1)` and returns `None` when no peak
meets the 10% relative-amplitude threshold:

```rust,ignore
use ritk_tractography::fod_peak_direction_field;

let result = euler_tractography(&seeds, track_config, fod_peak_direction_field(&fod))?;
```

### Whole-brain fields

`fod_volume_direction_field` and `noddi_direction_field` perform spatial
neighbourhood lookups at each integration step — they query a 3-D volume
of pre-computed model fits rather than a single voxel.

`fod_volume_direction_field` trilinearly interpolates fODF coefficients
from the surrounding 2×2×2 voxel neighbourhood at each step, then extracts
the strongest peak via a 50×100 spherical-grid search with a 10%
relative-amplitude threshold:

```rust,ignore
use ritk_tractography::fod_volume_direction_field;

let result = euler_tractography(&seeds, track_config, fod_volume_direction_field(&volume))?;
```

`noddi_direction_field` uses nearest-neighbour spatial lookup — NODDI
intrinsically yields a single fibre orientation per voxel, so no peak
extraction is needed:

```rust,ignore
use ritk_tractography::noddi_direction_field;

let result = euler_tractography(&seeds, track_config, noddi_direction_field(&volume))?;
```

### Custom fields

Any closure satisfying `Fn(&Point<3>) -> Option<Vector<3>>` is a valid
direction field. The closure defines the tracking domain: return `None`
outside the domain boundary and a unit `Vector<3>` inside it:

```rust,ignore
let result = euler_tractography(&seeds, config, |point| {
    let [x, y, z] = point.to_array();
    // Track only within a 10 mm radius sphere.
    let r_sq = x * x + y * y + z * z;
    if r_sq > 100.0 {
        return None;
    }
    // Constant horizontal field — every trackable point yields +x.
    Some(Vector::new([1.0, 0.0, 0.0]))
})?;
```

The integrator validates unit-norm at every sample; a non-unit direction
is a typed error, not silently corrected.

## Streamline export

`TractographyResult` provides export methods for all three tractogram
interchange formats. Tractography points are in physical millimetre
coordinates (the native coordinate system for all three formats).

### .trk — DSI Studio / TrackVis

`to_trk` writes the header with an identity voxel-to-RAS affine. Callers
that need anatomical space should set `dim` and `voxel_size` to match the
reference image:

```rust,ignore
let trk = result.to_trk(
    [128, 128, 60],   // dim
    [1.5, 1.5, 2.0],  // voxel_size (mm)
);
// trk can be written via ritk_trk::write_trk(&trk, path)?;
```

`to_trk_header` accepts an optional custom `vox_to_ras` affine:

```rust,ignore
let trk = result.to_trk_header(dim, voxel_size, Some(vox_to_ras_affine));
```

`to_trk_with_scalars` attaches per-point scalar values (e.g. FA, MD)
for DSI Studio colour-coding. Each inner `Box<[f32]>` must contain
`n_points × n_scalars` values in row-major (per-point scalar stride)
order:

```rust,ignore
let trk = result.to_trk_with_scalars(
    dim, voxel_size,
    &["FA", "MD"],
    fa_md_scalars,
);
```

### .tck — MRtrix3

`to_tck` writes with a default header (Float32LE datatype, no transform):

```rust,ignore
let tck = result.to_tck();
// tck can be written via ritk_tck::write_tck(&tck, path)?;
```

`to_tck_header` accepts optional `mrtrix_version`, `comments`, and a
4×4 voxel-to-scanner transform:

```rust,ignore
let tck = result.to_tck_header(
    Some("3.0.4".into()),
    Some("RITK Euler tractography".into()),
    Some(transform_matrix),
);
```

Per-point scalars are not stored natively in the `.tck` format; use the
MRtrix3 `tckmap` weights sidecar (`ritk_tck::write_tck_weights`) when
scalar export is needed.

### .trx — Tractography Reference eXchange

`to_trx` writes with a default header (`"float32"` dtype):

```rust,ignore
let trx = result.to_trx();
// trx can be written as a directory via ritk_trx::write_dir(&trx, path)?;
```

`to_trx_with_dpv` attaches per-vertex data arrays (DPV) such as FA and
MD. The caller must populate the header's `dpv` map with `TrxArrayDef`
entries declaring each array's dtype and component count, and provide the
corresponding raw encoded byte buffers:

```rust,ignore
use std::collections::HashMap;
let mut trx = result.to_trx_with_dpv(HashMap::new());
trx.header.dpv.insert(
    "FA".into(),
    ritk_trx::TrxArrayDef { dtype: "float32".into(), n_components: 1 },
);
trx.dpv_data.insert("FA".into(), fa_bytes);
```

### Format comparison

| Property | `.trk` | `.tck` | `.trx` |
|---|---|---|---|
| Coordinate system | Voxel (affine → RAS) | Scanner mm | Physical mm |
| Per-point scalars | Yes (header `n_scalars`) | No (tckmap weights sidecar) | Yes (`dpv` arrays) |
| Per-streamline properties | Yes | No | Yes (`dps` arrays) |
| Binary layout | Single file, binary | Single file, binary | Directory of .raw files + JSON header |

## Error types

`TractographyError` covers every failure mode with seed and step indices
for localisation:

| Variant | Condition | Index context |
|---|---|---|
| `InvalidStepSize` | `step_size` is NaN, ∞, zero, or negative | — (config validation) |
| `InvalidMaxSteps` | `max_steps` is zero or overflows `usize` | — |
| `InvalidTurnLimit` | Turn angle not in `[0°, 180°]` | — |
| `InvalidDirection` | Field returned non-finite or non-unit vector | `(seed_index, step_index)` |
| `NonFinitePoint` | A proposed point became NaN or ∞ | `(seed_index, step_index)` |
| `Allocation` | Vec pre-allocation failed | Requested capacity |
| `Geometry` | Gaia rejected the generated polyline | Propagated from `PolylineError` |

A seed at which the field returns `None`, or whose integration produces
fewer than two points, is an expected untrackable seed — it produces no
line and is not an error.

## Output types

`Streamline` pairs Gaia polyline geometry with termination diagnostics:

| Method | Returns |
|---|---|
| `geometry()` | `&Polyline<f64>` |
| `forward_termination()` | `TerminationReason` |
| `backward_termination()` | `Option<TerminationReason>` — `None` for forward-only |

`TractographyResult` aggregates the output of one call to
`euler_tractography`:

| Method | Returns |
|---|---|
| `streamlines()` | `&[Streamline]` — in seed order, excluding untrackable seeds |
| `seeds_attempted()` | Total input seeds queried |
| `streamlines_generated()` | Number of `Streamline` values |

## What the current algorithm establishes

The implementation and tests establish deterministic geometry, sign
continuity, bounded memory growth, explicit stopping reasons, and rejection
of invalid field samples. They do not establish anatomical validity,
clinical utility, crossing-fibre resolution, uncertainty, or invariance to
acquisition and preprocessing choices. Such claims require independent
phantoms and validated in-vivo protocols.

The [signal-to-streamlines example](examples/diffusion_tractography.md)
closes the loop end-to-end: known tensor → synthetic signals → model fit
→ direction field → streamlines. Each visual element — seeds, orientation
glyphs, domain boundaries, generated trajectories — has a defined meaning.
