# Deterministic Streamline Tractography

Tractography integrates a local orientation field into curves. It does not
directly reconstruct axons and it does not resolve the biological cause of a
local orientation. RITK's current `ritk-tractography` algorithm is a
deterministic Euler baseline whose output is Gaia polyline geometry.

## Integration rule

For a physical seed point \(\mathbf x_0\), step size \(h>0\), and local unit
orientation \(\mathbf v(\mathbf x)\), one forward step is

\[
\mathbf x_{k+1}=\mathbf x_k+h\,\mathbf v(\mathbf x_k).
\]

Diffusion orientations are antipodally symmetric: \(\mathbf v\) and
\(-\mathbf v\) describe the same axis. At each sample RITK flips the returned
orientation when needed so its dot product with the preceding direction is
nonnegative. This prevents arbitrary sign choices in an ODF peak extractor
from making a streamline reverse direction.

Bidirectional tracking integrates both signs from the seed, reverses the
backward half, and joins the two halves with the seed exactly once.

## Termination is part of the result

A half-streamline records one of three reasons:

- `FieldBoundary`: the proposed point has no trackable orientation;
- `TurningAngle`: the next orientation exceeds the configured turn limit; or
- `StepLimit`: the bounded accepted-step count is exhausted.

The first out-of-domain proposal is not appended. A malformed field direction
(NaN, infinity, or non-unit norm) is a typed error rather than a silently
dropped streamline. Configuration validates finite positive step size,
nonzero bounded step count, and a finite turn limit in \([0,180]\) degrees.

~~~rust,ignore
use ritk_spatial::{Point, Vector};
use ritk_tractography::{
    TrackingDirection, TractographyConfig, euler_tractography,
};

let seeds = [Point::new([0.0, 0.0, 0.0])];
let config = TractographyConfig::new(
    0.5,
    20,
    45.0,
    TrackingDirection::Bidirectional,
)?;
let result = euler_tractography(&seeds, config, |point| {
    (point[0].abs() <= 4.0).then_some(Vector::new([1.0, 0.0, 0.0]))
})?;
assert_eq!(result.streamlines_generated(), 1);
# Ok::<(), ritk_tractography::TractographyError>(())
~~~

## What the current algorithm establishes

The implementation and tests establish deterministic geometry, sign
continuity, bounded memory growth, explicit stopping reasons, and rejection of
invalid field samples. They do not establish anatomical validity, clinical
utility, crossing-fiber resolution, uncertainty, or invariance to acquisition
and preprocessing choices. Such claims require independent phantoms and
validated in-vivo protocols.

The [worked example](examples/diffusion_tractography.md) separates local
direction estimation from streamline integration and overlays seeds,
orientation glyphs, domain boundaries, and generated trajectories so each
visual element has a defined meaning.
