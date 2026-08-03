# Example: Signal to Streamlines

This runnable example connects two independently verifiable stages: a
single-fiber diffusion-tensor signal is converted to an analytical Q-ball ODF,
then a separate curved unit-vector field is integrated into streamlines.

![Directional diffusion acquisition, analytical Q-ball ODF, and bounded streamline integration](../figures/diffusion_tractography.svg)

Read the numbered panels from left to right:

1. Blue points are the 48 actual unit gradient directions supplied to
   `GradientScheme`, all at \(b=1500\) s/mm². Two additional b0 measurements
   establish \(S_0\).
2. Orange points show the generated tensor signal against alignment with the
   known fiber axis. Stronger alignment causes greater attenuation. The blue
   polar shape is the ODF evaluated from RITK's fitted coefficients; the dashed
   red line is the independent analytical axis. The reported angular error is
   computed before the SVG is written.
3. Gray line segments are local direction samples, orange points are seeds,
   dashed curves bound the trackable field, and blue curves are Gaia polylines
   returned by `euler_tractography`. No curve contains the first proposal
   outside the dashed boundary.

The ODF panel and tractography panel intentionally use separate fields. This
prevents the figure from implying that one voxel's ODF is enough to form a
whole tract: tractography requires a spatial field of local orientations.

## Source and command

Source: `crates/ritk-diffusion/examples/book_diffusion_tractography.rs`

```text
cargo run -p ritk-diffusion --example book_diffusion_tractography -- \
  docs/book/figures/diffusion_tractography.svg
```

The example fails unless:

- a deterministic one-degree full-sphere search places the analytical Q-ball
  peak within two degrees of the known antipodal x axis;
- all five seeds produce streamlines;
- every emitted point remains inside the analytical vector-field domain; and
- the figure can be written without an unbounded or infallible allocation
  assumption in the library path.

This synthetic result verifies the stated numerical and geometry contracts.
It is not a claim of tractography accuracy on patient data.

## Scalar export

The example also estimates a diffusion tensor from the same signals to
compute FA and MD, then exports the tractography result in all three
interchange formats:

- **`.trk`** — with per-point FA and MD scalars for DSI Studio colour-coding
  via `to_trk_with_scalars`.
- **`.tck`** — plain streamline geometry via `to_tck_header`, with provenance
  metadata (MRtrix version and a comment).
- **`.tck` weights sidecar** — a FA scalar file (`.tsf`) via
  `write_tck_weights`, compatible with MRtrix3 `tckmap` and `mrview`
  `-tck_weights_in`.

```rust,ignore
let dti_tensor = estimate_dti(&scheme, &signals, DtiConfig::default())?;
let fa = dti_tensor.fa();
let md = dti_tensor.md();

// .trk with FA + MD interleaved per point.
let fa_md_scalars: Vec<Box<[f32]>> = result
    .streamlines()
    .iter()
    .map(|s| {
        let n = s.geometry().len();
        let mut scalars = Vec::with_capacity(n * 2);
        for _ in 0..n {
            scalars.push(fa as f32);
            scalars.push(md as f32);
        }
        scalars.into_boxed_slice()
    })
    .collect();
let trk = result.to_trk_with_scalars(
    [128, 128, 1], [1.0, 1.0, 1.0], &["FA", "MD"], fa_md_scalars,
);
trk.write(&mut file)?;

// .tck with provenance metadata.
let tck = result.to_tck_header(
    Some("3.0.4".into()),
    Some("RITK book example: synthetic tensor → Q-ball → Euler tractography".into()),
    None,
);
tck.write(&mut tck_file)?;

// .tck weights sidecar — one FA scalar per point.
let fa_only: Vec<Box<[f32]>> = result
    .streamlines()
    .iter()
    .map(|s| {
        let n = s.geometry().len();
        (0..n).map(|_| fa as f32).collect::<Vec<_>>().into_boxed_slice()
    })
    .collect();
ritk_tck::write_tck_weights(&fa_only, ritk_tck::TckDatatype::Float32LE, &mut tsf_file)?;
```

All three files are written alongside the SVG figure:

| File | Content |
|---|---|
| `diffusion_tractography.trk` | Streamlines + per-point FA/MD scalars |
| `diffusion_tractography.tck` | Streamlines + provenance header |
| `diffusion_tractography.tsf` | FA weights sidecar (tckmap compatible) |

The `.trk` file can be opened in DSI Studio or TrackVis; the `.tck` file and
`.tsf` sidecar can be loaded in MRtrix3's `mrview` with `-tck_weights_in`.
