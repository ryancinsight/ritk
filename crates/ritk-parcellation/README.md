# ritk-parcellation

Anatomical parcellation volumes: the label map that says which region every
voxel belongs to, and the measures taken from it.

A parcellation is the shared vocabulary between whatever *produces* one — an
atlas registered onto a subject, a segmentation, a surface annotation rasterised
into a volume — and whatever *consumes* one: connectome construction,
region-wise statistics, targeting. This crate owns the type and depends on
nothing but spatial primitives, so a consumer does not inherit a registration
stack to hold a label volume.

## What it provides

- **`Parcellation`** — a labelled volume with region names, label remapping (to
  coarsen a fine atlas), and subsetting.
- **`ParcellationGrid`** — the voxel-to-physical affine, *including the
  direction cosines*. Almost no acquired volume is axis aligned, and dropping
  the obliquity does not fail loudly: it returns the label of a different
  region, and every label involved is legitimate.
- **`RegionStatistics`** — per-region volume, centroid, and extent in one pass
  over the volume. Region volume is what normalises a connectome, since a larger
  region attracts more streamline endpoints for reasons of geometry rather than
  anatomy.
- **`NearestLabelSearch`** — the nearest labelled voxel to a point, within a
  radius. Streamlines terminate at the grey/white boundary while a cortical
  parcellation labels only grey matter, so an exact endpoint lookup discards
  most of a tractogram.
- **`freesurfer`** — colour lookup tables, surface annotations, the binary
  triangle surface format, and cortical-ribbon rasterisation from a surface
  annotation into a volumetric parcellation.

## Example

```rust
use ritk_parcellation::{Parcellation, ParcellationGrid};
use ritk_spatial::Point;

let grid = ParcellationGrid::axis_aligned([2, 1, 1], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0])?;
let parcellation = Parcellation::new(
    Box::new([1, 7]),
    grid,
    vec![(1, "Left".into()), (7, "Right".into())],
)?;

assert_eq!(parcellation.label_at(&Point::new([2.0, 0.0, 0.0])), Some(7));
assert_eq!(parcellation.name_of(7), Some("Right"));
# Ok::<(), ritk_parcellation::ParcellationError>(())
```

## Related crates

- [`ritk-connectome`](../ritk-connectome) builds and measures the region graph.
- [`ritk-registration`](../ritk-registration) produces a parcellation by warping
  a labelled atlas onto a subject.

## Documentation

API documentation: <https://docs.rs/ritk-parcellation>.
The RITK book: <https://ryancinsight.github.io/ritk/>.

## Licence

MIT OR Apache-2.0.
