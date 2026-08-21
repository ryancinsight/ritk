# Anatomical Parcellation

A *parcellation* is a label volume: every voxel carries the identifier of the
anatomical region it belongs to, with the reserved label `0` meaning background.
It is the shared vocabulary between whatever produces one — an atlas registered
onto a subject, a segmentation, a surface annotation rasterised into a volume —
and whatever consumes one: connectome construction, region-wise statistics,
targeting.

`ritk-parcellation` owns the type and depends on nothing but spatial primitives.
`ritk-registration` produces one by warping a labelled atlas onto a subject.

## Geometry: why the direction cosines are not optional

A label volume is only useful if a physical point can be mapped to the voxel
containing it. `ParcellationGrid` carries the standard medical-imaging affine:

\\[ p = \\mathbf{o} + R \\, (\\mathbf{s} \\odot \\mathbf{i}) \\]

where \\(\\mathbf{i}\\) is the continuous voxel index, \\(\\mathbf{s}\\) the
per-axis spacing, \\(R\\) the direction cosine matrix whose columns are the
physical directions of the index axes, and \\(\\mathbf{o}\\) the physical
position of voxel \\((0,0,0)\\)'s centre.

Reducing this to \\(\\mathbf{o} + \\mathbf{s} \\odot \\mathbf{i}\\) — treating
the index axes as parallel to the physical ones — is correct only for an
axis-aligned volume. Almost no acquired volume is: a scanner stores an oblique
slice stack, and NIfTI's `qform`/`sform`, NRRD's space directions, and DICOM's
image orientation all exist to record that obliquity.

Dropping it does not fail loudly. It returns the label of a **different**
region, displaced by up to the volume's extent times the sine of the obliquity.
Connectome edges built on such lookups are wrong in a way nothing downstream
catches, because every label involved is a legitimate label.

```rust,ignore
use ritk_parcellation::{Parcellation, ParcellationGrid};

// The general case: shape, spacing, origin, and row-major direction cosines.
let grid = ParcellationGrid::new(
    [256, 256, 128],           // [nx, ny, nz]
    [1.0, 1.0, 1.5],           // spacing in mm, by spatial axis
    [-127.5, -127.5, -95.0],   // physical position of voxel (0, 0, 0)
    direction_cosines,         // [f64; 9], row-major
)?;

// The axis-aligned special case.
let grid = ParcellationGrid::axis_aligned([256, 256, 128], [1.0, 1.0, 1.5], [0.0; 3])?;
```

Labels are stored z-major — index `[ix, iy, iz]` at offset
`iz·ny·nx + iy·nx + ix`. That is this crate's contract, not a universal one:
NIfTI, NRRD, and MetaImage store volumes in this order, but a format that does
not converts at the I/O boundary rather than here.

## Construction and lookup

`Parcellation::new` rejects a label array that does not cover the grid, and one
in which every voxel is background: a parcellation with no regions cannot answer
any question asked of it, so it fails where it is built rather than returning
empty results at every call site.

```rust,ignore
let parcellation = Parcellation::new(labels, grid, region_names)?;

// Nearest-neighbour lookup at a physical point.
let label = parcellation.label_at(&point);
```

`label_at` distinguishes two answers that are often conflated:

| Result | Meaning |
|---|---|
| `None` | The point is outside the volume, or not finite |
| `Some(0)` | Inside the volume, on an unlabelled voxel |
| `Some(n)` | Inside region `n` |

Outside the field of view is not the same claim as inside it but unassigned, so
they are not collapsed. A caller counting discarded streamlines needs both.

## Region statistics

`region_statistics()` walks the volume once and returns per-region size,
position, and extent — one pass rather than one pass per region, which for a
hundred-region atlas over a million voxels is the difference between a million
reads and a hundred million.

| Measure | Meaning |
|---|---|
| `voxel_count()` | Voxels carrying the label |
| `volume()` | mm³ — voxel count times the grid's voxel volume |
| `centroid()` | Mean of the region's voxel centres, in physical space |
| `lower_index()` / `upper_index()` / `extent()` | Axis-aligned index bounding box |

Two of these are load-bearing beyond description:

**Volume** normalises a connectome. A raw streamline count between two regions
grows with how big those regions are, because a larger region presents a larger
surface for streamlines to terminate on. Comparing a raw count across regions of
different size, or across subjects whose regions differ in size, therefore
compares anatomy to arithmetic.

**Centroid** turns a region into a target point. Note that for a concave or
disconnected region the centroid can lie outside the region itself — a C-shaped
gyrus has its centre of mass in the gap. That is a property of centroids, not a
defect; a point guaranteed to be *inside* the region is a different query.

## Coarsening an atlas

A fine-grained atlas is collapsed to a coarser one through `remap_labels`, which
is a total mapping: a label the caller does not mention keeps its value, and
returning `0` removes a region. `retain_regions` keeps a named subset — the
usual preparation for a cortex-only connectome.

```rust,ignore
// Merge the two hemispheres' matching parcels.
let bilateral = parcellation.remap_labels(|label| label % 1000, names)?;

// Keep only the cortical parcels.
let cortex = parcellation.retain_regions(&cortical_labels)?;
```

## Nearest labelled voxel

Assigning a streamline endpoint to a region by reading the label directly under
it discards most of a tractogram. Streamlines are tracked through white matter
and stop where the orientation field stops being coherent, at or just short of
the grey-matter boundary — while a cortical parcellation labels the grey matter
and leaves the white matter background. The endpoint lands in a region-less
voxel and the streamline is dropped despite ending exactly where it should.

`NearestLabelSearch` recovers those. The offsets within a radius are enumerated
and sorted once, so a per-streamline loop does not re-derive the same
neighbourhood a million times.

The walk does not simply return the first labelled voxel it meets. Offsets are
ordered by the distance between voxel *centres*, while what matters is the
distance from the *point*, which sits anywhere inside its voxel — the two differ
by up to half a voxel diagonal. For an endpoint near a parcel boundary that is
exactly the difference between the parcel it is in and the one across the
border, so every candidate is scored and the nearest kept, with an early exit
once no remaining offset can improve on the best found.

```rust,ignore
use ritk_parcellation::NearestLabelSearch;

let search = NearestLabelSearch::new(parcellation.grid(), 2.0)?;
if let Some(found) = search.find(&parcellation, &endpoint) {
    // found.label, found.index, found.distance
}
```

The radius trades recall against specificity, and the trade is not free in
either direction. Too small and the tractogram is decimated. Too large and
endpoints reach past their own gyrus into a neighbouring parcel across a sulcus
— anatomically adjacent, functionally unconnected — and manufacture edges no
fibre supports. Cortical thickness is 2–4 mm and sulcal walls approach within a
millimetre of each other, so a few millimetres is where the two failure modes
balance.

The search measures distance, not connectivity: it cannot distinguish "the
nearest parcel across the sulcus" from "the parcel this fibre actually entered".
That is why the assignment reports the distance it required.

## Full-brain parcellation from an atlas

`ritk-registration` parcellates a subject by deforming an already-labelled brain
onto it and carrying the labels along.

```text
atlas intensity ──► register to subject ──► deformation
atlas labels    ──────────────────────────────┴──► warp ──► Parcellation
```

```rust,ignore
use ritk_registration::{
    AtlasParcellationConfig, LabelledAtlas, parcellate_with_atlas_set,
};

let result = parcellate_with_atlas_set(
    &subject_t1,
    &atlases,
    &AtlasParcellationConfig::default(),
)?;

let parcellation = result.parcellation;   // on the subject's own grid
let agreement = result.agreement;         // per-voxel, in [0, 1]
let quality = result.registration_quality; // final CC per atlas
```

### Running it without writing Rust

The same pipeline is a command and a Python call, so a parcellation can be
produced by whichever surface the rest of the analysis lives on.

```bash
ritk parcellate atlas --subject T1.nii.gz \
                      --atlas-intensity a1.nii.gz --atlas-labels a1_dseg.nii.gz \
                      --atlas-intensity a2.nii.gz --atlas-labels a2_dseg.nii.gz \
                      --output dseg.nii.gz --agreement agreement.nii.gz
```

```python
import ritk

result = ritk.registration.parcellate_with_atlases(
    subject, atlas_intensities, atlas_labels, fusion="majority"
)
parcellation = result.parcellation          # feeds build_connectivity_matrix
agreement = result.agreement                # [Z, Y, X], in [0, 1]
```

Every atlas must already lie on the subject's grid. All three surfaces reject a
mismatch rather than resampling it: a registration recovers a deformation, never
a resampling, so an atlas of the wrong size quietly accepted would produce
labels for a different brain.

### Labels are warped, never interpolated

Label values are identifiers, not measurements. Region 17 and region 19 do not
average to region 18, and any interpolation that produced 18 would invent an
anatomical claim out of two unrelated ones — silently, since 18 is a valid
label. Every resampling in the pipeline is nearest-neighbour for that reason.
The cost is a jagged boundary at the voxel scale, which is the honest
representation of what a label map can say.

### Why one atlas is usually not enough

A single atlas transfers not only its anatomy but its idiosyncrasies, and
wherever the registration is locally wrong, the labels are locally wrong with no
signal that they are. Registering several independently labelled brains and
fusing their votes is the standard remedy: an error must be shared by a majority
of the atlases to survive, and disagreement becomes measurable rather than
invisible.

| Fusion | Behaviour |
|---|---|
| `MajorityVote` | The label most atlases agree on; ties to the smaller label |
| `JointLabelFusion` | Weighted voting, each atlas weighted by local intensity match |

Majority voting treats every atlas as equally trustworthy everywhere — right
when the atlases are interchangeable, wrong when some registered better than
others in a particular region. Joint label fusion lets a well-registered atlas
outvote a poorly registered one *in the region where that is true*, at the cost
of a patch comparison and a small dense solve per voxel.

The returned `agreement` map is the per-voxel confidence. Low agreement marks
where the result is a coin toss between neighbouring parcels — usually the
boundaries, which is exactly where the answer matters most for a connectome,
since that is where streamlines end.

### Checking that it worked

A parcellation always looks plausible, so the question is not whether one came
back but whether it is right. The [atlas parcellation
example](examples/atlas_parcellation.md) synthesises a subject whose correct
parcellation is known, deforms three atlases onto it — one of them deliberately
mislabelled — and reports Dice against the truth alongside the agreement map.
It is also where the choice between the two fusion rules becomes a measurement
rather than a preference.

### What atlas propagation cannot do

It transfers a *predefined* parcellation. It cannot discover a region the atlas
does not contain, and it cannot represent anatomy the registration could not
reach — a resected cavity, a large lesion, or a malformation has no counterpart
in a healthy atlas, and the labels warped over it are meaningless rather than
absent. The agreement map is the only signal of that, and it is a weak one when
every atlas is equally wrong.

## FreeSurfer formats

The `freesurfer` module reads the colour lookup table and surface annotation
files.

```rust,ignore
use ritk_parcellation::freesurfer::{SurfaceAnnotation, read_freesurfer_lut};

let names = read_freesurfer_lut(lut_file)?;      // Vec<(u32, String)>
let annotation = SurfaceAnnotation::read(file)?; // per-vertex labels
```

A `SurfaceAnnotation` labels *vertices of a mesh*, not voxels. Converting one to
a volumetric `Parcellation` needs the geometry those vertices belong to —
`Surface` reads the binary triangle format — and a rasterisation of the cortical
ribbon, which `rasterise_ribbon` performs.

```rust,ignore
use ritk_parcellation::freesurfer::{Surface, rasterise_ribbon};

let white = Surface::read(std::fs::File::open("lh.white")?)?.translated(c_ras);
let pial = Surface::read(std::fs::File::open("lh.pial")?)?.translated(c_ras);

let (parcellation, report) = rasterise_ribbon(&white, &pial, &annotation, &grid, 16)?;
println!("{} of {} columns filled", report.columns - report.unfilled_columns, report.columns);
```

### The frame, which is where this goes wrong

FreeSurfer stores surfaces in **surface RAS** (tkrRAS), not the scanner frame a
volume carries. The two differ by a translation: surface RAS puts the origin at
the centre of the conformed 256³ volume, scanner RAS puts it where the scanner
did. The offset is the volume's `c_ras`, typically tens of millimetres — enough
to place a cortical ribbon outside the brain entirely, without ever failing.

The reader returns coordinates as stored and does not guess, because the surface
file does not contain the offset. `Surface::translated` applies it once known.
Rasterising with the wrong frame lands no column inside the volume, which is
rejected rather than returned as an empty parcellation.

### What the rasterisation does and does not do

The two surfaces share a vertex numbering: vertex *i* of `lh.white` and vertex
*i* of `lh.pial` are the inner and outer end of the same cortical column.
Walking that segment and stamping the vertex's label into every voxel it crosses
fills the ribbon one column at a time — the same approach as FreeSurfer's
`mri_surf2vol --fill-ribbon`.

It fills the ribbon; it does not tessellate it. A voxel no column crosses stays
background even if it lies geometrically inside the ribbon, which happens where
the mesh is coarse relative to the voxel grid. `steps` closes gaps *along* a
column and does nothing for gaps *between* columns, so a coarse mesh on a fine
grid is a limitation of the input rather than of the setting.
`RibbonReport::unfilled_columns` is the signal that the two disagree in
resolution, and `contested_voxels` counts where two parcels wanted the same
voxel — concentrated at boundaries and inside sulcal folds, which is exactly
where a connectome's endpoints land.

The `read_freesurfer_lut` table is separately usable as the `region_names` of a
volumetric parcellation from any source.

## Next

The [Connectome Construction and Graph Measures](connectome.md) chapter builds
the region graph from a parcellation and a tractogram.
