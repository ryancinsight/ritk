# NRRD Format Boundary

Single source of truth for NRRD file I/O.

## Ownership

`ritk-nrrd` owns the NRRD file reader and writer. `ritk-io::format::nrrd`
is a facade re-export.

## Spatial Contract

NRRD file-axis `[x,y,z]` maps to RITK `[depth,row,col]` via `crates/ritk-nrrd/src/spatial.rs`.
The reader constructs the tensor directly as `[nz,ny,nx]` from X-fastest NRRD
raw bytes; the writer emits RITK ZYX flat data directly.

## Direction Vectors

- Reader: NRRD `space directions` vectors `[x,y,z]` become internal metadata
  columns `[depth,row,col] = [z,y,x]`.
- Writer: NRRD `space directions` are generated from internal columns
  `[col,row,depth]`.

## Invariant

NRRD parser/writer dependency changes stay behind `ritk-nrrd`; callers
in `ritk-io`, CLI, and viewer code consume the same authoritative API.

## Diffusion gradient metadata

`read_nrrd_gradient_scheme` implements the NA-MIC DWI convention. One nominal
`DWMRI_b-value` is combined with each `DWMRI_gradient_XXXX` squared norm to
recover the per-volume effective b-value. The measurement frame maps gradients
to world coordinates; RAS world coordinates are converted once to RITK LPS.
Missing indices, non-finite values, `DWMRI_NEX`, and B-matrix encodings fail
explicitly rather than being guessed.
