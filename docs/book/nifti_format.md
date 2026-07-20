# NIfTI Format Boundary

Single source of truth for NIfTI file I/O.

## Ownership

`ritk-nifti` owns the NIfTI file reader and writer. `ritk-io::format::nifti`
is a facade re-export.

## Spatial Contract

NIfTI file-axis RAS maps to RITK `[depth,row,col]` via `crates/ritk-nifti/src/spatial.rs`.
The reader constructs the tensor directly as `[nz,ny,nx]` from X-fastest NIfTI
raw bytes; the writer emits RITK ZYX flat data directly.

## Affine Conversion

- Reader: file affine columns `[x,y,z]` become internal metadata columns
  `[depth,row,col] = [z,y,x]`.
- Writer: sform columns are emitted as `[internal_col, internal_row, internal_depth]`.

## Invariant

NIfTI parser/writer dependency changes stay behind `ritk-nifti`; callers
in `ritk-io`, CLI, and viewer code consume the same authoritative API.
