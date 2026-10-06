# NRRD Format Boundary

Single source of truth for NRRD file I/O.

## Ownership

`ritk-nrrd` owns the NRRD file reader and writer. `ritk-io::format::nrrd`
is a facade re-export.

`read_nrrd_header` parses the bounded header without decoding the payload. Its
result exposes canonical standard fields, the declared format version,
comment lines, an effective custom key/value map, and source-ordered decoded
key/value records, including repeated keys. The effective map follows NRRD's
rule that the last value for a repeated key is the stored value; the record
list retains earlier values for inspection during format conversion. The
parser limits headers to 16 MiB and 65,536 retained entries across standard
fields, comments, and key/value records. These header records do not imply
that payload conversion preserves every format's metadata.

Field identifiers are case-insensitive. Teem's equivalent spellings
(`byteskip`, `lineskip`, `datafile`, `axismins`, `axismaxs`, `centerings`,
`blocksize`, `oldmin`, `oldmax`, and `sampleunits`) map to canonical lowercase
field names before duplicate checks. Empty comment strings are ignored;
retained comment lines keep their leading `#` and source order. NRRD header
lines use ASCII; non-ASCII bytes return a typed parse error. These delimiter,
key, field, alias, and comment rules follow the [Teem NRRD format
definition](https://teem.sourceforge.net/nrrd/format.html), Sections 1.2, 1.6,
5, and 6.

## Spatial Contract

NRRD file-axis `[x,y,z]` maps to RITK `[depth,row,col]` via `crates/ritk-nrrd/src/spatial.rs`.
The reader constructs the tensor directly as `[nz,ny,nx]` from X-fastest NRRD
raw bytes; the writer emits RITK ZYX flat data directly.

## Direction Vectors

- Reader: NRRD `space directions` vectors `[x,y,z]` become internal metadata
  columns `[depth,row,col] = [z,y,x]`.
- Writer: NRRD `space directions` are generated from internal columns
  `[col,row,depth]`.

Every finite positive spacing retains its direction, including values below
`1e-9`. Stored-volume construction rejects an axis component whose
direction-times-spacing product overflows or underflows to zero, before a
format writer can emit unrepresentable geometry.

A rank-2 array can be embedded in either a two- or three-dimensional world.
For a named patient space such as LPS or RAS, each direction vector and the
origin use three world coordinates even though the array has two axes. RITK
derives the missing slice direction from the plane normal and promotes the
array to `[1,Y,X]` with a one-millimeter through-plane step. When no named
space is declared, rank-2 vectors retain the two-component interpretation and
are promoted into the XY plane. Component count follows the world-space
dimension, independently of array rank, as specified by [Teem's NRRD space
fields](https://teem.sourceforge.net/nrrd/format.html#space).

NRRD's named patient spaces are normalized to RITK's LPS millimeter
coordinates. RAS reverses the first two physical components; LAS reverses the
anterior/posterior component. Supported `space units` (`mm`, `cm`, `m`, `um`,
and `nm`) scale both origins and direction vectors before spacing and
orientation are derived. Unknown spaces, anonymous coordinate frames, and
unrecognized units fail instead of being relabeled as patient coordinates.
These mappings follow [Section 4 of the Teem NRRD format specification](https://teem.sourceforge.net/nrrd/format.html).

## Invariant

NRRD parser/writer dependency changes stay behind `ritk-nrrd`; callers
in `ritk-io`, CLI, and viewer code consume the same authoritative API.

## Diffusion gradient metadata

`read_nrrd_gradient_scheme` implements the NA-MIC DWI convention. One nominal
`DWMRI_b-value` is combined with each `DWMRI_gradient_XXXX` squared norm to
recover the per-volume effective b-value. The measurement frame maps gradient
coordinates to world coordinates; RAS world coordinates are converted once to
RITK LPS. Although NRRD makes this field optional, a nonzero gradient without
it has no defined coordinate mapping. RITK rejects that input instead of
assuming the gradient frame matches the image orientation; an all-zero baseline
does not need a gradient-frame mapping. See [Teem's NRRD specification,
section 4](https://teem.sourceforge.net/nrrd/format.html) for the coordinate-
frame contract.
Every encoded nonzero weighting is preserved, including values below the
scanner-input baseline threshold used by `ritk-diffusion-scheme` constructors.
Missing indices, non-finite values, `DWMRI_NEX`, and B-matrix encodings fail
explicitly rather than being guessed.

## Stored samples and format conversion

`NrrdDocument` combines a validated `StoredSeries` with retained comments and
custom records without an intermediate file. Writing derives structural fields
from the samples, validates first, and leaves an existing destination
unchanged on rejection.

Use `read_nrrd_stored` when NRRD is an input to a format conversion. It
returns `ritk_image_io::StoredVolume`, retaining the element type, each stored
value, the spatial metadata, the coordinate map, and the calibration state.
The ordinary `read_nrrd` API remains the compute path and converts values to
`f32`.

The stored reader supports signed and unsigned 8-, 16-, 32-, and 64-bit
integers plus IEEE 754 32- and 64-bit floats. It accepts the aliases listed
in the NRRD type table and both endian markers. NRRD's `encoding` field is
required. RITK reads `raw`, `ascii` (`text`, `txt`), and `gzip` (`gz`). Binary
multi-byte samples require an explicit endian marker; one-byte samples and
ASCII samples do not, as specified in Section 5 of the [Teem NRRD
format](https://teem.sourceforge.net/nrrd/format.html). ASCII values are
whitespace-delimited and each token is limited to 128 bytes to bound parser
scratch space. Readers consume the declared array payload and ignore following
bytes, as permitted by the NRRD encoding rules in Section 5.
Binary reads and writes retain signed zero and NaN payload bits. ASCII values
are parsed from text, so they do not carry binary NaN payload bits. Element type
names follow the [Teem NRRD type table and payload rules](https://teem.sourceforge.net/nrrd/format.html),
including the standard `int8` through `uint64` aliases. RITK also accepts the
historical `char` alias as signed 8-bit for existing files, although Teem does
not list bare `char` as a NRRD type descriptor.

`read_nrrd_stored_series` returns one stored volume per acquisition entry. It
preserves acquisition order for both leading interleaved axes and trailing
contiguous axes. `write_nrrd_stored_series` writes the trailing contiguous
layout; the existing compute-image series writer keeps its leading NA-MIC
layout.

Diffusion series output uses the NRRD0005 magic and writes an explicit identity
measurement frame because its gradient vectors are already in LPS coordinates.
Other stored series retain NRRD0004. The reader accepts the standard
`axismins`, `axismaxs`, and `centerings` aliases and canonicalizes them before
checking duplicate fields. Per-axis physical units without `space directions`
or `spacings` are rejected because they cannot determine sample geometry.
Stored scalar or list data with a `measurement frame` but no DWMRI scheme is
rejected before payload reading because the stored-volume model cannot retain
that frame. Axis support bounds and cell/node centering are likewise rejected
instead of being discarded.

The custom RITK coordinate-map field serializes Cartesian, curvilinear,
phased-array, and per-slice transform maps. Slice-series transforms store nine
row-major direction components and three translation coordinates per depth
slice, with round-trip decimal formatting.

NRRD does not have a standard modality-calibration field. The stored writer
therefore accepts only value-identity calibration and returns
`NrrdStoredWriteError::UnsupportedCalibration` before it creates an output
file when the volume carries a non-identity linear or lookup-table transform.
Convert to a target format that can encode the calibration, or apply it
explicitly before choosing a format whose contract stores only pixel values.

Detached data files must be relative to the NRRD header's directory. Absolute
paths and parent traversal are rejected at the read boundary; raw and
decompressed payloads are bounded by the declared shape and element type.
Detached data currently accepts one file name. `line skip` and `byte skip`
fields are parsed before payload decoding; positive byte skips on gzip data
apply after decompression, while `byte skip: -1` is accepted only for raw data.

Stored reads accept an `ImageReadBudget` that caps encoded payload bytes,
decoded sample bytes, gzip-expanded payload bytes including discarded byte-skip
data, and the number of volumes in an acquisition series.
The default limits are 1 GiB for each byte count and 65,536 volumes. The
adapter checks declared counts before allocating decoded sample storage. The
stored writer validates format semantics before opening the destination and
streams samples through a buffered file writer without making a second
volume-sized encoded payload.
