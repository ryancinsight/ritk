# MGH and MGZ Format Boundary

MGH is FreeSurfer's native volume format. MGZ stores the same byte stream
inside gzip compression. RITK exposes both through the `ritk-mgh` crate:

- `read_mgh` reads one-volume `.mgh`, `.mgz`, and `.mgh.gz` files, with
  ASCII case-insensitive suffix matching, and rejects multi-frame input;
- `read_mgh_series` reads every frame of an acquisition or time series;
- `write_mgh` writes the representation selected by the path extension;
- `MghReader` is a stateless adapter whose `read` method receives the backend;
- `MghWriter` retains a backend for repeated operations.

FreeSurfer describes MGH as an internal format whose field contract is defined
by its reader and writer, and its tools report dimensions, frame count, voxel
size, orientation, and voxel-to-RAS transforms. See the
[FreeSurfer MGH format note](https://freesurfer.net/fswiki/FsTutorial/MghFormat),
[`mri_info` reference](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_info), and
[`mri_convert` reference](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert).

## On-disk organization

An uncompressed RITK-readable file has two regions:

```text
byte 0                                                        end of file
┌──────────────────────── 284 bytes ────────────────────────┬──────────────┐
│ version, dimensions, frames, type, DOF, RAS geometry, pad │ voxel bytes  │
└───────────────────────────────────────────────────────────┴──────────────┘
                                                               x → y → z
```

All numeric header fields and voxel scalars are big-endian. The first voxel
axis varies fastest, followed by the second and third axes. RITK maps that
ordering to the crate's `[z, y, x]` image shape without transposing the stored
voxel sequence.

MGH stores four sample types, named by the header's `type` field:

| MGH type | Stored sample | Exact reads | Cast reads |
|---|---|---|---|
| `MRI_UCHAR` | `u8` | `u8` and every wider type | any type |
| `MRI_SHORT` | big-endian `i16` | `i16`, `i32`, `i64`, `f32`, `f64` | any type |
| `MRI_INT` | big-endian `i32` | `i32`, `i64`, `f64` | any type |
| `MRI_FLOAT` | big-endian `f32` | `f32`, `f64` | any type |

`read_mgh::<T, ..>` returns an image of the caller's sample type `T` under a
conversion policy (ADR 0053). `Exact` returns the stored values or a lossless
widening and refuses a read that could change a value: an `MRI_INT` file read
as `f32` fails, because binary32 holds integers exactly only up to 2^24.
`Cast` converts and logs a warning when the stored type does not widen to `T`.
The `f32` surfaces of `ritk-io` read under `Cast`, the conversion they always
performed.

The writer stores the image's own sample type: a `u8`, `i16`, `i32`, or `f32`
image writes `MRI_UCHAR`, `MRI_SHORT`, `MRI_INT`, or `MRI_FLOAT`, and every
sample round-trips bit for bit, including signed zero. MGH has no code for the
other sample types, so the writer refuses them before creating a file; convert
such an image to one of the four first.

## Frames are a dimensional contract

The MGH header can describe several consecutive frames with common geometry.
A diffusion series or time series may therefore have `nframes > 1`. RITK's
current MGH API returns `Image<_, _, 3>`, which can represent one volume but
cannot represent a fourth frame axis.

The single-volume `read_mgh` entry point consequently requires `nframes == 1`.
It rejects a multi-frame file and names its declared frame count. Use
`read_mgh_series` when every frame of an acquisition or time series is
required. Returning only frame zero would be more dangerous than rejecting the
file: the operation would report success after silently discarding the rest of
an acquisition.

This is a type-boundary decision, not a limitation of gzip or scalar decoding.
The series API exposes that additional dimension as an ordered collection of
3-D images with shared geometry.

## RAS geometry

When `goodRASFlag == 1`, MGH stores:

- voxel spacing `D = diag(d_x, d_y, d_z)`;
- three direction-cosine columns in `Mdc`;
- the physical RAS coordinate `c_ras` of the volume center.

RITK stores the physical coordinate of voxel index zero. For dimensions
`(width, height, depth)`, define

```text
h = [(width - 1)/2, (height - 1)/2, (depth - 1)/2]ᵀ
```

Then the conversion is

```text
origin = c_ras - Mdc · D · h
```

Writing applies the inverse relation:

```text
c_ras = origin + Mdc · D · h
```

When the RAS flag is absent, the reader uses zero origin, unit spacing, and
identity direction. Applications that require scanner-space agreement should
inspect geometry before combining volumes; equal array dimensions alone do
not imply equal physical space.

## Bounded streaming decode

The reader validates version, dimensions, frame count, scalar type, and
geometry before constructing an image. It then decodes the payload through a
fixed 16 KiB input step directly into the final vector of the stored sample
type (`SampleBuffer::read_from` over consus-core's `read_extend`), and converts
that vector to `T` only when `T` differs.

The output allocation and the frame table grow only after the corresponding
input bytes have been read: each frame is appended once its bytes have arrived.
This matters for untrusted files: a header can declare a large volume or up to
`i32::MAX` frames, but a truncated payload cannot force the reader to commit
the decoded allocation, or a slot per declared frame, before proving that data
exists. A test pins this: a one-voxel file declaring `i32::MAX` frames fails at
frame 1 with the truncation error while peak allocation stays under 16 KiB. Multiplication of
dimensions and byte counts uses checked arithmetic, and allocation failure is
returned as an error.

For a 256 × 256 × 256 `MRI_FLOAT` volume, the decoded image itself is 64 MiB.
The former whole-payload path additionally retained another 64 MiB encoded
buffer while converting it. The streaming path retains approximately the
decoded output plus 16 KiB of input scratch. The output vector can have unused
geometric capacity, and a reallocation can temporarily involve both its old
and new allocations. A read into a type other than the stored one also holds
the stored vector and its converted copy until the conversion finishes. This is an allocation model, not a process-RSS claim:
allocator, backend, gzip, and image-construction state still contribute to
observed resident memory.

On the committed 128 × 128 × 64 public-reader benchmark, streaming also
reduced median read time by 11.7% for MGH and 15.9% for MGZ on the development
host. The benchmark includes file open, optional decompression, endian
conversion, and image construction; it does not isolate disk hardware or
claim the same percentage for every host.

## Failure behavior

The reader returns errors for:

- unsupported format versions or scalar type codes;
- non-positive or overflowing dimensions;
- frame counts other than one;
- truncated headers or voxel payloads;
- invalid spatial metadata or image construction;
- allocation failure.

Truncation errors name the first voxel whose complete encoded value could not
be confirmed. MGZ decompression errors retain their gzip context.

## Next

The [round-trip example](examples/mgh_roundtrip.md) writes both representations,
checks their voxel and geometry contracts, and makes the exact reconstruction
visible with an absolute-difference panel.
