# ADR 0029: DICOM RGB display preservation

Status: Accepted

Date: 2026-09-10

Driver: [RITK-SNAP-COLOR-001](../../backlog.md#RITK-SNAP-COLOR-001).

## Context

The DICOM boundary already decodes admitted RGB objects into interleaved
`f32` samples with three channels per voxel. The viewer's slice extractor
selected only the first channel, and scalar window/level rendering therefore
turned a color image into a misleading grayscale or colormap display. The
scalar MIP, volume-rendering, GPU, and fused-compare paths had the same
first-channel risk.

## Decision

`LoadedVolume` retains its `[depth, rows, columns, channels]` storage contract
and exposes channel-preserving slice extraction alongside the existing scalar
extraction used by scalar algorithms. `SliceRenderer` dispatches on the
validated channel count: one channel follows DICOM window/level and Iris
colormap mapping; three channels copy finite integral samples in `[0, 255]`
directly into opaque RGBA pixels. RGB rendering never applies scalar
windowing, a colormap, modality rescale, or a first-channel reduction.

Scalar MIP, volume rendering, GPU projections, and fused compare reject
non-scalar volumes with an explicit diagnostic rather than silently selecting a
channel. The RITK IO color readers remain the trust boundary and reject scalar,
palette, YBR, CMYK, planar, signed, unsupported codec, and malformed RGB
objects before a `LoadedVolume` is constructed.

## Alternatives

Passing RGB samples through scalar window/level is rejected because it destroys
channel identity and can make a red or blue voxel appear unrelated to its
source. Converting RGB to grayscale is rejected because the viewer would lose
diagnostic color information. Creating a second volume carrier is rejected
because it duplicates the existing validated geometry and metadata boundary;
the channel dimension already expresses the real variation.

## Verification

The loader tests decode a synthetic two-frame RGB Part 10 object through both
filesystem and dropped-byte paths, assert exact interleaved samples, and check
red, green, blue, white, cyan, magenta, yellow, and neutral voxels in axial,
coronal, and sagittal renders. The allocating and scratch render paths are
pixel-identical. Existing RITK IO tests cover planar and unsupported RGB
inputs. The bounded `dicom_workflow` example emits three RGB captures, and
`scripts/viewer.py` compares their enlarged pixel grids byte-for-byte with the
manual images.
