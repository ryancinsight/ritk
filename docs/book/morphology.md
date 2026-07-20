# Morphological Operations

Binary and grayscale morphological filters: erosion, dilation, opening,
closing, top-hat transforms, hit-or-miss, and morphological reconstruction.

## Design

All morphological operations operate on flat host buffers via substrate-agnostic
pure functions. Binary operations use safe-border padding (zero padding at
boundaries); grayscale operations preserve spatial metadata.

## Binary Morphology

- `BinaryErodeFilter`, `BinaryDilateFilter` — structuring element via
  connectivity (6-connectivity for 3D)
- `BinaryFillholeFilter` — flood-fill based hole filling
- `BinaryMorphologicalClosing`, `BinaryMorphologicalOpening` —
  erosion followed by dilation (closing) or dilation followed by erosion (opening)

## Grayscale Morphology

- `GrayscaleErosion`, `GrayscaleDilation` — min/max filter with
  safe-border padding
- `GrayscaleClosingFilter`, `GrayscaleOpeningFilter` — anti-extensive/
  extensive operations
- `WhiteTopHatFilter`, `BlackTopHatFilter` — residue of opening/closing
- `GrayscaleGradientFilter` — erosion minus dilation

## Reconstruction

- `GrayscaleGeodesicErosion`, `GrayscaleGeodesicDilation` — geodesic
  operations using reconstruction

## Verification

Each operation is differentially tested against its Coeus-generic counterpart
via `assert_coeus_matches_coeus`. Analytical oracles verify:
- Radius 0 is identity
- All-foreground/all-background edge cases
- Opening is anti-extensive, closing is extensive
