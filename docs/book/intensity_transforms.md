# Intensity Transformations

Pointwise intensity remapping filters: windowing, rescaling, thresholding,
sigmoid, arithmetic operations, and histogram equalization.

## Design

All intensity filters operate on flat host buffers via substrate-agnostic
pure functions. Each filter follows the extract → compute → reconstruct
sequence through `ritk-image::Image` boundary. No Coeus tensor is constructed
in the hot path.

## Filter Families

- **Windowing**: `IntensityWindowingFilter` — maps input range to output range
- **Rescaling**: `RescaleIntensityFilter` — affine remap with saturation clipping
- **Thresholding**: `ThresholdImageFilter` — binary threshold (inside/outside)
- **Sigmoid**: `SigmoidImageFilter` — logistic intensity remap
- **Arithmetic**: `Add`, `Subtract`, `Multiply`, `Divide` — unary/binary ops
- **Equalization**: `HistogramEqualizationFilter`, `AdaptiveHistogramEqualizationFilter`
- **Clamp/Shift-Scale**: `ClampImageFilter`, `ShiftScaleImageFilter`

## Spatial Intensity Non-Uniformity

`N4BiasFieldCorrectionFilter` estimates a smooth multiplicative MRI bias field
in log-intensity space with histogram sharpening and multi-resolution cubic
B-spline fitting. It is a spatial correction rather than a pointwise remap;
the [N4 book example](examples/n4_bias_correction.md) shows the source slice,
the corrected slice, and the estimated field on the RIRE MR fixture.

## Verification

Each filter is differentially tested against its Coeus-generic counterpart
via `assert_coeus_matches_coeus`. Analytical oracles verify endpoint mapping
and saturation behavior.
