# Spatial Filtering

Convolution-based spatial filters: Gaussian smoothing, gradient magnitude,
Canny edge detection, and separable gradient computation.

## Design

All spatial filters operate on flat z-major host buffers via substrate-agnostic
pure functions. The gradient magnitude and Canny detectors share a common
`canny_edges_flat` host core that implements stages 2-4 of the Canny
algorithm (gradient computation, non-maximum suppression, hysteresis).

## Gaussian Smoothing

Separable zero-padded Gaussian smoothing using `convolve_zero_pad_3d`.
The per-axis kernel is constructed via `gaussian_kernel`; the separable
application uses `convolve_separable` from `ritk-filter`. No Coeus tensor
is constructed.

## Gradient Magnitude

Computes the magnitude of the gradient via central differences, then
optionally smooths with a discrete Gaussian kernel.

## Canny Edge Detection

Five-stage pipeline:
1. Gaussian smoothing (optional, via shared `discrete_gaussian_smooth_flat` core)
2. Gradient magnitude + direction
3. Non-maximum suppression (via shared `canny_edges_flat` core)
4. Hysteresis thresholding

## Verification

Each filter is differentially tested against its Coeus-generic counterpart
via `assert_coeus_matches_coeus`. The Canny detector is compared against
SimpleITK ground truth.
