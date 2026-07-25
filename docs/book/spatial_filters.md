# Spatial Filtering

Convolution-based spatial filters: Gaussian smoothing, gradient magnitude,
Canny edge detection, and separable gradient computation.

## Design

All spatial filters operate on flat z-major host buffers via substrate-agnostic
pure functions. The gradient magnitude and Canny detectors share a common
host core that implements gradient computation, non-maximum suppression, and
hysteresis. Public `apply_native` methods preserve the `ritk-image::Image`
geometry boundary while executing those kernels without constructing a second
tensor representation.

## Gaussian Smoothing

`GaussianFilter` builds one normalized sampled kernel per axis from physical
sigma and voxel spacing, then applies separable zero-padded convolution. The
native path and the generic path share the kernel and host-core contracts.

## Gradient Magnitude

Computes the magnitude of the gradient via central differences, then
optionally smooths with a discrete Gaussian kernel.

## Canny Edge Detection

Four stages form the binary edge map:

1. Gaussian smoothing.
2. Gradient magnitude and direction.
3. Non-maximum suppression along the continuous gradient direction.
4. Hysteresis thresholding.

The complete filter gallery uses the real public pipeline on a deterministic
phantom and writes the figure below:

![Input, Gaussian smoothing, and Canny edge map](figures/filter_gallery.svg)

Run it from the repository root with:

```text
cargo run -p ritk-filter --example book_filter_gallery -- \
  docs/book/figures/filter_gallery.svg
```

## Verification

The filter example is source-linked from [Gaussian Smoothing](examples/gaussian_smoothing.md)
and [Canny Edge Detection](examples/canny_edges.md). Package tests provide
the value-semantic and differential coverage; the figure is a visual smoke
check of the same public calls.
