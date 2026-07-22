# Example: Thresholding

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/thresholding.rs` *(not yet created)*

## Description

This planned example will cover several thresholding styles commonly needed in medical and scientific imaging: manual binary thresholding, Otsu-style automatic threshold selection, and multi-level segmentation cuts. The intended focus is on the ritk intensity filter surface around `ThresholdImageFilter`, `BinaryThresholdImageFilter`, and related threshold modes, using a `ritk-image::Image` as the stable boundary object throughout. By keeping the example on scalar data, the page can emphasize the contract that thresholding changes voxel classes but not image geometry.

Atlas integration is relevant here because thresholding often becomes the handoff between raw I/O and more complex pipelines such as morphology, distance transforms, or registration masks. The eventual example should run entirely within the Coeus-backed image layer, leaving file format and backend choice outside the core logic. That makes it useful both as a quick segmentation primitive and as a validation tool for understanding histogram structure before choosing a registration metric.

## Planned workflow

- Load or synthesize a scalar volume with a clear foreground/background split.
- Apply a manual threshold and compare inside/outside assignments.
- Compute an Otsu threshold and compare it to the manual choice.
- Demonstrate multi-level thresholds for coarse tissue classes.

## Verification goals

- Threshold outputs contain only the requested class values.
- Changing the cutoff changes labels, not geometry.
- Automatic thresholds are reproducible on the same input volume.
