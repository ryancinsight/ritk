# Example: Windowing and Rescaling

CT voxels carry Hounsfield units (HU), but the useful display range depends on
the tissue being inspected. Windowing is a saturating affine map:

```text
W(I) = clamp(I, a, b)
O(I) = (W(I) − a) / (b − a) · (o_max − o_min) + o_min
```

Values below `a` map to `o_min`, values above `b` map to `o_max`, and values
inside the window retain their relative order. Rescaling is different: it
uses the global image minimum and maximum as `a` and `b`, so it changes the
representation range without selecting a tissue-specific interval.

The runnable example applies both native filters to the real RIRE Patient 001
CT volume. The generated figure is deliberately labeled and uses one fixed
display convention per panel:

![RIRE CT windowing and rescaling figure with labeled output panels and an HU histogram](../figures/windowing_rescale.svg)

1. **Input CT** uses a fixed `[-1000, 1000]` HU display window.
2. **Soft-tissue window** applies `[-160, 240]` HU and emits `[0, 1]`.
3. **Lung window** applies `[-1000, 400]` HU and emits `[0, 1]`.
4. **Global rescale** uses the observed CT extrema and emits `[0, 255]`.
5. **Input distribution** plots the same axial slice and marks both HU windows.
6. **Filter contract** states the saturation and output-range behavior next to
   the image panels.

The histogram prevents a common interpretation error: the soft-tissue and lung
panels are not alternative contrast adjustments applied after rendering. They
are different numeric maps of the same CT slice, and the marked intervals show
which input HU values are expanded or saturated.

## Source and command

Source: `crates/ritk-io/examples/book_windowing_rescale.rs`

```text
cargo run -p ritk-io --example book_windowing_rescale -- \
  docs/book/figures/windowing_rescale.svg
```

The source image, filtered outputs, and histogram all come from the in-tree
RIRE fixture. The example uses the Coeus-native path and does not write a
second copy of the image through a separate display library.

## Verification

- Each filter preserves the CT shape and physical metadata through the native
  image boundary.
- Windowed values stay inside `[0, 1]`; the global rescale stays inside
  `[0, 255]`.
- The figure is regenerated from the actual source and filter outputs.
- Analytical endpoint and saturation behavior is covered by the filter's
  native tests; the figure checks the real-data visual behavior separately.
