# Example: GrowCut from Sparse Seeds

The runnable example constructs a deterministic two-tissue image with a
circular target, places a `3 × 3` background seed and a `3 × 3`
target seed, and executes `GrowCutFilter::apply_native`.

![GrowCut labels spreading from sparse seeds and stopping at an intensity boundary](../figures/growcut.svg)

Read the four numbered panels from left to right:

- orange marks the background label and cyan marks the circular target label;
- the first panel overlays only the two `3 × 3` seed regions on the
  grayscale input;
- the next two panels are actual `GrowCutFilter` outputs after 8 and 40
  synchronous sweeps, not illustrative drawings;
- the last panel is the converged RITK output. It reports foreground Dice and
  the exact label-error count against the analytical circle.

At each sweep, labeled voxels try to transfer their label to face-connected
neighbors. A transfer is strong when the two intensities are similar. In this
phantom, equal-intensity neighbors have `g = 1`, so both colored fronts spread
without losing confidence. At the circular boundary the intensity difference
equals the complete image range, giving `g = 0`; neither label can attack
across it. The calculation under the panels uses these exact phantom values.

Pixels that retain the grayscale input in the two middle panels remain
undecided at that iteration. They are not a third output class.

## Source and command

Source: `crates/ritk-segmentation/examples/book_growcut.rs`

```text
cargo run -p ritk-segmentation --example book_growcut -- \
  docs/book/figures/growcut.svg
```

The example fails unless GrowCut:

- preserves shape, origin, spacing, and direction;
- labels every voxel exactly as the analytical circle specifies;
- returns zero false labels; and
- produces foreground Dice `= 1`.

The truth image remains an independent oracle generated from the circle
equation, even though it is no longer a standalone panel. The analytical
phantom verifies label propagation, competition at a high-contrast boundary,
and spatial metadata. It does not establish clinical accuracy on heterogeneous
anatomy.
