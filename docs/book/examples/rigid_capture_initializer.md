# Example: Robust Rigid Capture Initializer

This example turns bidirectional block correspondences into a full fixed-to-
moving rigid transform, then uses that transform as the zero residual of the
bounded six-parameter search. It manufactures a known 20-degree rotation and
translation, injects inconsistent correspondences below the 50% breakdown
boundary, and checks the recovered transform against the analytical result.

The two correspondence slices have explicit directions. Forward entries store
fixed-to-moving matches. Reverse entries store moving-to-fixed matches. RITK
normalizes the reverse pairs before one joint least-trimmed-squares fit. For a
rigid transform this is symmetric because rotation preserves Euclidean
residual norms. The estimator retains half of the combined pairs and returns
the retained root-mean-square residual as a diagnostic.

`RigidSearchAnchor` validates that the initializer is finite, homogeneous,
orthonormal, and proper. Search parameters then describe only the residual
rotation and translation around that anchor; all existing global and local
bounds remain active. A zero residual reproduces the supplied transform.

```rust,ignore
{{#include ../../../crates/ritk-registration/examples/book_rigid_initializer.rs}}
```

Run the same source used by this page:

```bash
cargo run -p ritk-registration --example book_rigid_initializer
```

The example is synthetic by design: it verifies the estimator and transform
composition independently of an image metric. A CT/MR application still has
to generate correspondences from physical image grids, reject low-confidence
matches, and validate the final pose with held-out landmarks or segmentations.
