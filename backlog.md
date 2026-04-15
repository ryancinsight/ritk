## Sprint 11 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| FLT-CAD | Curvature Anisotropic Diffusion | COMPLETED | Medium | Alvarez et al. 1992 mean curvature motion; `ritk-core/src/filter/diffusion/curvature.rs` |
| FLT-SATO | Sato Line Filter | COMPLETED | Medium | Multi-scale Hessian line detection (Sato 1998); `ritk-core/src/filter/vesselness/sato.rs` |
| FLT-HESS | Hessian Module | COMPLETED | Medium | 3-D physical-space Hessian + Cardano eigenvalue solver; `ritk-core/src/filter/vesselness/hessian.rs` |

---

## Sprint 10 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| SEG-CC | Confidence Connected Region Growing | COMPLETED | Medium | Yanowitz/Bruckstein adaptive statistics; `ritk-core/src/segmentation/region_growing/confidence_connected.rs` |
| SEG-NC | Neighborhood Connected Region Growing | COMPLETED | Medium | Rectangular neighborhood admissibility predicate; `ritk-core/src/segmentation/region_growing/neighborhood_connected.rs` |
| SEG-SK | Skeletonization | COMPLETED | Low | Topology-preserving thinning (Zhang–Suen 2-D, directional 3-D); `ritk-core/src/segmentation/morphology/skeletonization.rs` |

---