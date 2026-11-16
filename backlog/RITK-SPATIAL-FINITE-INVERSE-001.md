<a id="RITK-SPATIAL-FINITE-INVERSE-001"></a>

## RITK-SPATIAL-FINITE-INVERSE-001 — Reject non-finite direction inversion [correctness] [patch] — todo
- milestone: M-viewer-reading
- outcome: `Direction::try_inverse` returns `None` for non-finite input or arithmetic, and Cartesian geometry reports its existing typed error instead of retaining an invalid transform pair.
- acceptance: `cargo nextest run -p ritk-spatial -E 'test(non_finite_direction_entries_are_rejected_in_every_position) or test(normalization_overflow_is_rejected) or test(elimination_overflow_is_rejected) or test(inverse_only_elimination_overflow_is_rejected) or test(a_non_finite_direction_is_reported_as_a_singular_direction) or test(exact_pivoted_shear_keeps_its_exact_inverse) or test(extreme_finite_diagonal_keeps_its_exact_subnormal_inverse) or test(small_diagonal_above_threshold_keeps_its_exact_reciprocal) or test(small_diagonal_below_threshold_stays_singular)'` passes with the singular threshold and elimination order unchanged.
- scope: `crates/ritk-spatial/src/direction.rs`, `crates/ritk-spatial/src/grid_geometry.rs`, `crates/ritk-spatial/src/tests_direction_inverse.rs`, `crates/ritk-spatial/src/tests_grid_geometry.rs`, `crates/ritk-spatial/Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`
- next: Complete pull request #941 at its exact head and advance the Atlas gitlink after the member change lands.
- basis: 3ae7003d451f23b3eaecdf50cba14a3f3bd523c8
- needs: none
- priority: correctness
- links: [ATLAS-RITK-SPATIAL-FINITE-INVERSE](../../../backlog.md#atlas-ritk-spatial-finite-inverse)
