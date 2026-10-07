<a id="RITK-SNAP-OBLIQUE-APP-ADAPTER-001"></a>

## RITK-SNAP-OBLIQUE-APP-ADAPTER-001 — Route oblique viewer actions — todo
- outcome: apply pointer and navigation actions to the RITK oblique view model.
- acceptance: production oblique viewport mapping consumes the checked shared geometry; clicks link the correct voxel, wheel translates the plane, and orientation keys update its basis; invalid actions reject.
- scope: `crates/ritk-snap/src/app/{action_adapter.rs,pointer_ops.rs,screen_image_geometry.rs,oblique_viewport.rs}` and oblique tests
- next: add the adapter with the smallest complete happy-path test.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981
- status: todo
- needs: none
- priority: feature
