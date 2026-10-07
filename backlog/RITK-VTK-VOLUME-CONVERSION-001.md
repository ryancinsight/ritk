<a id="RITK-VTK-VOLUME-CONVERSION-001"></a>

## RITK-VTK-VOLUME-CONVERSION-001 — Preserve VTK image-volume semantics — blocked
- outcome: convert VTK image-data files through RITK while retaining scalar arrays and physical geometry.
- acceptance: supported scalar types, origin, spacing, direction, and array layout round-trip or return typed loss before output changes.
- scope: crates/ritk-vtk/, crates/ritk-io/, VTK image-data tests and manual
- next: After the image contract, implement the VTK image adapter and verify volume conversion semantics.
- basis: 2c277e2d5b1e2c3b60b2c8afb587ab72d68f690a
- status: blocked
- blocker: Prerequisite items must merge before this item is ready.
- needs: RITK-VTK-IMAGE-CONTRACT-001
- priority: correctness
