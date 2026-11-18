<a id="RITK-SNAP-ANNOTATION-COPY-001"></a>
## RITK-SNAP-ANNOTATION-COPY-001 — Copy and duplicate image annotations — todo
- outcome: Selected or all annotations can be copied, pasted, duplicated to target images/series, and deleted together.
- acceptance: Assert exact source/target annotation sets, units and source locations after copy/paste/duplicate/delete, including invalid payload and cancellation cases.
- scope: crates/ritk-snap/src/app/annotation_edit*, crates/ritk-snap/src/session/
- next: Define the versioned annotation clipboard payload and its source/target geometry rules.
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- milestone: M-viewer-reading
- priority: feature
- reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
