# ADR 0057: Viewer release scope

- Status: Proposed

## Context

`ritk-snap` is built as a native alternative to the RadiAnt DICOM Viewer, on
the Métis host. An audit of the native session at `30b8ba098` found that
navigation, multi-series panels and orthogonal MPR are reachable, while
measurement drawing, display controls, clinical overlays, cross-series linking,
PACS and 3D rendering exist only in the eframe shell, as unrouted library code,
or not at all. The board had no statement of what finishing means.

## Decision

Version 1.0 of the viewer means native reachability of the feature list
published at
<https://www.radiantviewer.com/dicom-viewer-manual/radiant_dicom_viewer_features.html>,
delivered through the four milestones in `backlog/milestones.md`. Each milestone
closes on runnable exit checks. Behavior is matched from that manual; RadiAnt's
name, artwork and text are not reproduced in the application.

Excluded from 1.0: encrypted ZIP archives, optical-disc import beyond reading a
mounted folder, the DCMTK send add-on, Windows 7/8 and 32-bit targets, and
translations beyond English.

Métis is the viewer's only shell. The eframe shell, its feature and its
egui, eframe and rfd dependencies are removed rather than maintained beside
it. Métis remains format-neutral: it supplies host capabilities
(dialogs, drop events, clipboard), while DICOM semantics stay in RITK.

## Alternatives

- Keep the eframe shell compiling beside Métis. Rejected: the user directed a
  complete migration on 2026-11-09, ADR 0026 retires the shell, and a second
  shell hides what the native session cannot do.
- Leave scope implicit in item titles. Rejected: nothing could establish that
  the viewer is finished, and version numbers named no goal.

## Consequences

The ladder orders work by what a reader needs first. Removing or weakening an
exit check requires a revision note here.

## Overturning evidence

A user decision narrowing or widening the feature list, or a consumer need for
a feature the manual does not list.
