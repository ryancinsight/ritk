# ADR 0026: Viewer presentation migration to Métis

Status: Accepted

Date: 2026-09-06

Driver: RITK-SNAP-METIS-001 established the migration decision.

Revision 2026-09-28: [PR #676](https://github.com/ryancinsight/ritk/pull/676) implements the RadiAnt-clone desktop workspace: organized menus and toolbar, a bottom series preview bar, draggable series assignment, and a selectable 1×1 through 5×4 grid picker. Each pane owns its loaded volume and viewer state; three-plane MPR remains available. Full-window public MRI and MRI/CT captures document the application controls and rendered images. RITK retains DICOM parsing and medical semantics.

## Context

RITK needs a presentation host for native desktop and browser deployments while preserving one DICOM and viewer implementation. At the decision basis, the desktop shell used eframe and egui; the requested target was Métis. Tauri was not a RITK dependency. A framework comparison does not establish behavioral or performance parity, so the migration uses the existing RITK viewer as a source of requirements and independent DICOM fixtures as correctness oracles.

The target is a RadiAnt desktop clone. This increment implements its documented core viewing workflow: organized menus and toolbar, a persistent bottom series preview bar, a selectable 1×1 through 5×4 panel grid, and direct series assignment. The viewer renders its own controls and keeps DICOM parsing and medical semantics in RITK. RadiAnt documents [series browsing](https://www.radiantviewer.com/dicom-viewer-manual/browse_series_and_images.html) and [multiple-series panels](https://www.radiantviewer.com/dicom-viewer-manual/view_multiple_series.html); the latter assigns preview-bar series to separate panels.

## Decision

RITK owns DICOM parsing, transfer-syntax and pixel interpretation, acquisition identity and selection, physical geometry, medical display transforms, viewer state, navigation, measurements and analysis. All hosts receive RITK-produced presentation values and return bounded host events; Métis does not inspect DICOM tags or maintain a parallel medical model.

Métis owns native-window and browser-canvas presentation, input delivery, lifecycle and application controls. Moirai supplies shared host and execution services. Shared viewer behavior remains in RITK so native and browser hosts apply the same domain transitions.

Migrate shell behavior in complete, tested increments. Keep any compatibility shell only while required workflows lack Métis coverage; cut it over only when the acceptance criteria below pass. A failure to open, decode, select or present a study remains visible to the user and does not replace a valid current study with stale or partial state.

## Ownership and trust boundaries

The protected assets are local study bytes, patient identity, acquisition selection, decoded pixels, geometry and current viewer state. DICOM files, directory indexes, browser file objects and persisted selections are untrusted inputs. Validate identity and file-set membership in RITK before decoding; bound encoded bytes, parser work, decoded storage and pending loads. Root-confined native opens use the validated handle for subsequent reads. A cancellation, failed replacement or superseded asynchronous load cannot publish stale state.

Native filesystem access and browser file access are different capabilities. A browser receives only user-selected file bytes and cannot inherit native path authority. Host events carry presentation coordinates and user actions, not DICOM metadata. Métis receives the rendered frames and format-neutral controls; RITK retains clinical data and policy.

Screenshots and diagnostics must not expose private studies or identifiers. The native toolbar and series preview show no patient name or identifier. The manual demonstration uses the public MRI-DIR porcine-head phantom under its [CC BY 4.0 source](https://doi.org/10.7937/K9/TCIA.2018.3f08iejt).

## Presentation contract

The native viewer presents RITK image panels in a dark workspace, with File, View, Tools and Window menus, grouped study/navigation/measurement/display tools, a persistent horizontal series preview bar above the status bar, decoded-series thumbnails, and a status bar. Selecting a series loads it into the active MPR workspace. Split screen opens a 5-column by 4-row picker covering every 1×1 through 5×4 grid. Users drag a series card into a pane or click a card to load it into the active pane. Clicking an assigned card activates its pane. Every pane retains its own volume, navigation and display state, including separate panes assigned the same series. A failed replacement preserves that pane's prior volume. Chrome input is consumed by controls and never becomes an image-pane gesture. The pane-only capture remains a separate pixel oracle.

The browser surface follows the same RITK state and frame contracts using browser-appropriate controls. Native window chrome is not evidence of browser-window behavior; each host requires its own input and visual tests. The Métis shared demonstration contract remains in [V09](../../../metis/docs/VERIFICATION.md#V09).

## Acceptance

- Load the selected public or local study through the host's real input path; verify series identity, dimensions, slice navigation and rendered values in RITK.
- Keep all discovered series in the scrollable bottom preview bar, switch the active MPR series by selecting a card, select every grid from 1×1 through 5×4, assign distinct series by click and drag/drop, preserve independent panel state and contents after a failed replacement, and verify duplicate-series assignment.
- Verify menus and grouped toolbar actions, preview-card selection and scrolling, pane interaction, resize, failure recovery and orderly host close with value-semantic native tests and runtime checks.
- Verify browser file-byte loading, presentation and input independently; browser tests do not imply native filesystem access.
- Capture the running application window for user-facing evidence. Keep pixel-only images and full-window captures distinct, record source and image provenance, and inspect the rendered controls, series previews and anatomical planes.
- Keep the dependency direction host-to-RITK. A DICOM parser, series model or patient metadata layer in Métis violates this decision.

## Rejected alternatives

Keeping egui as the permanent shell does not meet the requested Métis target. Removing the current shell before Métis covers required workflows loses working behavior. Moving DICOM parsing or clinical state into Métis duplicates the format owner and crosses the host boundary. Maintaining a separate viewer implementation for each host creates divergent selection, geometry and display behavior.

## Consequences

RITK tests define viewer behavior independently of the shell. Métis and Moirai gaps are fixed in their owning repositories; RITK adapts to those first-party contracts without downstream format implementations. Framework agreement is differential evidence only; independent known-value DICOM cases and inspected output establish correctness. Performance claims require matched measurements.

The DICOMDIR directory-record structure follows [DICOM PS3.3 F.3.2.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.3.2.2.html). Current delivery state and remaining migration work live in the [RITK backlog](../../backlog.md) and the [viewer workflow manual](../manual/dicom-workflow.md), not in this decision record.
