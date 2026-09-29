# ADR 0026: Viewer presentation migration to Métis

Status: Accepted

Date: 2026-09-06

Driver: RITK-SNAP-METIS-001 established the migration decision.

Revision 2026-09-29: [PR #676](https://github.com/ryancinsight/ritk/pull/676) keeps event hit tests bound to the framebuffer visible when a native batch begins. If a maximize action changes panel ownership before later queued input is reduced, the route map sends that input to the original panel's retained state, even when that panel is hidden by the new layout; the regression checks that the maximized series stays unchanged and the hidden series advances. A current review of RadiAnt's [series-browsing guide](https://www.radiantviewer.com/dicom-viewer-manual/browse_series_and_images.html), [keyboard shortcuts](https://www.radiantviewer.com/dicom-viewer-manual/keyboard_shortcuts.html), and [multi-series manual](https://www.radiantviewer.com/dicom-viewer-manual/PDF/radiantmanual402.pdf) corrects the earlier assumption: its preview bar sits on the left, with patient and study details, active-series highlighting, and thumbnail image counts. RITK adopts the documented controls and arrangement while retaining DICOM parsing and medical semantics.

## Context

RITK needs a presentation host for native desktop and browser deployments while preserving one DICOM and viewer implementation. At the decision basis, the desktop shell used eframe and egui; the requested target was Métis. Tauri was not a RITK dependency. A framework comparison does not establish behavioral or performance parity, so the migration uses the existing RITK viewer as a source of requirements and independent DICOM fixtures as correctness oracles.

The target is a RadiAnt desktop clone. The native workspace keeps the menu and compact toolbar above a dark, independently controlled image-panel grid, with a vertical study-and-series preview bar on the left. Cards group study details and show decoded thumbnails, modality, image-count badges, and panel assignments. Users select several series through the F4 dialog, assign a series by click or drag, or Ctrl-click to open another panel. Panel layouts cover 1×1 through 5×4. RadiAnt documents [series browsing](https://www.radiantviewer.com/dicom-viewer-manual/browse_series_and_images.html) and [multiple-series panels](https://www.radiantviewer.com/dicom-viewer-manual/PDF/radiantmanual402.pdf), including direct assignment, multi-selection, and per-panel lifecycle.

## Decision

RITK owns DICOM parsing, transfer-syntax and pixel interpretation, acquisition identity and selection, physical geometry, medical display transforms, viewer state, navigation, measurements and analysis. All hosts receive RITK-produced presentation values and return bounded host events; Métis does not inspect DICOM tags or maintain a parallel medical model.

Métis owns native-window and browser-canvas presentation, input delivery, lifecycle and application controls. Moirai supplies shared host and execution services. Shared viewer behavior remains in RITK so native and browser hosts apply the same domain transitions.

Migrate shell behavior in complete, tested increments. Keep any compatibility shell only while required workflows lack Métis coverage; cut it over only when the acceptance criteria below pass. A failure to open, decode, select or present a study remains visible to the user and does not replace a valid current study with stale or partial state.

## Ownership and trust boundaries

The protected assets are local study bytes, patient identity, acquisition selection, decoded pixels, geometry and current viewer state. DICOM files, directory indexes, browser file objects and persisted selections are untrusted inputs. Validate identity and file-set membership in RITK before decoding; bound encoded bytes, parser work, decoded storage and pending loads. Root-confined native opens use the validated handle for subsequent reads. A cancellation, failed replacement or superseded asynchronous load cannot publish stale state.

Native filesystem access and browser file access are different capabilities. A browser receives only user-selected file bytes and cannot inherit native path authority. Host events carry presentation coordinates and user actions, not DICOM metadata. Métis receives the rendered frames and format-neutral controls; RITK retains clinical data and policy.

Screenshots and diagnostics must not expose private studies or identifiers. The manual demonstration uses the public MRI-DIR porcine-head phantom under its [CC BY 4.0 source](https://doi.org/10.7937/K9/TCIA.2018.3f08iejt). Native study labels may include Patient Name when present; they never show Patient ID in this workflow.

## Presentation contract

The native viewer presents RITK image panels in a dark workspace, with File, View, Tools and Window menus, a compact toolbar, a left study-and-series preview bar, and a status bar. The rail highlights assigned series, groups study details, and places each series image count on its thumbnail. Selecting a series loads it into the active panel. Left/Right and horizontal-wheel input switch series in the active panel. Split screen opens a 5-column by 4-row picker covering every 1×1 through 5×4 grid. The F4 dialog filters the study catalog and opens several selected series together; Ctrl-click opens another panel, and drag/drop assigns a series to a selected panel. Clicking an assigned card activates its panel. Each panel retains its own volume, navigation and display state, including separate panels assigned the same series. Maximize, restore, close, close-all, and Tab navigation preserve panel identities and remaining state. A failed replacement preserves that panel's prior volume. Chrome input is consumed by controls and never becomes an image-pane gesture. The pane-only capture remains a separate pixel oracle.

The browser surface follows the same RITK state and frame contracts using browser-appropriate controls. Native window chrome is not evidence of browser-window behavior; each host requires its own input and visual tests. The Métis shared demonstration contract remains in [V09](../../../metis/docs/VERIFICATION.md#V09).

## Acceptance

- Load the selected public or local study through the host's real input path; verify series identity, dimensions, slice navigation and rendered values in RITK.
- Keep all discovered series in the scrollable left preview bar, show study details and per-thumbnail image counts, switch the active panel by selecting a card or using Left/Right/horizontal wheel, select every grid from 1×1 through 5×4, open multiple selected series through F4, assign distinct series by click, Ctrl-click, and drag/drop, and preserve independent panel state through maximize, restore, close, close-all, and failed replacement.
- Verify menus and grouped toolbar actions, preview-card selection and scrolling, pane interaction, resize, failure recovery and orderly host close with value-semantic native tests and runtime checks.
- Verify browser file-byte loading, presentation and input independently; browser tests do not imply native filesystem access.
- Capture the running application window for user-facing evidence. Keep pixel-only images and full-window captures distinct, record source and image provenance, and inspect the rendered controls, series previews and anatomical planes.
- Keep the dependency direction host-to-RITK. A DICOM parser, series model or patient metadata layer in Métis violates this decision.

## Rejected alternatives

Keeping egui as the permanent shell does not meet the requested Métis target. Removing the current shell before Métis covers required workflows loses working behavior. Moving DICOM parsing or clinical state into Métis duplicates the format owner and crosses the host boundary. Maintaining a separate viewer implementation for each host creates divergent selection, geometry and display behavior.

## Consequences

RITK tests define viewer behavior independently of the shell. Métis and Moirai gaps are fixed in their owning repositories; RITK adapts to those first-party contracts without downstream format implementations. Framework agreement is differential evidence only; independent known-value DICOM cases and inspected output establish correctness. Performance claims require matched measurements.

The DICOMDIR directory-record structure follows [DICOM PS3.3 F.3.2.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.3.2.2.html). Current delivery state and remaining migration work live in the [RITK backlog](../../backlog.md) and the [viewer workflow manual](../manual/dicom-workflow.md), not in this decision record.
