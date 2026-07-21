use crate::render::colormap::Colormap;
use crate::tools::kind::ToolKind;
use crate::ui::anatomical_label_for_axis;
use crate::{LoadedVolume, ViewerState};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub(crate) const CLINICAL_DISTRIBUTION_DIR: &str = "clinical_distribution";
pub(crate) const REPORT_FILE_NAME: &str = "report.md";
pub(crate) const MEDIA_DIR_NAME: &str = "media";
pub(crate) const CURRENT_SLICE_FILE_NAME: &str = "current_slice.png";
pub(crate) const MPR_DIR_NAME: &str = "mpr";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClinicalDistributionExportSummary {
    pub root: PathBuf,
    pub report_path: PathBuf,
    pub current_slice_path: PathBuf,
    pub mpr_root: PathBuf,
    pub current_slice_written: bool,
    pub mpr_written: usize,
    pub mpr_failed: usize,
}

/// DICOM metadata presence flags for clinical distribution reports.
///
/// Replaces 7 individual `bool` fields in `ClinicalDistributionSummary`,
/// eliminating boolean blindness at construction and call sites.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DcmPresenceFlags {
    pub patient_name: bool,
    pub patient_id: bool,
    pub study_date: bool,
    pub series_description: bool,
    pub source: bool,
    pub segmentation: bool,
    pub rt_struct: bool,
    pub rt_dose: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ClinicalDistributionSummary<'a> {
    pub modality: Option<&'a str>,
    pub plane_label: &'a str,
    pub axis: usize,
    pub slice_index: usize,
    pub shape: [usize; 3],
    pub spacing: [f64; 3],
    pub origin: [f64; 3],
    pub direction: [f64; 9],
    pub window_center: Option<f64>,
    pub window_width: Option<f64>,
    pub colormap_label: &'a str,
    pub active_tool_label: &'a str,
    pub annotation_count: usize,
    pub presence: DcmPresenceFlags,
}

pub(crate) fn distribution_root(base: &Path) -> PathBuf {
    base.join(CLINICAL_DISTRIBUTION_DIR)
}

pub(crate) fn report_path(root: &Path) -> PathBuf {
    root.join(REPORT_FILE_NAME)
}

pub(crate) fn media_root(root: &Path) -> PathBuf {
    root.join(MEDIA_DIR_NAME)
}

pub(crate) fn current_slice_path(root: &Path) -> PathBuf {
    media_root(root).join(CURRENT_SLICE_FILE_NAME)
}

pub(crate) fn mpr_root(root: &Path) -> PathBuf {
    media_root(root).join(MPR_DIR_NAME)
}

pub(crate) fn build_clinical_distribution_report(
    summary: &ClinicalDistributionSummary<'_>,
) -> String {
    let mut report = String::new();

    writeln!(&mut report, "# Clinical Distribution Report").expect("infallible write");
    writeln!(&mut report).expect("infallible write");
    writeln!(
        &mut report,
        "This package is anonymized. Direct identifiers are redacted in the report body."
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "Rendered media is stored under `media/` relative to this folder."
    )
    .expect("infallible: validated precondition");
    writeln!(&mut report).expect("infallible write");

    writeln!(&mut report, "## Study summary").expect("infallible write");
    writeln!(
        &mut report,
        "- Modality: {}",
        summary.modality.unwrap_or("—")
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Current plane: {} (axis {})",
        summary.plane_label, summary.axis
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Current slice index: {}",
        summary.slice_index
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Volume shape [depth, rows, cols]: {} × {} × {}",
        summary.shape[0], summary.shape[1], summary.shape[2]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Spacing [dz, dy, dx] mm: {:.4} × {:.4} × {:.4}",
        summary.spacing[0], summary.spacing[1], summary.spacing[2]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Origin [z, y, x] mm: {:.4} × {:.4} × {:.4}",
        summary.origin[0], summary.origin[1], summary.origin[2]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Direction matrix row 0: [{:.4}, {:.4}, {:.4}]",
        summary.direction[0], summary.direction[1], summary.direction[2]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Direction matrix row 1: [{:.4}, {:.4}, {:.4}]",
        summary.direction[3], summary.direction[4], summary.direction[5]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Direction matrix row 2: [{:.4}, {:.4}, {:.4}]",
        summary.direction[6], summary.direction[7], summary.direction[8]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Window centre: {}",
        summary
            .window_center
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "—".to_owned())
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Window width: {}",
        summary
            .window_width
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "—".to_owned())
    )
    .expect("infallible: validated precondition");
    writeln!(&mut report, "- Colormap: {}", summary.colormap_label).expect("infallible: validated precondition");
    writeln!(&mut report, "- Active tool: {}", summary.active_tool_label).expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- Annotation count: {}",
        summary.annotation_count
    )
    .expect("infallible: validated precondition");
    writeln!(&mut report).expect("infallible write");

    writeln!(&mut report, "## Protected identifiers").expect("infallible write");
    write_redacted_line(&mut report, "Patient name", summary.presence.patient_name);
    write_redacted_line(&mut report, "Patient ID", summary.presence.patient_id);
    write_redacted_line(&mut report, "Study date", summary.presence.study_date);
    write_redacted_line(
        &mut report,
        "Series description",
        summary.presence.series_description,
    );
    write_redacted_line(&mut report, "Source path", summary.presence.source);
    writeln!(&mut report).expect("infallible write");

    writeln!(&mut report, "## Associated overlays").expect("infallible write");
    writeln!(
        &mut report,
        "- Segmentation: {}",
        if summary.presence.segmentation {
            "present"
        } else {
            "absent"
        }
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- RT-STRUCT overlay: {}",
        if summary.presence.rt_struct {
            "present"
        } else {
            "absent"
        }
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- RT-DOSE overlay: {}",
        if summary.presence.rt_dose {
            "present"
        } else {
            "absent"
        }
    )
    .expect("infallible: validated precondition");
    writeln!(&mut report).expect("infallible write");

    writeln!(&mut report, "## Media layout").expect("infallible write");
    writeln!(&mut report, "- `report.md`").expect("infallible write");
    writeln!(&mut report, "- `media/{CURRENT_SLICE_FILE_NAME}`").expect("infallible write");
    writeln!(
        &mut report,
        "- `media/{}/*.png` ({} files)",
        plane_folder_name(summary.axis),
        summary.shape[summary.axis.min(2)]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- `media/coronal/*.png` ({} files)",
        summary.shape[1]
    )
    .expect("infallible: validated precondition");
    writeln!(
        &mut report,
        "- `media/sagittal/*.png` ({} files)",
        summary.shape[2]
    )
    .expect("infallible: validated precondition");

    report
}

pub(crate) fn summary_from_loaded_volume<'a>(
    volume: &'a LoadedVolume,
    viewer_state: &'a ViewerState,
    axis: usize,
    colormap: Colormap,
    active_tool: ToolKind,
    annotation_count: usize,
    segmentation_present: bool,
    rt_struct_present: bool,
    rt_dose_present: bool,
) -> ClinicalDistributionSummary<'a> {
    ClinicalDistributionSummary {
        modality: volume.modality.as_deref(),
        plane_label: anatomical_label_for_axis(Some(volume), axis),
        axis,
        slice_index: viewer_state.slice_index,
        shape: volume.shape,
        spacing: volume.spacing,
        origin: volume.origin,
        direction: volume.direction,
        window_center: viewer_state.window_center.map(f64::from),
        window_width: viewer_state.window_width.map(f64::from),
        colormap_label: colormap.label(),
        active_tool_label: active_tool.label(),
        annotation_count,
        presence: DcmPresenceFlags {
            patient_name: volume.patient_name.is_some(),
            patient_id: volume.patient_id.is_some(),
            study_date: volume.study_date.is_some(),
            series_description: volume.series_description.is_some(),
            source: volume.source.is_some(),
            segmentation: segmentation_present,
            rt_struct: rt_struct_present,
            rt_dose: rt_dose_present,
        },
    }
}

fn write_redacted_line(report: &mut String, label: &str, present: bool) {
    writeln!(
        report,
        "- {}: {}",
        label,
        if present { "[redacted]" } else { "—" }
    )
    .expect("infallible: validated precondition");
}

fn plane_folder_name(axis: usize) -> &'static str {
    match axis {
        0 => "axial",
        1 => "coronal",
        _ => "sagittal",
    }
}
