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
    pub patient_name_present: bool,
    pub patient_id_present: bool,
    pub study_date_present: bool,
    pub series_description_present: bool,
    pub source_present: bool,
    pub segmentation_present: bool,
    pub rt_struct_present: bool,
    pub rt_dose_present: bool,
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

    writeln!(&mut report, "# Clinical Distribution Report").unwrap();
    writeln!(&mut report).unwrap();
    writeln!(
        &mut report,
        "This package is anonymized. Direct identifiers are redacted in the report body."
    )
    .unwrap();
    writeln!(
        &mut report,
        "Rendered media is stored under `media/` relative to this folder."
    )
    .unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Study summary").unwrap();
    writeln!(
        &mut report,
        "- Modality: {}",
        summary.modality.unwrap_or("—")
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Current plane: {} (axis {})",
        summary.plane_label, summary.axis
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Current slice index: {}",
        summary.slice_index
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Volume shape [depth, rows, cols]: {} × {} × {}",
        summary.shape[0], summary.shape[1], summary.shape[2]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Spacing [dz, dy, dx] mm: {:.4} × {:.4} × {:.4}",
        summary.spacing[0], summary.spacing[1], summary.spacing[2]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Origin [z, y, x] mm: {:.4} × {:.4} × {:.4}",
        summary.origin[0], summary.origin[1], summary.origin[2]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Direction matrix row 0: [{:.4}, {:.4}, {:.4}]",
        summary.direction[0], summary.direction[1], summary.direction[2]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Direction matrix row 1: [{:.4}, {:.4}, {:.4}]",
        summary.direction[3], summary.direction[4], summary.direction[5]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Direction matrix row 2: [{:.4}, {:.4}, {:.4}]",
        summary.direction[6], summary.direction[7], summary.direction[8]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Window centre: {}",
        summary
            .window_center
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "—".to_owned())
    )
    .unwrap();
    writeln!(
        &mut report,
        "- Window width: {}",
        summary
            .window_width
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "—".to_owned())
    )
    .unwrap();
    writeln!(&mut report, "- Colormap: {}", summary.colormap_label).unwrap();
    writeln!(&mut report, "- Active tool: {}", summary.active_tool_label).unwrap();
    writeln!(
        &mut report,
        "- Annotation count: {}",
        summary.annotation_count
    )
    .unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Protected identifiers").unwrap();
    write_redacted_line(&mut report, "Patient name", summary.patient_name_present);
    write_redacted_line(&mut report, "Patient ID", summary.patient_id_present);
    write_redacted_line(&mut report, "Study date", summary.study_date_present);
    write_redacted_line(
        &mut report,
        "Series description",
        summary.series_description_present,
    );
    write_redacted_line(&mut report, "Source path", summary.source_present);
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Associated overlays").unwrap();
    writeln!(
        &mut report,
        "- Segmentation: {}",
        if summary.segmentation_present {
            "present"
        } else {
            "absent"
        }
    )
    .unwrap();
    writeln!(
        &mut report,
        "- RT-STRUCT overlay: {}",
        if summary.rt_struct_present {
            "present"
        } else {
            "absent"
        }
    )
    .unwrap();
    writeln!(
        &mut report,
        "- RT-DOSE overlay: {}",
        if summary.rt_dose_present {
            "present"
        } else {
            "absent"
        }
    )
    .unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Media layout").unwrap();
    writeln!(&mut report, "- `report.md`").unwrap();
    writeln!(&mut report, "- `media/{CURRENT_SLICE_FILE_NAME}`").unwrap();
    writeln!(
        &mut report,
        "- `media/{}/*.png` ({} files)",
        plane_folder_name(summary.axis),
        summary.shape[summary.axis.min(2)]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- `media/coronal/*.png` ({} files)",
        summary.shape[1]
    )
    .unwrap();
    writeln!(
        &mut report,
        "- `media/sagittal/*.png` ({} files)",
        summary.shape[2]
    )
    .unwrap();

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
        patient_name_present: volume.patient_name.is_some(),
        patient_id_present: volume.patient_id.is_some(),
        study_date_present: volume.study_date.is_some(),
        series_description_present: volume.series_description.is_some(),
        source_present: volume.source.is_some(),
        segmentation_present,
        rt_struct_present,
        rt_dose_present,
    }
}

fn write_redacted_line(report: &mut String, label: &str, present: bool) {
    writeln!(
        report,
        "- {}: {}",
        label,
        if present { "[redacted]" } else { "—" }
    )
    .unwrap();
}

fn plane_folder_name(axis: usize) -> &'static str {
    match axis {
        0 => "axial",
        1 => "coronal",
        _ => "sagittal",
    }
}
