//! Single-series scanning with explicit input identity.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use ritk_dicom::{
    parse_bytes_with_budget, read_file_with_budget, read_file_within_root_with_budget,
    DicomRsBackend,
};

use super::dicomdir::{discover_files_with_budget, is_dicomdir};
use super::parse::extract_dicom_metadata;
use super::types::{DicomReadBudget, DicomSeriesInfo, DicomSliceMetadata, SeriesFirstSeen};
use crate::format::dicom::identity::image_series_uid;
use crate::format::dicom::networking::scp::StoredInstance;
use crate::format::dicom::object_model::{DicomObjectModel, DicomObjectNode, DicomTag};

mod finalize;
mod geometry;
mod thresholds;

use finalize::finalize_scanned_series;

/// Scan a directory containing exactly one image series.
///
/// # Errors
/// Rejects malformed files, missing identity, and multiple series. An existing
/// DICOMDIR is authoritative; invalid references never trigger folder fallback.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<DicomSeriesInfo> {
    scan_dicom_directory_with_budget(path, &DicomReadBudget::DEFAULT)
}

/// Scan a directory using an explicit parser resource budget.
pub fn scan_dicom_directory_with_budget<P: AsRef<Path>>(
    path: P,
    budget: &DicomReadBudget,
) -> Result<DicomSeriesInfo> {
    let path = path.as_ref();
    if !path.is_dir() && !is_dicomdir(path) {
        bail!("DICOM input path is not a directory");
    }
    let parser_budget = budget.parser();
    let root = file_set_root(path)?;
    let discovery_path = if path.is_dir() { root.as_path() } else { path };
    let paths = discover_files_with_budget(discovery_path, &parser_budget)?;
    scan_files(&paths, None, budget, Some(&root))
}

/// Scan the exact members of one image series.
///
/// Paths may span directories. Duplicate paths are rejected rather than
/// duplicating slices. Non-image SOP classes do not contribute metadata.
/// Each retained slice owns the exact bytes validated during scanning, so
/// subsequent decoding does not reopen a potentially replaced file.
///
/// # Examples
/// ```no_run
/// # use std::path::PathBuf;
/// let series = ritk_io::scan_dicom_files(&[PathBuf::from("study/slice.dcm")])?;
/// assert_eq!(series.num_slices, 1);
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Errors
/// Rejects empty input, unreadable or malformed members, missing or invalid
/// SeriesInstanceUID, multiple series, and inconsistent image dimensions.
pub fn scan_dicom_files(paths: &[PathBuf]) -> Result<DicomSeriesInfo> {
    scan_dicom_files_with_budget(paths, &DicomReadBudget::DEFAULT)
}

/// Scan exact DICOM members using an explicit parser resource budget.
pub fn scan_dicom_files_with_budget(
    paths: &[PathBuf],
    budget: &DicomReadBudget,
) -> Result<DicomSeriesInfo> {
    scan_files(paths, None, budget, None)
}

/// Scan a directory, a selected DICOM instance, or an explicit DICOMDIR.
///
/// A selected instance determines the SeriesInstanceUID before collecting
/// matching members from its containing file set. A directory or DICOMDIR
/// containing multiple image series requires explicit member selection.
///
/// # Examples
/// ```no_run
/// let series = ritk_io::scan_dicom_path("study/selected.dcm")?;
/// let (image, metadata) = ritk_io::load_dicom_from_series(
///     series, &coeus_core::SequentialBackend,
/// )?;
/// assert_eq!(image.shape()[0], metadata.slices.len());
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Errors
/// Rejects malformed selected input, ambiguous series, or invalid DICOMDIR
/// references. A selected file cannot silently load a neighboring acquisition.
pub fn scan_dicom_path(path: impl AsRef<Path>) -> Result<DicomSeriesInfo> {
    scan_dicom_path_with_budget(path, &DicomReadBudget::DEFAULT)
}

/// Scan a selected DICOM path using an explicit parser resource budget.
pub fn scan_dicom_path_with_budget(
    path: impl AsRef<Path>,
    budget: &DicomReadBudget,
) -> Result<DicomSeriesInfo> {
    let path = path.as_ref();
    if path.is_dir() || is_dicomdir(path) {
        return scan_dicom_directory_with_budget(path, budget);
    }
    let parser_budget = budget.parser();
    let root = file_set_root(path)?;
    let name = path
        .file_name()
        .context("selected DICOM path has no final component")?;
    let selected_path = root.join(name);
    let bytes = read_file_within_root_with_budget(&selected_path, &root, &parser_budget)
        .context("failed to read selected DICOM instance")?;
    let object = parse_bytes_with_budget::<DicomRsBackend>(&bytes, &parser_budget)
        .context("failed to parse selected DICOM instance")?;
    let uid = image_series_uid(&object)?.context("selected DICOM instance is not image-bearing")?;
    let paths = discover_files_with_budget(&root, &parser_budget)?;
    let resolved = selected_path
        .canonicalize()
        .context("failed to resolve selected DICOM instance")?;
    scan_files(
        &paths,
        Some(SelectedInstance {
            resolved,
            uid,
            object,
            bytes,
        }),
        budget,
        Some(&root),
    )
}

fn file_set_root(path: &Path) -> Result<PathBuf> {
    let source = if path.is_dir() {
        path
    } else {
        path.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
    };
    source
        .canonicalize()
        .with_context(|| format!("failed to resolve DICOM file-set root {:?}", source))
}

struct SelectedInstance {
    resolved: PathBuf,
    uid: ArrayString<64>,
    object: DefaultDicomObject,
    bytes: Vec<u8>,
}

fn scan_files(
    paths: &[PathBuf],
    mut selected: Option<SelectedInstance>,
    budget: &DicomReadBudget,
    confined_root: Option<&Path>,
) -> Result<DicomSeriesInfo> {
    let source = paths
        .first()
        .context("no DICOM files provided for scanning")?;
    let mut accumulator = SeriesScan::default();
    let mut seen = std::collections::HashSet::with_capacity(paths.len());
    let selected_uid = selected.as_ref().map(|instance| instance.uid);
    let parser_budget = budget.parser();
    let mut ordered: Vec<_> = paths.iter().collect();
    ordered.sort();
    for path in ordered {
        let resolved = path
            .canonicalize()
            .context("DICOM member is missing or inaccessible")?;
        let is_selected = selected
            .as_ref()
            .is_some_and(|instance| instance.resolved == resolved);
        if !seen.insert(resolved) {
            bail!("duplicate DICOM file in selected member set");
        }
        let (object, bytes) = if is_selected {
            let instance = selected
                .take()
                .expect("invariant: selected instance matches this member");
            (instance.object, instance.bytes)
        } else {
            let bytes = match confined_root {
                Some(root) => read_file_within_root_with_budget(path, root, &parser_budget),
                None => read_file_with_budget(path, &parser_budget),
            }
            .context("failed to read DICOM member")?;
            let object = parse_bytes_with_budget::<DicomRsBackend>(&bytes, &parser_budget)
                .context("failed to parse DICOM member")?;
            (object, bytes)
        };
        accumulator.push(
            &object,
            path.clone(),
            bytes,
            selected_uid.as_ref().map(ArrayString::as_str),
            budget,
        )?;
    }
    if selected.is_some() {
        bail!("selected DICOM instance is not a member of the containing file set");
    }
    accumulator.finish(
        source
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    )
}

/// Scan SCP-received instances as one image series without writing files.
///
/// # Errors
/// Rejects empty, malformed, unidentified, or mixed-series inputs.
pub fn scan_dicom_instances(instances: &[StoredInstance]) -> Result<DicomSeriesInfo> {
    scan_dicom_instances_with_budget(instances, &DicomReadBudget::DEFAULT)
}

/// Scan SCP-received instances using an explicit parser resource budget.
pub fn scan_dicom_instances_with_budget(
    instances: &[StoredInstance],
    budget: &DicomReadBudget,
) -> Result<DicomSeriesInfo> {
    let mut accumulator = SeriesScan::default();
    let parser_budget = budget.parser();
    for instance in instances {
        let bytes = instance.make_part10_bytes();
        let object = parse_bytes_with_budget::<DicomRsBackend>(&bytes, &parser_budget)
            .context("failed to parse SCP DICOM instance")?;
        accumulator.push(
            &object,
            PathBuf::from(format!("scp://{}", instance.sop_instance_uid)),
            bytes,
            None,
            budget,
        )?;
    }
    accumulator.finish(PathBuf::from("scp://series"))
}

/// Scan named Part 10 payloads as one image series without writing files.
///
/// Names are diagnostic identities, never filesystem paths to open.
///
/// # Errors
/// Rejects empty, malformed, unidentified, or mixed-series inputs. A pathless
/// DICOMDIR cannot resolve filesystem references and is rejected.
pub fn scan_dicom_part10_bytes(files: &[(&str, &[u8])]) -> Result<DicomSeriesInfo> {
    scan_dicom_part10_bytes_with_budget(files, &DicomReadBudget::DEFAULT)
}

/// Scan named Part 10 payloads using an explicit parser resource budget.
pub fn scan_dicom_part10_bytes_with_budget(
    files: &[(&str, &[u8])],
    budget: &DicomReadBudget,
) -> Result<DicomSeriesInfo> {
    let mut accumulator = SeriesScan::default();
    let parser_budget = budget.parser();
    for (name, bytes) in files {
        if is_dicomdir(Path::new(name)) {
            bail!("pathless DICOMDIR references cannot be resolved");
        }
        let object = parse_bytes_with_budget::<DicomRsBackend>(bytes, &parser_budget)
            .context("failed to parse DICOM byte payload")?;
        accumulator.push(
            &object,
            PathBuf::from(format!("dropped://{name}")),
            bytes.to_vec(),
            None,
            budget,
        )?;
    }
    accumulator.finish(PathBuf::from("dropped://series"))
}

#[derive(Default)]
struct SeriesScan {
    slices: Vec<DicomSliceMetadata>,
    first: SeriesFirstSeen,
    uid: Option<ArrayString<64>>,
    dimensions: Option<(u32, u32)>,
    rejected_sop_classes: Vec<String>,
    retained_bytes: usize,
}

impl SeriesScan {
    fn push(
        &mut self,
        object: &DefaultDicomObject,
        path: PathBuf,
        bytes: Vec<u8>,
        selected_uid: Option<&str>,
        budget: &DicomReadBudget,
    ) -> Result<()> {
        let Some(uid) = image_series_uid(object)? else {
            let sop = object.element(Tag(0x0008, 0x0016))?.to_str()?.into_owned();
            self.rejected_sop_classes.push(sop);
            return Ok(());
        };
        if selected_uid.is_some_and(|selected| selected != uid.as_str()) {
            return Ok(());
        }
        if self.uid.as_ref().is_some_and(|previous| previous != &uid) {
            bail!("ambiguous DICOM input: multiple SeriesInstanceUID values require explicit selection");
        }
        self.uid = Some(uid);
        let (mut slice, dimensions) = extract_dicom_metadata(object, path, &mut self.first);
        if self
            .dimensions
            .is_some_and(|previous| previous != dimensions)
        {
            bail!("inconsistent image dimensions within selected DICOM series");
        }
        self.dimensions = Some(dimensions);
        let retained_bytes = self
            .retained_bytes
            .checked_add(bytes.len())
            .context("DICOM retained study byte total overflow")?;
        budget.checked_retained_bytes(retained_bytes)?;
        self.retained_bytes = retained_bytes;
        slice.part10_bytes = Some(bytes);
        self.slices.push(slice);
        Ok(())
    }

    fn finish(self, path: PathBuf) -> Result<DicomSeriesInfo> {
        if self.slices.is_empty() {
            bail!("no DICOM image instances: none are image-bearing SOP classes; rejected SOP class UIDs: [{}]", self.rejected_sop_classes.join(", "));
        }
        Ok(finalize_scanned_series(
            self.slices,
            self.first,
            path,
            |a, b| {
                a.sop_instance_uid
                    .cmp(&b.sop_instance_uid)
                    .then_with(|| a.path.cmp(&b.path))
            },
        ))
    }
}
/// Build a `DicomObjectModel` from the slice metadata for a series.
///
/// This constructs a lightweight object model populated with the key
/// per-instance tags (SOP Instance UID, Instance Number, Slice Location,
/// Image Position/Orientation, Pixel Spacing, Slice Thickness, SOP Class UID)
/// so downstream consumers can inspect series-level DICOM attributes without
/// re-parsing the original files.
fn build_series_object(path: &Path, slices: &[DicomSliceMetadata]) -> DicomObjectModel {
    let mut series_object = DicomObjectModel::with_source(path.to_path_buf());
    for slice in slices {
        if let Some(uid) = slice.sop_instance_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0018),
                "UI",
                uid.as_str().to_string(),
            ));
        }
        if let Some(instance_number) = slice.instance_number {
            series_object.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0020, 0x0013),
                "IS",
                instance_number,
            ));
        }
        if let Some(slice_location) = slice.slice_location {
            series_object.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0020, 0x1041),
                "DS",
                slice_location,
            ));
        }
        if let Some(position) = slice.image_position_patient {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0020, 0x0032),
                "DS",
                format!("{:.6}\\{:.6}\\{:.6}", position[0], position[1], position[2]),
            ));
        }
        if let Some(orientation) = slice.image_orientation_patient {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0020, 0x0037),
                "DS",
                format!(
                    "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
                    orientation[0],
                    orientation[1],
                    orientation[2],
                    orientation[3],
                    orientation[4],
                    orientation[5]
                ),
            ));
        }
        if let Some(pixel_spacing) = slice.pixel_spacing {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0028, 0x0030),
                "DS",
                format!("{:.6}\\{:.6}", pixel_spacing[0], pixel_spacing[1]),
            ));
        }
        if let Some(slice_thickness) = slice.slice_thickness {
            series_object.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0018, 0x0050),
                "DS",
                slice_thickness,
            ));
        }
        if let Some(sop_class_uid) = slice.sop_class_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0016),
                "UI",
                sop_class_uid.as_str().to_string(),
            ));
        }
    }
    series_object
}
