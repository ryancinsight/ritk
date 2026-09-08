//! Authoritative DICOM file-set discovery shared by browsing and loading.

use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use ritk_dicom::{parse_bytes_with_budget, read_file_with_budget, DicomRsBackend};
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use ritk_dicom::ParseBudget;

use super::detection::is_likely_dicom_file;
use super::dicomdir_bytes::directory_record_offsets;

pub(super) fn is_dicomdir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("DICOMDIR"))
}

/// Resolve a directory or an explicitly selected index to its exact file set.
/// An existing index is authoritative: malformed or missing references fail.
pub(in crate::format::dicom) fn discover_files(path: &Path) -> Result<Vec<PathBuf>> {
    discover_files_with_budget(path, &ParseBudget::DEFAULT)
}

pub(in crate::format::dicom) fn discover_files_with_budget(
    path: &Path,
    budget: &ParseBudget,
) -> Result<Vec<PathBuf>> {
    if is_dicomdir(path) && !path.is_dir() {
        return read_dicomdir(path, budget);
    }
    let entries = std::fs::read_dir(path)
        .context("failed to read DICOM directory")?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let indexes: Vec<_> = entries.iter().filter(|entry| is_dicomdir(entry)).collect();
    match indexes.as_slice() {
        [] => {
            let mut paths: Vec<_> = entries
                .into_iter()
                .filter(|entry| entry.is_file() && is_likely_dicom_file(entry))
                .collect();
            paths.sort();
            Ok(paths)
        }
        [index] => read_dicomdir(index, budget),
        _ => bail!("multiple DICOMDIR indexes in one directory"),
    }
}

fn read_dicomdir(index: &Path, budget: &ParseBudget) -> Result<Vec<PathBuf>> {
    let root = index
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .context("failed to resolve DICOMDIR root")?;
    let bytes = read_file_with_budget(index, budget).context("failed to open DICOMDIR")?;
    let obj = parse_bytes_with_budget::<DicomRsBackend>(&bytes, budget)
        .context("failed to parse DICOMDIR")?;
    if obj.meta().transfer_syntax.trim_end_matches('\0').trim() != "1.2.840.10008.1.2.1" {
        bail!("DICOMDIR must use Explicit VR Little Endian transfer syntax");
    }
    let sequence = obj
        .element(Tag(0x0004, 0x1220))
        .context("DICOMDIR missing DirectoryRecordSequence (0004,1220)")?;
    let items = sequence
        .value()
        .items()
        .context("DICOMDIR DirectoryRecordSequence is not a sequence")?;
    let offsets = directory_record_offsets(&bytes)?;
    if offsets.len() != items.len() {
        bail!(
            "DICOMDIR record count mismatch: encoded {} items, parsed {} records",
            offsets.len(),
            items.len()
        );
    }
    let records = items
        .iter()
        .zip(offsets)
        .map(|(item, offset)| DirectoryRecord::from_item(item, offset))
        .collect::<Result<Vec<_>>>()?;
    let offset_to_index = validate_record_graph(&records)?;
    let first_root = required_u32(&obj, Tag(0x0004, 0x1200), "FirstRootDirectoryRecord")?;
    let last_root = required_u32(&obj, Tag(0x0004, 0x1202), "LastRootDirectoryRecord")?;
    validate_root_chain(&records, &offset_to_index, first_root, last_root)?;
    let reachable = reachable_records(&records, &offset_to_index, first_root)?;
    let mut paths = Vec::with_capacity(items.len());
    for (record, status) in records.iter().zip(reachable) {
        if !status.reachable || record.in_use != RECORD_IN_USE || !status.active_lineage {
            continue;
        }
        if !record.record_type.eq_ignore_ascii_case("IMAGE") {
            continue;
        }
        let reference = record
            .referenced_file_id
            .as_deref()
            .context("DICOMDIR image record missing ReferencedFileID")?;
        let candidate = resolve_reference(&root, reference)?;
        let resolved = candidate
            .canonicalize()
            .context("DICOMDIR referenced file is missing or inaccessible")?;
        if !resolved.starts_with(&root) {
            bail!("DICOMDIR referenced file escapes the file-set root");
        }
        if !resolved.is_file() {
            bail!("DICOMDIR reference is not a file");
        }
        verify_record_identity(record, &candidate, budget)?;
        paths.push(candidate);
    }
    if paths.is_empty() {
        bail!("DICOMDIR contained no image ReferencedFileID entries");
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

const RECORD_IN_USE: u16 = u16::MAX;

#[derive(Debug, Clone)]
struct DirectoryRecord {
    offset: u32,
    next: u32,
    lower: u32,
    in_use: u16,
    record_type: String,
    referenced_file_id: Option<String>,
    referenced_sop_class_uid: Option<String>,
    referenced_sop_instance_uid: Option<String>,
    referenced_transfer_syntax_uid: Option<String>,
}

impl DirectoryRecord {
    fn from_item(item: &dicom::object::InMemDicomObject, offset: u32) -> Result<Self> {
        let in_use = required_u16(item, Tag(0x0004, 0x1410), "RecordInUseFlag")?;
        if in_use != 0 && in_use != RECORD_IN_USE {
            bail!("DICOMDIR RecordInUseFlag has unsupported value {in_use:#06x}");
        }
        Ok(Self {
            offset,
            next: required_u32(item, Tag(0x0004, 0x1400), "OffsetOfTheNextDirectoryRecord")?,
            lower: required_u32(
                item,
                Tag(0x0004, 0x1420),
                "OffsetOfReferencedLowerLevelDirectoryEntity",
            )?,
            in_use,
            record_type: required_text(item, Tag(0x0004, 0x1430), "DirectoryRecordType")?,
            referenced_file_id: optional_text(item, Tag(0x0004, 0x1500), "ReferencedFileID")?,
            referenced_sop_class_uid: optional_text(
                item,
                Tag(0x0004, 0x1510),
                "ReferencedSOPClassUID",
            )?,
            referenced_sop_instance_uid: optional_text(
                item,
                Tag(0x0004, 0x1511),
                "ReferencedSOPInstanceUID",
            )?,
            referenced_transfer_syntax_uid: optional_text(
                item,
                Tag(0x0004, 0x1512),
                "ReferencedTransferSyntaxUID",
            )?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct RecordReachability {
    reachable: bool,
    active_lineage: bool,
}

fn required_u16<O>(object: &O, tag: Tag, name: &'static str) -> Result<u16>
where
    O: dicom::object::DicomObject,
{
    use dicom::object::DicomAttribute;

    object
        .attr(tag)
        .with_context(|| format!("DICOMDIR record missing {name}"))?
        .to_u16()
        .with_context(|| format!("invalid DICOMDIR {name}"))
}

fn required_u32<O>(object: &O, tag: Tag, name: &'static str) -> Result<u32>
where
    O: dicom::object::DicomObject,
{
    use dicom::object::DicomAttribute;

    object
        .attr(tag)
        .with_context(|| format!("DICOMDIR missing {name}"))?
        .to_u32()
        .with_context(|| format!("invalid DICOMDIR {name}"))
}

fn required_text<O>(object: &O, tag: Tag, name: &'static str) -> Result<String>
where
    O: dicom::object::DicomObject,
{
    use dicom::object::DicomAttribute;

    let attribute = object
        .attr(tag)
        .with_context(|| format!("DICOMDIR record missing {name}"))?;
    let value = attribute
        .to_str()
        .with_context(|| format!("invalid DICOMDIR {name}"))?;
    let value = normalize_text(&value);
    if value.is_empty() {
        bail!("DICOMDIR {name} is empty");
    }
    Ok(value)
}

fn optional_text<O>(object: &O, tag: Tag, name: &'static str) -> Result<Option<String>>
where
    O: dicom::object::DicomObject,
{
    use dicom::object::DicomAttribute;

    let Ok(attribute) = object.attr(tag) else {
        return Ok(None);
    };
    let value = attribute
        .to_str()
        .with_context(|| format!("invalid DICOMDIR {name}"))?;
    let value = normalize_text(&value);
    if value.is_empty() {
        bail!("DICOMDIR {name} is empty");
    }
    Ok(Some(value))
}

fn normalize_text(value: &str) -> String {
    value.trim_end_matches('\0').trim().to_owned()
}

fn resolve_reference(root: &Path, reference: &str) -> Result<PathBuf> {
    let mut relative = PathBuf::new();
    for component in reference.trim().split('\\') {
        // DICOM File IDs cannot contain host path syntax. Check both
        // separators and drive syntax independently of the current OS.
        let mut components = Path::new(component).components();
        if component.is_empty()
            || component.contains(['/', ':'])
            || !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
        {
            bail!("invalid DICOMDIR ReferencedFileID path component");
        }
        relative.push(component);
    }
    Ok(root.join(relative))
}

fn validate_record_graph(records: &[DirectoryRecord]) -> Result<HashMap<u32, usize>> {
    let mut offset_to_index = HashMap::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        if offset_to_index.insert(record.offset, index).is_some() {
            bail!(
                "DICOMDIR contains duplicate directory-record offset {}",
                record.offset
            );
        }
    }
    let mut incoming = vec![0_u8; records.len()];
    for record in records {
        for pointer in [record.next, record.lower] {
            if pointer == 0 {
                continue;
            }
            let index = offset_to_index
                .get(&pointer)
                .copied()
                .context("DICOMDIR link points outside DirectoryRecordSequence")?;
            incoming[index] = incoming[index]
                .checked_add(1)
                .context("DICOMDIR directory record incoming-link count overflow")?;
            if incoming[index] > 1 {
                bail!("DICOMDIR directory record has multiple incoming links");
            }
        }
    }

    let mut state = vec![0_u8; records.len()];
    for start in 0..records.len() {
        if state[start] != 0 {
            continue;
        }
        let mut stack = vec![(start, false)];
        while let Some((index, exiting)) = stack.pop() {
            if exiting {
                state[index] = 2;
                continue;
            }
            match state[index] {
                1 => bail!("DICOMDIR directory-record links contain a cycle"),
                2 => continue,
                _ => {}
            }
            state[index] = 1;
            stack.push((index, true));
            for pointer in [records[index].next, records[index].lower] {
                if pointer == 0 {
                    continue;
                }
                let child = *offset_to_index
                    .get(&pointer)
                    .context("DICOMDIR link points outside DirectoryRecordSequence")?;
                stack.push((child, false));
            }
        }
    }
    Ok(offset_to_index)
}

fn validate_root_chain(
    records: &[DirectoryRecord],
    offset_to_index: &HashMap<u32, usize>,
    first_root: u32,
    last_root: u32,
) -> Result<()> {
    if first_root == 0 || last_root == 0 {
        bail!("DICOMDIR root directory-record offsets must be nonzero");
    }
    let mut cursor = first_root;
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(cursor) {
            bail!("DICOMDIR root directory-record links contain a cycle");
        }
        let index = offset_to_index
            .get(&cursor)
            .copied()
            .context("DICOMDIR root offset is outside DirectoryRecordSequence")?;
        let next = records[index].next;
        if next == 0 {
            if cursor != last_root {
                bail!("DICOMDIR LastRootDirectoryRecord does not terminate the root chain");
            }
            break;
        }
        cursor = next;
    }
    Ok(())
}

fn reachable_records(
    records: &[DirectoryRecord],
    offset_to_index: &HashMap<u32, usize>,
    first_root: u32,
) -> Result<Vec<RecordReachability>> {
    let mut result = vec![
        RecordReachability {
            reachable: false,
            active_lineage: false,
        };
        records.len()
    ];
    let root = *offset_to_index
        .get(&first_root)
        .context("DICOMDIR root offset is outside DirectoryRecordSequence")?;
    let mut stack = vec![(root, true)];
    while let Some((index, parent_active)) = stack.pop() {
        if result[index].reachable {
            continue;
        }
        let record = &records[index];
        let active_lineage = parent_active && record.in_use == RECORD_IN_USE;
        result[index] = RecordReachability {
            reachable: true,
            active_lineage,
        };
        if record.lower != 0 {
            let child = *offset_to_index
                .get(&record.lower)
                .context("DICOMDIR lower-level link is outside DirectoryRecordSequence")?;
            stack.push((child, active_lineage));
        }
        if record.next != 0 {
            let sibling = *offset_to_index
                .get(&record.next)
                .context("DICOMDIR next-record link is outside DirectoryRecordSequence")?;
            stack.push((sibling, parent_active));
        }
    }
    Ok(result)
}

fn verify_record_identity(
    record: &DirectoryRecord,
    path: &Path,
    budget: &ParseBudget,
) -> Result<()> {
    let expected_class = record
        .referenced_sop_class_uid
        .as_deref()
        .context("DICOMDIR image record missing ReferencedSOPClassUID")?;
    let expected_instance = record
        .referenced_sop_instance_uid
        .as_deref()
        .context("DICOMDIR image record missing ReferencedSOPInstanceUID")?;
    let expected_transfer = record
        .referenced_transfer_syntax_uid
        .as_deref()
        .context("DICOMDIR image record missing ReferencedTransferSyntaxUID")?;
    let bytes =
        read_file_with_budget(path, budget).context("failed to read DICOMDIR referenced file")?;
    let object = parse_bytes_with_budget::<DicomRsBackend>(&bytes, budget)
        .context("failed to parse DICOMDIR referenced file")?;
    let actual_class = required_text(&object, Tag(0x0008, 0x0016), "SOPClassUID")?;
    let actual_instance = required_text(&object, Tag(0x0008, 0x0018), "SOPInstanceUID")?;
    let meta_class = normalize_text(object.meta().media_storage_sop_class_uid());
    let meta_instance = normalize_text(object.meta().media_storage_sop_instance_uid());
    let actual_transfer = normalize_text(object.meta().transfer_syntax());
    if expected_class != actual_class || actual_class != meta_class {
        bail!("DICOMDIR ReferencedSOPClassUID does not match the referenced file");
    }
    if expected_instance != actual_instance || actual_instance != meta_instance {
        bail!("DICOMDIR ReferencedSOPInstanceUID does not match the referenced file");
    }
    if expected_transfer != actual_transfer {
        bail!("DICOMDIR ReferencedTransferSyntaxUID does not match the referenced file");
    }
    Ok(())
}
