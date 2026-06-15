//! Deterministic DICOM metadata table model for the viewer sidebar.
//!
//! This module converts `ritk-io` DICOM metadata into a presentation-neutral
//! row list. The egui sidebar renders rows only; it does not know how to walk
//! slice metadata, preservation nodes, or raw preserved elements.

use ritk_io::{DicomObjectNode, DicomPreservedElement, DicomReadMetadata, DicomValue};

#[path = "metadata_table_slice.rs"]
mod slice_rows;

use slice_rows::push_first_slice_rows;

/// Scope of a metadata row in the loaded study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataScope {
    /// Series-level DICOM or derived geometry field.
    Series,
    /// First loaded slice field used as representative image-plane metadata.
    FirstSlice,
    /// Preserved object-model node.
    PreservedNode,
    /// Raw preserved element.
    PreservedRaw,
    /// Private scalar tag map entry.
    PrivateTag,
}

impl MetadataScope {
    /// Stable label for rendering.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Series => "Series",
            Self::FirstSlice => "Slice[0]",
            Self::PreservedNode => "Preserved",
            Self::PreservedRaw => "Raw",
            Self::PrivateTag => "Private",
        }
    }
}

use std::borrow::Cow;

/// One row in the DICOM tag inspector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataRow<'a> {
    /// Row scope.
    pub scope: MetadataScope,
    /// Canonical tag text or derived-field identifier.
    pub tag: Cow<'a, str>,
    /// DICOM keyword or derived field name.
    pub keyword: Cow<'a, str>,
    /// Value representation when known.
    pub vr: Cow<'a, str>,
    /// Stable textual value.
    pub value: Cow<'a, str>,
}

impl<'a> MetadataRow<'a> {
    fn series(
        tag: &'a str,
        keyword: &'a str,
        vr: &'a str,
        value: impl Into<Cow<'a, str>>,
    ) -> Self {
        Self::new(MetadataScope::Series, tag, keyword, vr, value)
    }

    fn first_slice(
        tag: &'a str,
        keyword: &'a str,
        vr: &'a str,
        value: impl Into<Cow<'a, str>>,
    ) -> Self {
        Self::new(MetadataScope::FirstSlice, tag, keyword, vr, value)
    }

    fn private(tag: impl Into<Cow<'a, str>>, value: impl Into<Cow<'a, str>>) -> Self {
        Self::new(MetadataScope::PrivateTag, tag, "PrivateTag", "UN", value)
    }

    fn new(
        scope: MetadataScope,
        tag: impl Into<Cow<'a, str>>,
        keyword: impl Into<Cow<'a, str>>,
        vr: impl Into<Cow<'a, str>>,
        value: impl Into<Cow<'a, str>>,
    ) -> Self {
        Self {
            scope,
            tag: tag.into(),
            keyword: keyword.into(),
            vr: vr.into(),
            value: value.into(),
        }
    }
}

/// Build a deterministic tag-inspector row list from loaded DICOM metadata.
pub fn build_metadata_rows<'a>(metadata: &'a DicomReadMetadata) -> Vec<MetadataRow<'a>> {
    let mut rows = Vec::new();
    push_series_rows(metadata, &mut rows);
    if let Some(slice) = metadata.slices.first() {
        push_first_slice_rows(slice, &mut rows);
    }
    push_private_rows(metadata, &mut rows);
    push_preserved_rows(metadata, &mut rows);
    rows
}

fn push_series_rows<'a>(metadata: &'a DicomReadMetadata, rows: &mut Vec<MetadataRow<'a>>) {
    push_opt(
        rows,
        "0020,000E",
        "SeriesInstanceUID",
        "UI",
        metadata.series_instance_uid.as_deref(),
    );
    push_opt(
        rows,
        "0020,000D",
        "StudyInstanceUID",
        "UI",
        metadata.study_instance_uid.as_deref(),
    );
    push_opt(
        rows,
        "0020,0052",
        "FrameOfReferenceUID",
        "UI",
        metadata.frame_of_reference_uid.as_deref(),
    );
    push_opt(
        rows,
        "0008,103E",
        "SeriesDescription",
        "LO",
        metadata.series_description.as_deref(),
    );
    push_opt(
        rows,
        "0008,0060",
        "Modality",
        "CS",
        metadata.modality.as_deref(),
    );
    push_opt(
        rows,
        "0010,0020",
        "PatientID",
        "LO",
        metadata.patient_id.as_deref(),
    );
    push_opt(
        rows,
        "0010,0010",
        "PatientName",
        "PN",
        metadata.patient_name.as_deref(),
    );
    push_opt(
        rows,
        "0008,0020",
        "StudyDate",
        "DA",
        metadata.study_date.as_deref(),
    );
    push_opt(
        rows,
        "0008,0021",
        "SeriesDate",
        "DA",
        metadata.series_date.as_deref(),
    );
    push_opt(
        rows,
        "0008,0031",
        "SeriesTime",
        "TM",
        metadata.series_time.as_deref(),
    );
    rows.push(MetadataRow::series(
        "derived:dimensions",
        "Dimensions",
        "",
        format_usize3(metadata.dimensions),
    ));
    rows.push(MetadataRow::series(
        "derived:spacing",
        "Spacing",
        "DS",
        format_float_slice(metadata.spacing),
    ));
    rows.push(MetadataRow::series(
        "derived:origin",
        "Origin",
        "DS",
        format_float_slice(metadata.origin),
    ));
    rows.push(MetadataRow::series(
        "derived:direction",
        "Direction",
        "DS",
        format_float_slice(metadata.direction),
    ));
    if let Some(value) = metadata.bits_allocated {
        rows.push(MetadataRow::series(
            "0028,0100",
            "BitsAllocated",
            "US",
            value.to_string(),
        ));
    }
    if let Some(value) = metadata.bits_stored {
        rows.push(MetadataRow::series(
            "0028,0101",
            "BitsStored",
            "US",
            value.to_string(),
        ));
    }
    if let Some(value) = metadata.high_bit {
        rows.push(MetadataRow::series(
            "0028,0102",
            "HighBit",
            "US",
            value.to_string(),
        ));
    }
    push_opt(
        rows,
        "0028,0004",
        "PhotometricInterpretation",
        "CS",
        metadata.photometric_interpretation.as_deref(),
    );
}

fn push_private_rows<'a>(metadata: &'a DicomReadMetadata, rows: &mut Vec<MetadataRow<'a>>) {
    let mut tags: Vec<_> = metadata.private_tags.iter().collect();
    tags.sort_by_key(|(tag, _)| tag.as_str());
    for (tag, value) in tags {
        rows.push(MetadataRow::private(tag.as_str(), value.as_str()));
    }
}

fn push_preserved_rows<'a>(metadata: &'a DicomReadMetadata, rows: &mut Vec<MetadataRow<'a>>) {
    for node in &metadata.preservation.object.nodes {
        rows.push(row_from_node(MetadataScope::PreservedNode, node));
    }
    for element in &metadata.preservation.preserved {
        rows.push(row_from_preserved(element));
    }
    if let Some(slice) = metadata.slices.first() {
        for node in &slice.preservation.object.nodes {
            rows.push(row_from_node(MetadataScope::PreservedNode, node));
        }
        for element in &slice.preservation.preserved {
            rows.push(row_from_preserved(element));
        }
    }
}

fn row_from_node<'a>(scope: MetadataScope, node: &'a DicomObjectNode) -> MetadataRow<'a> {
    MetadataRow::new(
        scope,
        node.tag.canonical(),
        if node.is_private() {
            "PrivateNode"
        } else {
            "PreservedNode"
        },
        node.vr.as_deref().unwrap_or("UN"),
        value_to_text(&node.value),
    )
}

fn row_from_preserved<'a>(element: &'a DicomPreservedElement) -> MetadataRow<'a> {
    MetadataRow::new(
        MetadataScope::PreservedRaw,
        element.tag.canonical(),
        "RawElement",
        element.vr.as_deref().unwrap_or("UN"),
        format!("{} bytes", element.bytes.len()),
    )
}

fn push_opt<'a>(
    rows: &mut Vec<MetadataRow<'a>>,
    tag: &'a str,
    keyword: &'a str,
    vr: &'a str,
    value: Option<&'a str>,
) {
    if let Some(value) = value {
        rows.push(MetadataRow::series(tag, keyword, vr, value));
    }
}

pub(super) fn push_slice_opt<'a>(
    rows: &mut Vec<MetadataRow<'a>>,
    tag: &'a str,
    keyword: &'a str,
    vr: &'a str,
    value: Option<&'a str>,
) {
    if let Some(value) = value {
        rows.push(MetadataRow::first_slice(tag, keyword, vr, value));
    }
}

fn value_to_text<'a>(value: &'a DicomValue) -> Cow<'a, str> {
    match value {
        DicomValue::Text(val) => Cow::Borrowed(val.as_str()),
        DicomValue::Bytes(bytes) => Cow::Owned(format!("{} bytes", bytes.len())),
        DicomValue::U16(val) => Cow::Owned(val.to_string()),
        DicomValue::I32(val) => Cow::Owned(val.to_string()),
        DicomValue::F64(val) => Cow::Owned(format!("{val:.6}")),
        DicomValue::Sequence(items) => Cow::Owned(format!("{} items", items.len())),
        DicomValue::Empty => Cow::Borrowed(""),
    }
}

fn format_usize3(values: [usize; 3]) -> String {
    format!("{} x {} x {}", values[0], values[1], values[2])
}

pub(super) fn format_float_slice<const N: usize>(values: [f64; N]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(N * 12);
    for (i, &v) in values.iter().enumerate() {
        if i > 0 {
            s.push_str(" x ");
        }
        let _ = write!(s, "{v:.6}");
    }
    s
}

#[cfg(test)]
#[path = "metadata_table_tests.rs"]
mod tests;
