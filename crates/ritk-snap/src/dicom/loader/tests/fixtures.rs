//! Deterministic, non-patient Part 10 studies shared by loader tests and captures.

use std::path::Path;

use anyhow::{Context, Result};
use ritk_io::{DicomObjectModel, DicomObjectNode, DicomTag};

/// Study shape in depth, row, column order.
pub(crate) const SHAPE: [usize; 3] = [3, 2, 4];
/// Physical voxel pitch in depth, row, column order, in millimetres.
pub(crate) const SPACING: [f64; 3] = [2.0, 1.5, 0.5];
/// First voxel's LPS position, in millimetres.
pub(crate) const ORIGIN: [f64; 3] = [10.0, 20.0, 30.0];
/// Row-major direction matrix whose columns correspond to depth, row, column.
pub(crate) const DIRECTION: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
/// Fixed synthetic series identity; this contains no patient information.
pub(crate) const SERIES_UID: &str = "2.25.20260905001";
/// Stored samples, in spatial order, before slope 2 and intercept -20.
pub(crate) const SAMPLES: [u8; 24] = [
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    210, 220, 230,
];

/// Two-frame scalar multiframe fixture geometry in `[frames, rows, cols]`
/// order. The frame payloads are intentionally distinct so a frame-zero
/// fallback cannot satisfy the viewer assertions.
#[cfg(test)]
pub(crate) const MULTIFRAME_SHAPE: [usize; 3] = [2, 2, 2];
#[cfg(test)]
pub(crate) const MULTIFRAME_RAW: [u8; 8] = [1, 2, 3, 4, 11, 12, 13, 14];
#[cfg(test)]
pub(crate) const MULTIFRAME_SERIES_UID: &str = "2.25.20260905003";

/// Two-frame RGB multi-frame fixture geometry in `[frames, rows, cols]` order.
pub(crate) const COLOR_MULTIFRAME_SHAPE: [usize; 3] = [2, 2, 2];
/// Red, green, blue, white followed by cyan, magenta, yellow, and neutral.
pub(crate) const COLOR_MULTIFRAME_RAW: [u8; 24] = [
    255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 64, 64,
    64,
];

/// Stored signed samples used to exercise modality rescale and grayscale
/// presentation. The loader applies slope `2` and intercept `-10`.
pub(crate) const PRESENTATION_STORED: [i16; 4] = [-10, 0, 10, 20];

/// Write a one-frame signed monochrome Part 10 object with optional VOI
/// function metadata and return its name and exact bytes.
pub(crate) fn write_grayscale_presentation(
    root: &Path,
    photometric: &str,
    voi_function: Option<&str>,
) -> Result<(String, Vec<u8>)> {
    std::fs::create_dir_all(root).context("create grayscale presentation directory")?;
    let mut model = DicomObjectModel::new();
    for (group, element, vr, value) in [
        (0x0008, 0x0016, "UI", "1.2.840.10008.5.1.4.1.1.7.5"),
        (0x0008, 0x0018, "UI", "2.25.20260905005.1"),
        (0x0008, 0x0060, "CS", "CT"),
        (0x0008, 0x0064, "CS", "WSD"),
        (0x0020, 0x000D, "UI", "2.25.20260905006"),
        (0x0020, 0x000E, "UI", "2.25.20260905005"),
        (0x0020, 0x0032, "DS", "0\\0\\0"),
        (0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0"),
        (0x0028, 0x0004, "CS", photometric),
        (0x0028, 0x0030, "DS", "1\\1"),
        (0x0018, 0x0050, "DS", "1"),
        (0x0028, 0x1052, "DS", "-10"),
        (0x0028, 0x1053, "DS", "2"),
        (0x0028, 0x1050, "DS", "0"),
        (0x0028, 0x1051, "DS", "40"),
    ] {
        model.insert(DicomObjectNode::text(
            DicomTag::new(group, element),
            vr,
            value,
        ));
    }
    if let Some(function) = voi_function {
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x1056),
            "CS",
            function,
        ));
    }
    for (element, value) in [
        (0x0002, 1_u16),
        (0x0010, 1),
        (0x0011, 4),
        (0x0100, 16),
        (0x0101, 16),
        (0x0102, 15),
        (0x0103, 1),
    ] {
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, element),
            "US",
            value,
        ));
    }
    let pixels = PRESENTATION_STORED
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    model.insert(DicomObjectNode::bytes(
        DicomTag::new(0x7FE0, 0x0010),
        "OW",
        pixels,
    ));

    let filename = "grayscale-presentation.dcm".to_owned();
    let path = root.join(&filename);
    ritk_io::write_dicom_object(&model, &path)
        .context("write synthetic grayscale presentation object")?;
    Ok((
        filename,
        std::fs::read(path).context("read synthetic grayscale presentation object")?,
    ))
}

/// Write one scalar Part 10 multi-frame object and return its name and bytes.
///
/// `declared_frames` deliberately remains a parameter so malformed frame
/// counts can exercise the same scanner and reader path as valid input.
/// `temporal_positions`, when present, emits NumberOfTemporalPositions to
/// verify that the spatial viewer rejects temporal organizations explicitly.
#[cfg(test)]
pub(crate) fn write_multiframe(
    root: &Path,
    declared_frames: usize,
    temporal_positions: Option<u16>,
) -> Result<(String, Vec<u8>)> {
    std::fs::create_dir_all(root).context("create synthetic multiframe directory")?;
    let mut model = DicomObjectModel::new();
    for (group, element, vr, value) in [
        (0x0008, 0x0016, "UI", "1.2.840.10008.5.1.4.1.1.7.3"),
        (0x0008, 0x0018, "UI", "2.25.20260905003.1"),
        (0x0008, 0x0060, "CS", "CT"),
        (0x0008, 0x0064, "CS", "WSD"),
        (0x0020, 0x000D, "UI", "2.25.20260905004"),
        (0x0020, 0x000E, "UI", MULTIFRAME_SERIES_UID),
        (0x0020, 0x0032, "DS", "10\\20\\30"),
        (0x0020, 0x0037, "DS", "0\\1\\0\\0\\0\\1"),
        (0x0028, 0x0004, "CS", "MONOCHROME2"),
        (0x0028, 0x0030, "DS", "1.5\\0.5"),
        (0x0018, 0x0050, "DS", "2"),
        (0x0028, 0x1052, "DS", "-10"),
        (0x0028, 0x1053, "DS", "2"),
    ] {
        model.insert(DicomObjectNode::text(
            DicomTag::new(group, element),
            vr,
            value,
        ));
    }
    model.insert(DicomObjectNode::text(
        DicomTag::new(0x0028, 0x0008),
        "IS",
        declared_frames.to_string(),
    ));
    if let Some(positions) = temporal_positions {
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0020, 0x0105),
            "US",
            positions,
        ));
    }
    for (element, value) in [
        (0x0002, 1_u16),
        (0x0010, 2),
        (0x0011, 2),
        (0x0100, 8),
        (0x0101, 8),
        (0x0102, 7),
        (0x0103, 0),
    ] {
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, element),
            "US",
            value,
        ));
    }
    model.insert(DicomObjectNode::bytes(
        DicomTag::new(0x7FE0, 0x0010),
        "OB",
        MULTIFRAME_RAW.to_vec(),
    ));

    let filename = "multiframe.dcm".to_owned();
    let path = root.join(&filename);
    ritk_io::write_dicom_object(&model, &path).context("write synthetic multiframe object")?;
    Ok((
        filename,
        std::fs::read(path).context("read synthetic multiframe object")?,
    ))
}

/// Write a two-frame, interleaved unsigned RGB Part 10 object.
pub(crate) fn write_color_multiframe(root: &Path) -> Result<(String, Vec<u8>)> {
    std::fs::create_dir_all(root).context("create synthetic RGB multiframe directory")?;
    let mut model = DicomObjectModel::new();
    for (group, element, vr, value) in [
        (0x0008, 0x0016, "UI", "1.2.840.10008.5.1.4.1.1.7.4"),
        (0x0008, 0x0018, "UI", "2.25.20260905003.1"),
        (0x0008, 0x0060, "CS", "OT"),
        (0x0008, 0x0064, "CS", "WSD"),
        (0x0020, 0x000D, "UI", "2.25.20260905004"),
        (0x0020, 0x000E, "UI", "2.25.20260905003"),
        (0x0020, 0x0032, "DS", "1\\2\\3"),
        (0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0"),
        (0x0028, 0x0004, "CS", "RGB"),
        (0x0028, 0x0030, "DS", "0.5\\0.25"),
        (0x0018, 0x0050, "DS", "2.0"),
    ] {
        model.insert(DicomObjectNode::text(
            DicomTag::new(group, element),
            vr,
            value,
        ));
    }
    model.insert(DicomObjectNode::text(
        DicomTag::new(0x0028, 0x0008),
        "IS",
        COLOR_MULTIFRAME_SHAPE[0].to_string(),
    ));
    for (element, value) in [
        (0x0002, 3_u16),
        (0x0006, 0_u16),
        (0x0010, 2),
        (0x0011, 2),
        (0x0100, 8),
        (0x0101, 8),
        (0x0102, 7),
        (0x0103, 0),
    ] {
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, element),
            "US",
            value,
        ));
    }
    model.insert(DicomObjectNode::bytes(
        DicomTag::new(0x7FE0, 0x0010),
        "OB",
        COLOR_MULTIFRAME_RAW.to_vec(),
    ));

    let filename = "rgb-multiframe.dcm".to_owned();
    let path = root.join(&filename);
    ritk_io::write_dicom_object(&model, &path).context("write synthetic RGB multiframe object")?;
    Ok((
        filename,
        std::fs::read(path).context("read synthetic RGB multiframe object")?,
    ))
}

/// Write three unsigned, single-frame images and return their Part 10 bytes.
///
/// Filenames, instance numbers, and returned batch order oppose the spatial
/// order so the decoder must sort by ImagePositionPatient. Eight-bit samples
/// use the object writer's OB representation without per-slice quantization.
pub(crate) fn write_study(
    root: &Path,
    modality: &str,
    series_uid: &str,
) -> Result<Vec<(String, Vec<u8>)>> {
    std::fs::create_dir_all(root).context("create synthetic study directory")?;
    let mut files = Vec::with_capacity(SHAPE[0]);
    for (depth, filename, instance) in [(2_u16, "a.dcm", 1), (1, "b.dcm", 2), (0, "c.dcm", 3)] {
        let mut model = DicomObjectModel::new();
        let sop_uid = format!("{series_uid}.{instance}");
        let position = format!("{}\\20\\30", 10 + 2 * depth);
        for (group, element, vr, value) in [
            (0x0008, 0x0016, "UI", "1.2.840.10008.5.1.4.1.1.7"),
            (0x0008, 0x0018, "UI", sop_uid.as_str()),
            (0x0008, 0x0060, "CS", modality),
            (0x0008, 0x0064, "CS", "WSD"),
            (0x0020, 0x000D, "UI", "2.25.20260905"),
            (0x0020, 0x000E, "UI", series_uid),
            (0x0020, 0x0032, "DS", position.as_str()),
            (0x0020, 0x0037, "DS", "0\\1\\0\\0\\0\\1"),
            (0x0028, 0x0004, "CS", "MONOCHROME2"),
            (0x0028, 0x0030, "DS", "1.5\\0.5"),
            (0x0018, 0x0050, "DS", "2"),
            (0x0028, 0x1052, "DS", "-20"),
            (0x0028, 0x1053, "DS", "2"),
        ] {
            model.insert(DicomObjectNode::text(
                DicomTag::new(group, element),
                vr,
                value,
            ));
        }
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0020, 0x0013),
            "IS",
            instance.to_string(),
        ));
        for (element, value) in [
            (0x0002, 1_u16),
            (0x0010, 2),
            (0x0011, 4),
            (0x0100, 8),
            (0x0101, 8),
            (0x0102, 7),
            (0x0103, 0),
        ] {
            model.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0028, element),
                "US",
                value,
            ));
        }
        let start = usize::from(depth) * 8;
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x7FE0, 0x0010),
            "OB",
            SAMPLES[start..start + 8].to_vec(),
        ));
        let filename = format!("{series_uid}_{filename}");
        let path = root.join(&filename);
        ritk_io::write_dicom_object(&model, &path).context("write synthetic Part 10 instance")?;
        files.push((
            filename.to_owned(),
            std::fs::read(path).context("read synthetic Part 10 instance")?,
        ));
    }
    Ok(files)
}
