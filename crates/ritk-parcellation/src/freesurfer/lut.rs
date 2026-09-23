//! FreeSurfer colour lookup tables (`FreeSurferColorLUT.txt`).
//!
//! A lookup table names and colours the integer labels of a segmentation or
//! parcellation: `aseg` structures, `aparc` cortical parcels, and the rest of
//! FreeSurfer's label space. The same table, in binary form, is embedded in
//! every annotation file ([`super::SurfaceAnnotation`]). Both read into the
//! stack's one label-table type, [`LabelTable`]; this module is its text
//! format.
//!
//! # Text format
//!
//! Following FreeSurfer's reader, `CTABreadASCII2` in `utils/colortab.cpp`
//! (<https://github.com/freesurfer/freesurfer/blob/dev/utils/colortab.cpp>):
//!
//! ```text
//! #No. Label Name:                R   G   B   A
//! 0   Unknown                     0   0   0   0
//! 2   Left-Cerebral-White-Matter  245 245 245 0
//! ```
//!
//! An entry is a line that `sscanf("%d %s %d %d %d %d")` accepts: label, name,
//! red, green, blue, and a fourth component FreeSurfer stores as *transparency*
//! (`ai = 255 - t`, "alpha = 255-trans", in `CTABreadASCII2`), which becomes the
//! entry's alpha. An optional seventh integer is a tissue type, which this
//! reader ignores. Lines not beginning with an integer are comments. Two points
//! are stricter than the reference: a line that begins with an integer but is
//! not a complete entry is an error rather than a silently skipped line, and a
//! repeated label is an error rather than a warning with the first occurrence
//! kept. Both are signs of corruption, and a table that silently loses a region
//! misnames it everywhere downstream.

use std::io::{BufRead, BufReader, Read, Write};

use ritk_annotation::{LabelTable, RgbaBytes};

use super::{FreeSurferError, FreeSurferFormat};

const FORMAT: FreeSurferFormat = FreeSurferFormat::ColorLut;

/// Parse the text format into a table whose entries keep file order.
///
/// # Errors
///
/// [`FreeSurferError::Io`] on read failure or text that is not UTF-8;
/// [`FreeSurferError::Malformed`], indexed by 1-based line number, for an
/// incomplete entry, a negative label, a colour component outside `0..=255`,
/// or a repeated label; [`FreeSurferError::InvalidCount`] when the file holds
/// no entry at all, which FreeSurfer also rejects ("no structures found").
///
/// # Examples
///
/// ```
/// use ritk_parcellation::freesurfer::lut;
///
/// let text = "#No. Label Name: R G B A\n17 Left-Hippocampus 220 216 20 0\n";
/// let table = lut::read(text.as_bytes())?;
/// let entry = table.get_label(17).expect("present");
/// assert_eq!(entry.name, "Left-Hippocampus");
/// assert_eq!(entry.color.a(), 255);
/// # Ok::<(), ritk_parcellation::freesurfer::FreeSurferError>(())
/// ```
pub fn read(reader: impl Read) -> Result<LabelTable, FreeSurferError> {
    let mut table = LabelTable::new();
    for (index, line) in BufReader::new(reader).lines().enumerate() {
        let line_number = index + 1;
        if let Some((label, name, color)) = parse_line(&line?, line_number)? {
            table.add_label(label, name, color).map_err(|error| {
                FreeSurferError::malformed(FORMAT, "line", line_number, error.to_string())
            })?;
        }
    }
    if table.is_empty() {
        return Err(FreeSurferError::InvalidCount {
            format: FORMAT,
            field: "entry count",
            count: 0,
            max: i64::from(u32::MAX),
        });
    }
    Ok(table)
}

/// Write the text format, one entry per line in table order under a header
/// comment.
///
/// # Errors
///
/// [`FreeSurferError::Malformed`], indexed by label, when a name is empty or
/// contains whitespace, which the text format cannot represent;
/// [`FreeSurferError::Io`] on write failure.
pub fn write(table: &LabelTable, mut writer: impl Write) -> Result<(), FreeSurferError> {
    writeln!(writer, "#No. Label Name: R G B A")?;
    for entry in table.entries() {
        let label = u32::from(entry.id);
        if entry.name.is_empty() || entry.name.chars().any(char::is_whitespace) {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "name of label",
                label as usize,
                format!("{:?} is not a single whitespace-free token", entry.name),
            ));
        }
        let [red, green, blue, transparency] = stored_components(entry.color);
        writeln!(
            writer,
            "{label} {} {red} {green} {blue} {transparency}",
            entry.name
        )?;
    }
    Ok(())
}

/// `(label, name)` pairs in table order: the `region_names` a
/// [`crate::Parcellation`] takes.
#[must_use]
pub fn region_names(table: &LabelTable) -> Vec<(u32, String)> {
    table
        .entries()
        .iter()
        .map(|entry| (u32::from(entry.id), entry.name.clone()))
        .collect()
}

/// The colour FreeSurfer's four stored components denote: red, green, blue,
/// and transparency, with `alpha = 255 - transparency` (`CTABreadASCII2`).
pub(crate) const fn color_from_stored([red, green, blue, transparency]: [u8; 4]) -> RgbaBytes {
    RgbaBytes([red, green, blue, 255 - transparency])
}

/// The four components FreeSurfer stores for `color`; the inverse of
/// [`color_from_stored`].
pub(crate) const fn stored_components(color: RgbaBytes) -> [u8; 4] {
    let [red, green, blue, alpha] = color.0;
    [red, green, blue, 255 - alpha]
}

/// Parse one line: `None` for a comment, otherwise an entry or an error.
fn parse_line(
    line: &str,
    line_number: usize,
) -> Result<Option<(u32, String, RgbaBytes)>, FreeSurferError> {
    let mut tokens = line.split_whitespace();
    let Some(label) = tokens.next().and_then(|token| token.parse::<i64>().ok()) else {
        return Ok(None);
    };
    let bad = |reason: String| FreeSurferError::malformed(FORMAT, "line", line_number, reason);
    let label =
        u32::try_from(label).map_err(|_| bad(format!("label {label} is outside 0..=u32::MAX")))?;
    let name = tokens
        .next()
        .ok_or_else(|| bad("entry has a label but no name".to_owned()))?;
    let mut components = [0_u8; 4];
    for (slot, component) in components
        .iter_mut()
        .zip(["red", "green", "blue", "transparency"])
    {
        let token = tokens
            .next()
            .ok_or_else(|| bad(format!("entry is missing its {component} component")))?;
        *slot = token.parse::<u8>().map_err(|_| {
            bad(format!(
                "{component} {token:?} is not an integer in 0..=255"
            ))
        })?;
    }
    // `split_whitespace` guarantees a non-empty, whitespace-free name.
    Ok(Some((
        label,
        name.to_owned(),
        color_from_stored(components),
    )))
}

#[cfg(test)]
mod tests;
