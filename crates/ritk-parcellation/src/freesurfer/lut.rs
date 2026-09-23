//! FreeSurfer colour lookup tables (`FreeSurferColorLUT.txt`).
//!
//! A lookup table names and colours the integer labels of a segmentation or
//! parcellation: `aseg` structures, `aparc` cortical parcels, and the rest of
//! FreeSurfer's label space. The same table, in binary form, is embedded in
//! every annotation file ([`super::SurfaceAnnotation`]).
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
//! (`ai = 255 - t`, "alpha = 255-trans", in `CTABreadASCII2`). An optional
//! seventh integer is a tissue type, which this reader ignores. Lines not
//! beginning with an integer are comments. Two points are stricter than the
//! reference: a line that begins with an integer but is not a complete entry is
//! an error rather than a silently skipped line, and a repeated label is an
//! error rather than a warning with the first occurrence kept. Both are signs
//! of corruption, and a table that silently loses a region misnames it
//! everywhere downstream.

use std::io::{BufRead, BufReader, Read, Write};

use super::{FreeSurferError, FreeSurferFormat};

const FORMAT: FreeSurferFormat = FreeSurferFormat::ColorLut;

/// An entry's colour as FreeSurfer stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LutColor {
    /// Red, `0..=255`.
    pub red: u8,
    /// Green, `0..=255`.
    pub green: u8,
    /// Blue, `0..=255`.
    pub blue: u8,
    /// Transparency as stored; `0` is opaque.
    pub transparency: u8,
}

impl LutColor {
    /// Opacity, `255 - transparency`, as FreeSurfer derives it.
    #[must_use]
    pub const fn alpha(self) -> u8 {
        255 - self.transparency
    }

    /// The value identifying this colour in an `.annot` file:
    /// `red + green·2⁸ + blue·2¹⁶` (`read_annotation.m`, colour table column 5).
    #[must_use]
    pub const fn annotation_value(self) -> u32 {
        self.red as u32 | (self.green as u32) << 8 | (self.blue as u32) << 16
    }
}

/// One named, coloured label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LutEntry {
    label: u32,
    name: String,
    color: LutColor,
}

impl LutEntry {
    /// An entry, provided the name is writable as one text-format token.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Malformed`] when `name` is empty or contains
    /// whitespace, which the text format cannot represent.
    pub fn new(label: u32, name: String, color: LutColor) -> Result<Self, FreeSurferError> {
        if name.is_empty() || name.chars().any(char::is_whitespace) {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "name of label",
                label as usize,
                format!("{name:?} is not a single whitespace-free token"),
            ));
        }
        Ok(Self { label, name, color })
    }

    /// The integer label.
    #[must_use]
    pub const fn label(&self) -> u32 {
        self.label
    }

    /// The region name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The colour.
    #[must_use]
    pub const fn color(&self) -> LutColor {
        self.color
    }
}

/// A colour lookup table: entries with unique labels, ordered by label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorLut {
    entries: Box<[LutEntry]>,
}

impl ColorLut {
    /// A table from `entries` in any order.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Malformed`] when two entries share a label.
    pub fn new(entries: impl IntoIterator<Item = LutEntry>) -> Result<Self, FreeSurferError> {
        let mut entries: Vec<LutEntry> = entries.into_iter().collect();
        entries.sort_by_key(LutEntry::label);
        if let Some((first, second)) = entries.windows(2).find_map(|pair| match pair {
            [first, second] if first.label == second.label => Some((first, second)),
            _ => None,
        }) {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "label",
                second.label as usize,
                format!("repeated, as {:?} and {:?}", first.name, second.name),
            ));
        }
        Ok(Self {
            entries: entries.into_boxed_slice(),
        })
    }

    /// Parse the text format.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Io`] on read failure or text that is not UTF-8;
    /// [`FreeSurferError::Malformed`], indexed by 1-based line number, for an
    /// incomplete entry, a negative label, a colour component outside
    /// `0..=255`, or a repeated label; [`FreeSurferError::InvalidCount`] when
    /// the file holds no entry at all, which FreeSurfer also rejects ("no
    /// structures found").
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_parcellation::freesurfer::ColorLut;
    ///
    /// let text = "#No. Label Name: R G B A\n17 Left-Hippocampus 220 216 20 0\n";
    /// let lut = ColorLut::parse(text.as_bytes())?;
    /// let entry = lut.get(17).expect("present");
    /// assert_eq!(entry.name(), "Left-Hippocampus");
    /// assert_eq!(entry.color().alpha(), 255);
    /// # Ok::<(), ritk_parcellation::freesurfer::FreeSurferError>(())
    /// ```
    pub fn parse(reader: impl Read) -> Result<Self, FreeSurferError> {
        let mut entries = Vec::new();
        for (index, line) in BufReader::new(reader).lines().enumerate() {
            if let Some(entry) = parse_line(&line?, index + 1)? {
                entries.push(entry);
            }
        }
        if entries.is_empty() {
            return Err(FreeSurferError::InvalidCount {
                format: FORMAT,
                field: "entry count",
                count: 0,
                max: i64::from(u32::MAX),
            });
        }
        Self::new(entries)
    }

    /// Write the text format, one entry per line under a header comment.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Io`] on write failure.
    pub fn write(&self, mut writer: impl Write) -> Result<(), FreeSurferError> {
        writeln!(writer, "#No. Label Name: R G B A")?;
        for entry in &self.entries {
            let color = entry.color;
            writeln!(
                writer,
                "{} {} {} {} {} {}",
                entry.label, entry.name, color.red, color.green, color.blue, color.transparency
            )?;
        }
        Ok(())
    }

    /// Entries, ordered by label.
    #[must_use]
    pub fn entries(&self) -> &[LutEntry] {
        &self.entries
    }

    /// The entry for `label`, if the table has one.
    #[must_use]
    pub fn get(&self, label: u32) -> Option<&LutEntry> {
        self.entries
            .binary_search_by_key(&label, LutEntry::label)
            .ok()
            .and_then(|index| self.entries.get(index))
    }

    /// `(label, name)` pairs: the `region_names` a [`crate::Parcellation`]
    /// takes.
    #[must_use]
    pub fn region_names(&self) -> Vec<(u32, String)> {
        self.entries
            .iter()
            .map(|entry| (entry.label, entry.name.clone()))
            .collect()
    }
}

/// Parse one line: `None` for a comment, otherwise an entry or an error.
fn parse_line(line: &str, line_number: usize) -> Result<Option<LutEntry>, FreeSurferError> {
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
    let [red, green, blue, transparency] = components;
    let color = LutColor {
        red,
        green,
        blue,
        transparency,
    };
    // `split_whitespace` guarantees a non-empty, whitespace-free name.
    Ok(Some(LutEntry {
        label,
        name: name.to_owned(),
        color,
    }))
}

#[cfg(test)]
mod tests;
