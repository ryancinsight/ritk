//! FreeSurfer surface annotations (`lh.aparc.annot`).
//!
//! An annotation assigns each vertex of a surface to a structure of an
//! embedded colour table. It stores the assignment by *colour*: each vertex
//! carries the packed RGB value of its structure, and the structure is found by
//! matching that value against the table.
//!
//! # Format
//!
//! Big-endian `i32` throughout, per FreeSurfer's `matlab/read_annotation.m`
//! (<https://github.com/freesurfer/freesurfer/blob/dev/matlab/read_annotation.m>)
//! and nibabel's `freesurfer/io.py` `read_annot`/`write_annot`
//! (<https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py>):
//!
//! ```text
//! vertex count      n
//! n × (vertex, value)   value = red + green·2⁸ + blue·2¹⁶, 0 = unlabelled
//! colour-table tag  1
//! colour table, one of:
//!   old format:  entry count (> 0), path length, path,
//!                entry count × (name length, name, r, g, b, t)   index = position
//!   version 2:   -2, max index, path length, path, entry count,
//!                entry count × (index, name length, name, r, g, b, t)
//! ```
//!
//! Names and the path carry their terminating NUL inside the stated length.
//! [`SurfaceAnnotation::write`] emits version 2, as `write_annot` does.
//!
//! # Labels
//!
//! A vertex's label is its structure's colour-table index. An unlabelled vertex
//! (value `0`) reads as [`crate::BACKGROUND`], which coincides with index `0` —
//! FreeSurfer's own tables reserve that index for `unknown`, so the collision is
//! the convention rather than a loss. Two table entries with the same colour
//! make the file ambiguous and are rejected, as is a vertex value no entry has.

use std::collections::HashMap;
use std::io::{Read, Write};

use super::big_endian::{bounded_count, read_be, read_count, reserve_for, write_be, write_count};
use super::surface::MAX_ELEMENTS;
use super::{ColorLut, FreeSurferError, FreeSurferFormat, LutColor, LutEntry};
use crate::BACKGROUND;

const FORMAT: FreeSurferFormat = FreeSurferFormat::Annotation;

/// Largest colour-table index accepted.
///
/// FreeSurfer's full `FreeSurferColorLUT.txt` tops out near 15,000; a million
/// is far past any real table and short of what a corrupt field would demand.
const MAX_TABLE_INDEX: usize = 1_000_000;

/// Longest name or path string accepted, in bytes (`PATH_MAX` on Linux).
const MAX_STRING: usize = 4096;

/// The `-2` that opens a version 2 colour table.
const CTAB_VERSION_2: i32 = -2;

/// Per-vertex structure labels with the colour table that names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceAnnotation {
    vertex_labels: Box<[u32]>,
    color_table: ColorLut,
}

impl SurfaceAnnotation {
    /// An annotation from per-vertex labels and the table naming them.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Malformed`] when a vertex carries a label other than
    /// [`BACKGROUND`] that the table lacks, or two entries share a colour.
    pub fn new(vertex_labels: Box<[u32]>, color_table: ColorLut) -> Result<Self, FreeSurferError> {
        unique_colors(&color_table)?;
        if let Some((vertex, label)) = vertex_labels
            .iter()
            .enumerate()
            .find(|(_, label)| **label != BACKGROUND && color_table.get(**label).is_none())
        {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "vertex",
                vertex,
                format!("label {label} is not in the colour table"),
            ));
        }
        Ok(Self {
            vertex_labels,
            color_table,
        })
    }

    /// Read an annotation file.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::InvalidCount`] for a count outside what a real file
    /// holds; [`FreeSurferError::Unsupported`] for a missing colour table or a
    /// table version other than 2; [`FreeSurferError::Malformed`] for a vertex
    /// index outside the surface, a value no entry has, a colour component
    /// outside `0..=255`, a name that is not UTF-8, or duplicate entries;
    /// [`FreeSurferError::Io`] for a file shorter than its counts promise.
    pub fn read(mut reader: impl Read) -> Result<Self, FreeSurferError> {
        let vertex_count = read_count(&mut reader, FORMAT, "vertex count", MAX_ELEMENTS)?;
        let mut pairs = Vec::with_capacity(reserve_for(vertex_count));
        for _ in 0..vertex_count {
            pairs.push((read_be::<i32>(&mut reader)?, read_be::<i32>(&mut reader)?));
        }

        let tag = read_be::<i32>(&mut reader)?;
        if tag != 1 {
            return Err(FreeSurferError::Unsupported {
                format: FORMAT,
                field: "colour-table tag",
                got: i64::from(tag),
            });
        }
        let color_table = read_color_table(&mut reader)?;
        let by_value = unique_colors(&color_table)?;

        let mut vertex_labels = vec![BACKGROUND; vertex_count].into_boxed_slice();
        for (position, (vertex, value)) in pairs.into_iter().enumerate() {
            let slot = usize::try_from(vertex)
                .ok()
                .and_then(|vertex| vertex_labels.get_mut(vertex))
                .ok_or_else(|| {
                    FreeSurferError::malformed(
                        FORMAT,
                        "vertex record",
                        position,
                        format!("vertex {vertex} outside 0..{vertex_count}"),
                    )
                })?;
            if value == 0 {
                *slot = BACKGROUND;
                continue;
            }
            *slot = u32::try_from(value)
                .ok()
                .and_then(|value| by_value.get(&value).copied())
                .ok_or_else(|| {
                    FreeSurferError::malformed(
                        FORMAT,
                        "vertex record",
                        position,
                        format!("value {value:#08x} matches no colour-table entry"),
                    )
                })?;
        }
        Ok(Self {
            vertex_labels,
            color_table,
        })
    }

    /// Write the annotation with a version 2 colour table.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Io`] on write failure;
    /// [`FreeSurferError::InvalidCount`] when a count exceeds the `i32` fields.
    pub fn write(&self, mut writer: impl Write) -> Result<(), FreeSurferError> {
        let writer = &mut writer;
        write_count(writer, FORMAT, "vertex count", self.vertex_labels.len())?;
        for (vertex, label) in self.vertex_labels.iter().enumerate() {
            write_count(writer, FORMAT, "vertex", vertex)?;
            let value = match self.color_table.get(*label) {
                Some(entry) if *label != BACKGROUND => entry.color().annotation_value(),
                _ => 0,
            };
            write_count(writer, FORMAT, "annotation value", value as usize)?;
        }
        write_be(writer, 1_i32)?;
        write_be(writer, CTAB_VERSION_2)?;
        let max_index = self
            .color_table
            .entries()
            .last()
            .map_or(0, |entry| entry.label() as usize + 1);
        write_count(writer, FORMAT, "max index", max_index)?;
        write_string(writer, "NOFILE")?;
        write_count(
            writer,
            FORMAT,
            "entry count",
            self.color_table.entries().len(),
        )?;
        for entry in self.color_table.entries() {
            write_count(writer, FORMAT, "entry index", entry.label() as usize)?;
            write_string(writer, entry.name())?;
            let color = entry.color();
            for component in [color.red, color.green, color.blue, color.transparency] {
                write_be(writer, i32::from(component))?;
            }
        }
        Ok(())
    }

    /// Per-vertex labels: colour-table indices, [`BACKGROUND`] where unlabelled.
    #[must_use]
    pub fn vertex_labels(&self) -> &[u32] {
        &self.vertex_labels
    }

    /// The embedded colour table, keyed by structure index.
    #[must_use]
    pub const fn color_table(&self) -> &ColorLut {
        &self.color_table
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertex_labels.len()
    }
}

/// Map annotation value to label, rejecting a colour two entries share.
fn unique_colors(table: &ColorLut) -> Result<HashMap<u32, u32>, FreeSurferError> {
    let mut by_value = HashMap::with_capacity(table.entries().len());
    for entry in table.entries() {
        if let Some(previous) = by_value.insert(entry.color().annotation_value(), entry.label()) {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "colour-table entry",
                entry.label() as usize,
                format!("shares its colour with entry {previous}"),
            ));
        }
    }
    Ok(by_value)
}

/// Read the colour table in either the old or the version 2 layout.
fn read_color_table(reader: &mut impl Read) -> Result<ColorLut, FreeSurferError> {
    let first = read_be::<i32>(reader)?;
    let mut entries = Vec::new();
    if first > 0 {
        let count = bounded_count(first, FORMAT, "entry count", MAX_TABLE_INDEX)?;
        read_string(reader, 0)?;
        for index in 0..count {
            entries.push(read_entry(reader, index)?);
        }
    } else if first == CTAB_VERSION_2 {
        let max_index = read_count(reader, FORMAT, "max index", MAX_TABLE_INDEX)?;
        read_string(reader, 0)?;
        let count = read_count(reader, FORMAT, "entry count", max_index)?;
        for position in 0..count {
            let index = read_be::<i32>(reader)?;
            let index = usize::try_from(index)
                .ok()
                .filter(|index| *index < max_index)
                .ok_or_else(|| {
                    FreeSurferError::malformed(
                        FORMAT,
                        "colour-table entry",
                        position,
                        format!("index {index} outside 0..{max_index}"),
                    )
                })?;
            entries.push(read_entry(reader, index)?);
        }
    } else {
        return Err(FreeSurferError::Unsupported {
            format: FORMAT,
            field: "colour-table version",
            got: -i64::from(first),
        });
    }
    ColorLut::new(entries)
}

/// Read one entry's name and colour; `index` is its structure index.
fn read_entry(reader: &mut impl Read, index: usize) -> Result<LutEntry, FreeSurferError> {
    let name = read_string(reader, index)?;
    let mut components = [0_u8; 4];
    for slot in &mut components {
        let value = read_be::<i32>(reader)?;
        *slot = u8::try_from(value).map_err(|_| {
            FreeSurferError::malformed(
                FORMAT,
                "colour-table entry",
                index,
                format!("colour component {value} outside 0..=255"),
            )
        })?;
    }
    let [red, green, blue, transparency] = components;
    let label = u32::try_from(index).map_err(|_| {
        FreeSurferError::malformed(FORMAT, "colour-table entry", index, "index exceeds u32")
    })?;
    LutEntry::new(
        label,
        name,
        LutColor {
            red,
            green,
            blue,
            transparency,
        },
    )
}

/// Read a length-prefixed string whose length counts its trailing NUL.
fn read_string(reader: &mut impl Read, index: usize) -> Result<String, FreeSurferError> {
    let length = read_count(reader, FORMAT, "string length", MAX_STRING)?;
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes)?;
    let text = bytes.strip_suffix(&[0]).unwrap_or(&bytes);
    String::from_utf8(text.to_vec()).map_err(|_| {
        FreeSurferError::malformed(FORMAT, "colour-table entry", index, "name is not UTF-8")
    })
}

/// Write a string with its trailing NUL counted in the length prefix.
fn write_string(writer: &mut impl Write, text: &str) -> Result<(), FreeSurferError> {
    write_count(writer, FORMAT, "string length", text.len() + 1)?;
    writer.write_all(text.as_bytes())?;
    writer.write_all(&[0])?;
    Ok(())
}

#[cfg(test)]
mod tests;
