//! FreeSurfer ASCII label files (`lh.cortex.label`, `lh.V1_exvivo.label`).
//!
//! A label is a set of vertices, each with a position and a scalar, typically
//! a region drawn on a surface or thresholded from a map.
//!
//! # Format
//!
//! Per FreeSurfer's `matlab/read_label.m`
//! (<https://github.com/freesurfer/freesurfer/blob/dev/matlab/read_label.m>):
//!
//! ```text
//! #!ascii label  , from subject bert vox2ras=TkReg
//! 3
//! 1025  -35.1  -18.4  51.2  0.0
//! …
//! ```
//!
//! The first line is a comment, the second the vertex count, and then one
//! record per vertex: vertex number, `x y z` in surface RAS millimetres, and a
//! value — read by `fscanf('%d %f %f %f %f')`, so records are whitespace
//! separated with no regard to line breaks. A vertex number of `-1` marks a
//! point from a volume label that belongs to no surface vertex.
//!
//! Stricter than `fscanf`: the record count must match the header exactly,
//! trailing content is an error, and coordinates and values must be finite.

use std::io::{Read, Write};

use super::surface::MAX_ELEMENTS;
use super::{FreeSurferError, FreeSurferFormat};

const FORMAT: FreeSurferFormat = FreeSurferFormat::Label;

/// One labelled point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelVertex {
    /// The surface vertex, or `None` for a point that is not on a surface.
    pub vertex: Option<u32>,
    /// Position in surface RAS, millimetres.
    pub position: [f64; 3],
    /// The scalar carried with the point.
    pub value: f64,
}

/// A FreeSurfer label.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceLabel {
    comment: String,
    vertices: Box<[LabelVertex]>,
}

impl SurfaceLabel {
    /// A label with a header comment.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Malformed`] when the comment spans lines, or a
    /// coordinate or value is not finite.
    pub fn new(comment: String, vertices: Box<[LabelVertex]>) -> Result<Self, FreeSurferError> {
        if comment.contains(['\n', '\r']) {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "line",
                1,
                "comment spans more than one line",
            ));
        }
        if let Some(index) = vertices.iter().position(|point| {
            !point.value.is_finite() || point.position.iter().any(|axis| !axis.is_finite())
        }) {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "record",
                index,
                "coordinate or value is not finite",
            ));
        }
        Ok(Self { comment, vertices })
    }

    /// Parse a label file.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Io`] on read failure or text that is not UTF-8;
    /// [`FreeSurferError::InvalidCount`] for a count outside what a surface
    /// holds; [`FreeSurferError::Malformed`] for a missing header, a record
    /// that does not parse, fewer or more records than the count, a vertex
    /// number below `-1`, or a non-finite coordinate or value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_parcellation::freesurfer::SurfaceLabel;
    ///
    /// let text = "#!ascii label\n2\n7 1.0 2.0 3.0 0.5\n-1 4.0 5.0 6.0 0.0\n";
    /// let label = SurfaceLabel::read(text.as_bytes())?;
    /// assert_eq!(label.vertices()[0].vertex, Some(7));
    /// assert_eq!(label.vertices()[1].vertex, None);
    /// # Ok::<(), ritk_parcellation::freesurfer::FreeSurferError>(())
    /// ```
    pub fn read(mut reader: impl Read) -> Result<Self, FreeSurferError> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        let (comment, rest) = text
            .split_once('\n')
            .ok_or_else(|| FreeSurferError::malformed(FORMAT, "line", 1, "no header line"))?;
        let comment = comment.trim_end_matches('\r');
        let comment = comment.strip_prefix('#').unwrap_or(comment).to_owned();

        let mut tokens = rest.split_whitespace();
        let count_token = tokens
            .next()
            .ok_or_else(|| FreeSurferError::malformed(FORMAT, "line", 2, "no vertex count"))?;
        let count = count_token.parse::<i64>().map_err(|_| {
            FreeSurferError::malformed(
                FORMAT,
                "line",
                2,
                format!("vertex count {count_token:?} is not an integer"),
            )
        })?;
        let count = usize::try_from(count)
            .ok()
            .filter(|count| *count <= MAX_ELEMENTS)
            .ok_or(FreeSurferError::InvalidCount {
                format: FORMAT,
                field: "vertex count",
                count,
                max: i64::try_from(MAX_ELEMENTS).unwrap_or(i64::MAX),
            })?;

        let mut vertices = Vec::new();
        for index in 0..count {
            vertices.push(parse_record(&mut tokens, index)?);
        }
        if let Some(extra) = tokens.next() {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "record",
                count,
                format!("trailing content {extra:?} after {count} records"),
            ));
        }
        Self::new(comment, vertices.into_boxed_slice())
    }

    /// Write the label file.
    ///
    /// Numbers are written in Rust's shortest round-tripping form, so reading
    /// the output back yields identical values.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Io`] on write failure.
    pub fn write(&self, mut writer: impl Write) -> Result<(), FreeSurferError> {
        writeln!(writer, "#{}", self.comment)?;
        writeln!(writer, "{}", self.vertices.len())?;
        for point in &self.vertices {
            let vertex = point.vertex.map_or(-1, i64::from);
            let [x, y, z] = point.position;
            writeln!(writer, "{vertex} {x} {y} {z} {}", point.value)?;
        }
        Ok(())
    }

    /// The header comment, without its leading `#`.
    #[must_use]
    pub fn comment(&self) -> &str {
        &self.comment
    }

    /// The labelled points, in file order.
    #[must_use]
    pub fn vertices(&self) -> &[LabelVertex] {
        &self.vertices
    }
}

/// Parse one five-field record.
fn parse_record<'text>(
    tokens: &mut impl Iterator<Item = &'text str>,
    index: usize,
) -> Result<LabelVertex, FreeSurferError> {
    let bad = |reason: String| FreeSurferError::malformed(FORMAT, "record", index, reason);
    let mut next = |field: &str| {
        tokens
            .next()
            .ok_or_else(|| bad(format!("file ends before its {field}")))
    };
    let vertex_token = next("vertex number")?;
    let vertex = match vertex_token.parse::<i64>() {
        Ok(-1) => None,
        Ok(vertex) => Some(
            u32::try_from(vertex)
                .map_err(|_| bad(format!("vertex number {vertex} is neither -1 nor a vertex")))?,
        ),
        Err(_) => {
            return Err(bad(format!(
                "vertex number {vertex_token:?} is not an integer"
            )));
        }
    };
    let mut numbers = [0.0_f64; 4];
    for (slot, field) in numbers.iter_mut().zip(["x", "y", "z", "value"]) {
        let token = next(field)?;
        *slot = token
            .parse::<f64>()
            .map_err(|_| bad(format!("{field} {token:?} is not a number")))?;
    }
    let [x, y, z, value] = numbers;
    Ok(LabelVertex {
        vertex,
        position: [x, y, z],
        value,
    })
}

#[cfg(test)]
mod tests;
