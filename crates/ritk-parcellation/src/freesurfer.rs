//! FreeSurfer surface-based parcellation format readers.
//!
//! This module parses FreeSurfer surface annotation (`.annot`) files and
//! the FreeSurfer colour lookup table (`FreeSurferColorLUT.txt`).  These
//! formats define parcellation regions on cortical surface meshes rather
//! than in volumetric space.
//!
//! # Conversion to volume
//!
//! A [`SurfaceAnnotation`] labels *vertices of a mesh*, not voxels. Turning one
//! into a [`crate::Parcellation`] therefore needs the surface geometry the
//! annotation refers to — the vertex coordinates and faces of the matching
//! `lh.white`/`rh.white` surface — plus a rasterisation of the cortical ribbon
//! between the white and pial surfaces. Neither is supplied here, so an
//! annotation is currently readable but not yet convertible; the
//! [`read_freesurfer_lut`] table it pairs with is directly usable as the
//! `region_names` of a volumetric parcellation from any source.
//!
//! # References
//!
//! * FreeSurfer annotation format:
//!   <https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferFileFormats>

use std::io::{BufRead, BufReader, Read};

/// Error returned when reading FreeSurfer surface files.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FreeSurferSurfaceError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The `.annot` file magic bytes are wrong.
    #[error("invalid .annot magic: expected -2, got {got}")]
    InvalidMagic {
        /// The magic value that was read.
        got: i32,
    },

    /// A label table entry did not parse correctly.
    #[error("malformed label table entry at index {index}: {reason}")]
    MalformedLabelTable {
        /// Entry index.
        index: usize,
        /// Description of the parse failure.
        reason: String,
    },

    /// Vertex count is unreasonable.
    #[error("invalid vertex count {count}")]
    InvalidVertexCount {
        /// The vertex count read from the file.
        count: i32,
    },
}

// ── FreeSurfer colour lookup table (LUT) ─────────────────────────────────

/// Read the FreeSurfer `FreeSurferColorLUT.txt` file.
///
/// Returns `Vec<(label_id, region_name)>` suitable for feeding into
/// [`crate::Parcellation::new`] as `region_names`.
///
/// The LUT format is a plain-text table where most lines have the form
/// `label_id  region_name  R  G  B  A`.  Lines starting with `#` are
/// comments, blank lines are skipped, and all other leading/trailing
/// whitespace is stripped.
///
/// # Errors
///
/// Returns [`FreeSurferSurfaceError::Io`] on read failure.
pub fn read_freesurfer_lut(
    reader: impl Read,
) -> Result<Vec<(u32, String)>, FreeSurferSurfaceError> {
    let mut entries = Vec::new();
    for line in BufReader::new(reader).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Expected format: "<label_id> <name> <R> <G> <B> <A>"
        let mut parts = trimmed.split_whitespace();
        let label_str = parts.next().unwrap_or("");
        let name = parts.next().unwrap_or("");

        if let Ok(label) = label_str.parse::<u32>() {
            entries.push((label, name.to_string()));
        }
        // Non‑numeric first token is a header or version line — skip.
    }
    Ok(entries)
}

// ── FreeSurfer annotation (`.annot`) ────────────────────────────────────

/// Magic value identifying a FreeSurfer `.annot` file.
const ANNOT_MAGIC: i32 = -2;

/// Per-vertex region labels read from a FreeSurfer `.annot` file.
#[derive(Debug, Clone)]
pub struct SurfaceAnnotation {
    /// Number of surface vertices.
    pub vertex_count: u32,
    /// Label table: `(label_id, region_name)` pairs, ordered as they
    /// appear in the file (table index 0 is typically the unknown /
    /// background region).
    pub label_table: Vec<(u32, String)>,
    /// Per-vertex label IDs (same length as `vertex_count`).
    pub vertex_labels: Box<[u32]>,
}

impl SurfaceAnnotation {
    /// Read a FreeSurfer `.annot` file.
    ///
    /// The binary format is:
    ///
    /// * Magic number `ANNOT_MAGIC` (i32 LE)
    /// * Number of vertices (i32 LE)
    /// * Number of entries in the label table (i32 LE), then for each:
    ///   - structure index (i32 LE, always >= 0)
    ///   - length of the null-terminated structure name (i32 LE)
    ///   - structure name (bytes, null-terminated)
    ///   - R, G, B, A components (i32 LE × 4)
    /// * Per-vertex label indices (i32 LE × vertex_count)
    /// * Additional colour table entries (same format as label table, may
    ///   be larger)
    /// * Per-vertex colour values (i32 LE × vertex_count, often 0)
    ///
    /// # Errors
    ///
    /// Returns [`FreeSurferSurfaceError`] for invalid magic, unreasonable
    /// vertex counts, premature EOF, or malformed label table entries.
    pub fn read(mut reader: impl Read) -> Result<Self, FreeSurferSurfaceError> {
        let mut buf = [0u8; 4];

        // ── Magic ──────────────────────────────────────────────────────
        read_i32(&mut buf, &mut reader)?;
        let magic = i32::from_le_bytes(buf);
        if magic != ANNOT_MAGIC {
            return Err(FreeSurferSurfaceError::InvalidMagic { got: magic });
        }

        // ── Vertex count ───────────────────────────────────────────────
        read_i32(&mut buf, &mut reader)?;
        let vertex_count = i32::from_le_bytes(buf);
        if !(0..=1_000_000).contains(&vertex_count) {
            return Err(FreeSurferSurfaceError::InvalidVertexCount {
                count: vertex_count,
            });
        }
        let n_vertices = vertex_count as usize;

        // ── Label table ────────────────────────────────────────────────
        read_i32(&mut buf, &mut reader)?;
        let n_table_entries = i32::from_le_bytes(buf);
        if n_table_entries < 0 || n_table_entries > vertex_count {
            return Err(FreeSurferSurfaceError::InvalidVertexCount {
                count: n_table_entries,
            });
        }

        let mut label_table: Vec<(u32, String)> = Vec::with_capacity(n_table_entries as usize);

        let mut name_buf = Vec::new();
        for i in 0..n_table_entries as usize {
            read_i32(&mut buf, &mut reader)?;
            let _structure_idx = i32::from_le_bytes(buf);

            read_i32(&mut buf, &mut reader)?;
            let name_len = i32::from_le_bytes(buf);
            if !(0..=4096).contains(&name_len) {
                return Err(FreeSurferSurfaceError::MalformedLabelTable {
                    index: i,
                    reason: format!("name length {name_len} out of range"),
                });
            }

            name_buf.clear();
            name_buf.resize(name_len as usize, 0u8);
            reader.read_exact(&mut name_buf)?;

            // Trim trailing null.
            let name = String::from_utf8_lossy(name_buf.strip_suffix(&[0]).unwrap_or(&name_buf))
                .to_string();

            // Read RGBA (4 × i32 LE).  We don't use the colour values
            // directly; just consume them.
            let mut rgba = [0u8; 16];
            reader.read_exact(&mut rgba)?;

            // The structure index carries the canonical FreeSurfer label
            // when positive; when zero (conventionally the first "Unknown"
            // entry) fall back to the table index.
            let label = if _structure_idx > 0 {
                _structure_idx as u32
            } else {
                i as u32
            };

            label_table.push((label, name));
        }

        // ── Per-vertex labels ──────────────────────────────────────────
        let mut vertex_labels: Vec<u32> = Vec::with_capacity(n_vertices);
        for _ in 0..n_vertices {
            read_i32(&mut buf, &mut reader)?;
            let label_idx = i32::from_le_bytes(buf);
            let label = if label_idx >= 0 && (label_idx as usize) < label_table.len() {
                label_table[label_idx as usize].0
            } else {
                0u32 // out-of-range → background
            };
            vertex_labels.push(label);
        }

        // ── Colour table (skip — same layout) ──────────────────────────
        read_i32(&mut buf, &mut reader)?;
        let n_ctab_entries = i32::from_le_bytes(buf);
        if n_ctab_entries >= 0 && n_ctab_entries <= vertex_count {
            for _ in 0..n_ctab_entries {
                read_i32(&mut buf, &mut reader)?; // structure idx
                read_i32(&mut buf, &mut reader)?; // name len
                let nlen = i32::from_le_bytes(buf);
                if nlen > 0 && nlen < 4096 {
                    let mut skip = vec![0u8; nlen as usize];
                    reader.read_exact(&mut skip)?;
                }
                let mut rgba = [0u8; 16]; // RGBA
                reader.read_exact(&mut rgba)?;
            }
        }

        // ── Per-vertex colours (skip — one i32 per vertex) ────────────
        for _ in 0..n_vertices {
            let mut color = [0u8; 4];
            reader.read_exact(&mut color)?;
        }

        Ok(Self {
            vertex_count: vertex_count as u32,
            label_table,
            vertex_labels: vertex_labels.into_boxed_slice(),
        })
    }
}

/// Read exactly 4 bytes into `buf`, converting I/O errors.
fn read_i32(buf: &mut [u8; 4], reader: &mut impl Read) -> Result<(), FreeSurferSurfaceError> {
    reader.read_exact(buf)?;
    Ok(())
}

#[cfg(test)]
mod tests;
