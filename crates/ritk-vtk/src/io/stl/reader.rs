//! STL reader → VtkPolyData.
//!
//! # Format detection
//! Binary STL is detected by checking whether the file size satisfies the
//! invariant: `file_len == n_triangles × 50 + 84`, where `n_triangles` is the
//! `u32 LE` value at bytes `[80..84]`.  If the invariant holds, binary parsing
//! is used; otherwise, ASCII parsing is attempted.
//!
//! # Output layout
//! STL has no shared-vertex topology.  Each facet contributes three dedicated
//! point entries (3 N points for N triangles).  Facet normals are stored in
//! `cell_data["Normals"]` as `AttributeArray::Normals`.

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Read an STL file (ASCII or binary) and return a [`VtkPolyData`].
pub fn read_stl_mesh(path: impl AsRef<Path>) -> Result<VtkPolyData> {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).with_context(|| format!("reading STL file {}", path.display()))?;
    parse_stl(&bytes)
}

/// Parse STL from a byte slice.  Exposed for in-memory testing.
pub(crate) fn parse_stl(bytes: &[u8]) -> Result<VtkPolyData> {
    if is_binary_stl(bytes) {
        parse_stl_binary(bytes)
    } else {
        parse_stl_ascii(bytes)
    }
}

/// Returns `true` when the byte slice satisfies the binary STL size invariant.
///
/// Invariant: `bytes.len() == n_tri × 50 + 84` where `n_tri` is the LE `u32`
/// stored at `bytes[80..84]`.
fn is_binary_stl(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    bytes.len() == n * 50 + 84
}

// ── ASCII parser ──────────────────────────────────────────────────────────────

fn parse_stl_ascii(bytes: &[u8]) -> Result<VtkPolyData> {
    let text = std::str::from_utf8(bytes).context("STL ASCII file is not valid UTF-8")?;
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut polygons: Vec<Vec<u32>> = Vec::new();
    let mut cell_normals: Vec<[f32; 3]> = Vec::new();

    let mut current_normal: Option<[f32; 3]> = None;
    let mut current_verts: Vec<[f32; 3]> = Vec::new();
    let mut in_loop = false;

    for (line_idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let low = line.to_ascii_lowercase();

        if low.starts_with("facet normal") {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() < 5 {
                bail!("line {}: malformed 'facet normal'", line_idx + 1);
            }
            let nx: f32 = toks[2]
                .parse()
                .with_context(|| format!("line {}: bad normal x", line_idx + 1))?;
            let ny: f32 = toks[3]
                .parse()
                .with_context(|| format!("line {}: bad normal y", line_idx + 1))?;
            let nz: f32 = toks[4]
                .parse()
                .with_context(|| format!("line {}: bad normal z", line_idx + 1))?;
            current_normal = Some([nx, ny, nz]);
            current_verts.clear();
        } else if low.starts_with("outer loop") {
            in_loop = true;
        } else if low.starts_with("vertex") && in_loop {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() < 4 {
                bail!("line {}: malformed vertex", line_idx + 1);
            }
            let x: f32 = toks[1]
                .parse()
                .with_context(|| format!("line {}: bad vertex x", line_idx + 1))?;
            let y: f32 = toks[2]
                .parse()
                .with_context(|| format!("line {}: bad vertex y", line_idx + 1))?;
            let z: f32 = toks[3]
                .parse()
                .with_context(|| format!("line {}: bad vertex z", line_idx + 1))?;
            current_verts.push([x, y, z]);
        } else if low.starts_with("endloop") {
            in_loop = false;
        } else if low.starts_with("endfacet") {
            if current_verts.len() != 3 {
                bail!(
                    "facet does not have exactly 3 vertices ({} found)",
                    current_verts.len()
                );
            }
            let base = points.len() as u32;
            for v in &current_verts {
                points.push(*v);
            }
            polygons.push(vec![base, base + 1, base + 2]);
            if let Some(n) = current_normal {
                cell_normals.push(n);
            }
            current_normal = None;
            current_verts.clear();
        }
        // "solid …", "endsolid …": silently skipped.
    }

    build_stl_poly(points, polygons, cell_normals)
}

// ── Binary parser ─────────────────────────────────────────────────────────────

fn parse_stl_binary(bytes: &[u8]) -> Result<VtkPolyData> {
    // bytes[0..80]  : header (ignored)
    // bytes[80..84] : n_triangles as u32 LE
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let required = n * 50 + 84;
    if bytes.len() < required {
        bail!(
            "binary STL truncated: need {} bytes, have {}",
            required,
            bytes.len()
        );
    }

    let mut points = Vec::with_capacity(n * 3);
    let mut polygons = Vec::with_capacity(n);
    let mut cell_normals = Vec::with_capacity(n);

    let mut off = 84usize;
    for _ in 0..n {
        let normal = read_f32x3_le(bytes, off);
        off += 12;
        let base = points.len() as u32;
        for _ in 0..3 {
            points.push(read_f32x3_le(bytes, off));
            off += 12;
        }
        cell_normals.push(normal);
        polygons.push(vec![base, base + 1, base + 2]);
        off += 2; // attribute byte count (skip)
    }

    build_stl_poly(points, polygons, cell_normals)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn read_f32x3_le(bytes: &[u8], off: usize) -> [f32; 3] {
    let x = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    let y = f32::from_le_bytes([
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ]);
    let z = f32::from_le_bytes([
        bytes[off + 8],
        bytes[off + 9],
        bytes[off + 10],
        bytes[off + 11],
    ]);
    [x, y, z]
}

fn build_stl_poly(
    points: Vec<[f32; 3]>,
    polygons: Vec<Vec<u32>>,
    cell_normals: Vec<[f32; 3]>,
) -> Result<VtkPolyData> {
    let mut poly = VtkPolyData {
        points,
        polygons,
        ..Default::default()
    };
    if !cell_normals.is_empty() {
        poly.cell_data.insert(
            "Normals".to_string(),
            AttributeArray::Normals {
                values: cell_normals,
            },
        );
    }
    Ok(poly)
}
