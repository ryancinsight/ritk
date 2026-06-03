//! OBJ ASCII reader → VtkPolyData.
//!
//! # Specification
//! - `v x y z`  → `points`
//! - `vn nx ny nz` → accumulated normal list; per-face normal index references
//!   are mapped to point positions (last-seen reference wins per vertex).
//! - `f v1 v2 v3 …` (variants: `v`, `v/t`, `v/t/n`, `v//n`) → `polygons`
//!   (OBJ indices are 1-based; stored as 0-based `u32`).
//! - `vt`, `o`, `g`, `usemtl`, `mtllib`, `s`, `l`: silently skipped.
//! - `#` comment lines: skipped.
//! - Unknown directives: skipped (forward-compatible).
//! - Returns `Err` on malformed `v`, `vn`, or `f` values.

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read an OBJ file from `path` and return a [`VtkPolyData`].
pub fn read_obj_mesh(path: impl AsRef<Path>) -> Result<VtkPolyData> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening OBJ file {}", path.display()))?;
    parse_obj(BufReader::new(file))
}

/// Parse OBJ from any [`BufRead`] source.  Exposed for in-memory testing.
pub(crate) fn parse_obj(reader: impl BufRead) -> Result<VtkPolyData> {
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut normals_raw: Vec<[f32; 3]> = Vec::new();
    let mut polygons: Vec<Vec<u32>> = Vec::new();
    // Slot per point: normal assigned via the last face that references it.
    let mut point_normals: Vec<Option<[f32; 3]>> = Vec::new();
    let mut has_face_normals = false;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("I/O error on line {}", line_idx + 1))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split directive from the rest of the line on the first whitespace run.
        let mut iter = line.splitn(2, |c: char| c.is_ascii_whitespace());
        let directive = iter.next().unwrap_or("");
        let rest = iter.next().unwrap_or("").trim();

        match directive {
            "v" => {
                let p = parse_vec3(rest).with_context(|| {
                    format!("line {}: malformed vertex '{}'", line_idx + 1, rest)
                })?;
                points.push(p);
                point_normals.push(None);
            }
            "vn" => {
                let n = parse_vec3(rest).with_context(|| {
                    format!("line {}: malformed normal '{}'", line_idx + 1, rest)
                })?;
                normals_raw.push(n);
            }
            "f" => {
                let face = parse_face_line(rest, line_idx + 1)?;
                // Assign normals to point positions; last-seen face reference wins.
                for &(vi, _, ni_opt) in &face {
                    if let Some(ni) = ni_opt {
                        if let Some(&n) = normals_raw.get(ni as usize) {
                            if (vi as usize) < point_normals.len() {
                                point_normals[vi as usize] = Some(n);
                                has_face_normals = true;
                            }
                        }
                    }
                }
                polygons.push(face.into_iter().map(|(vi, _, _)| vi).collect());
            }
            // Recognised but unsupported directives – skip silently.
            "vt" | "o" | "g" | "usemtl" | "mtllib" | "s" | "l" => {}
            _ => {} // Unknown directives ignored for forward compatibility.
        }
    }

    let mut poly = VtkPolyData {
        points,
        polygons,
        ..Default::default()
    };

    if has_face_normals {
        let normals: Vec<[f32; 3]> = point_normals
            .into_iter()
            .map(|n| n.unwrap_or([0.0, 0.0, 0.0]))
            .collect();
        poly.point_data.insert(
            "Normals".to_string(),
            AttributeArray::Normals { values: normals },
        );
    }

    Ok(poly)
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

fn parse_vec3(s: &str) -> Result<[f32; 3]> {
    let mut parts = s.split_whitespace();
    let x: f32 = parts
        .next()
        .context("missing x coordinate")?
        .parse()
        .context("x")?;
    let y: f32 = parts
        .next()
        .context("missing y coordinate")?
        .parse()
        .context("y")?;
    let z: f32 = parts
        .next()
        .context("missing z coordinate")?
        .parse()
        .context("z")?;
    Ok([x, y, z])
}

type ObjFaceVertex = (u32, Option<u32>, Option<u32>);

/// Returns `Vec<(vertex_idx_0based, texcoord_idx_0based_opt, normal_idx_0based_opt)>`.
fn parse_face_line(s: &str, line_no: usize) -> Result<Vec<ObjFaceVertex>> {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.len() < 3 {
        bail!("line {}: face has fewer than 3 vertices", line_no);
    }
    tokens
        .iter()
        .map(|tok| parse_face_vertex(tok, line_no))
        .collect()
}

fn parse_face_vertex(token: &str, line_no: usize) -> Result<(u32, Option<u32>, Option<u32>)> {
    let parts: Vec<&str> = token.split('/').collect();
    let vi = parse_obj_index(parts[0], line_no, "vertex")?;
    let ti = parts
        .get(1)
        .filter(|s| !s.is_empty())
        .map(|s| parse_obj_index(s, line_no, "texcoord"))
        .transpose()?;
    let ni = parts
        .get(2)
        .filter(|s| !s.is_empty())
        .map(|s| parse_obj_index(s, line_no, "normal"))
        .transpose()?;
    Ok((vi, ti, ni))
}

/// Convert a 1-based OBJ index string to a 0-based `u32`.
fn parse_obj_index(s: &str, line_no: usize, kind: &str) -> Result<u32> {
    let idx: i64 = s
        .parse()
        .with_context(|| format!("line {}: invalid {} index '{}'", line_no, kind, s))?;
    if idx < 1 {
        bail!(
            "line {}: {} index {} must be >= 1 (OBJ is 1-based)",
            line_no,
            kind,
            idx
        );
    }
    Ok((idx - 1) as u32)
}
