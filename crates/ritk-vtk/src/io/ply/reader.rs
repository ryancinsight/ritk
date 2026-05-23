//! PLY reader → VtkPolyData.
//!
//! Parses the PLY 1.0 header to determine format and element schemas, then
//! reads vertex and face data in ASCII or binary little-endian encoding.
//!
//! # Supported header subset
//! - `format ascii 1.0` and `format binary_little_endian 1.0`
//!   (big-endian returns `Err`)
//! - `element vertex N` with scalar properties (not list)
//! - `element face M` with `property list <count_type> <index_type> vertex_indices`
//! - Property types: `char` `uchar` `short` `ushort` `int` `uint` `float` `double`
//!   and their sized aliases (`int8` … `float64`).
//! - `comment` and `obj_info` lines: skipped.

use super::types::{PlyFormat, PlyHeader, PlyType};
use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{bail, Context, Result};
use std::path::Path;

// ── Public interface ──────────────────────────────────────────────────────────

/// Read a PLY file (ASCII or binary little-endian) → [`VtkPolyData`].
pub fn read_ply_mesh(path: impl AsRef<Path>) -> Result<VtkPolyData> {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).with_context(|| format!("reading PLY file {}", path.display()))?;
    parse_ply(&bytes)
}

/// Parse PLY from a byte slice.  Exposed for in-memory testing.
pub(crate) fn parse_ply(bytes: &[u8]) -> Result<VtkPolyData> {
    let data_off = find_data_offset(bytes).context("PLY file missing 'end_header'")?;
    let header_text =
        std::str::from_utf8(&bytes[..data_off]).context("PLY header is not valid UTF-8")?;
    let hdr = parse_header(header_text)?;

    match hdr.format {
        PlyFormat::Ascii => parse_ascii_body(
            std::str::from_utf8(&bytes[data_off..]).context("PLY ASCII body is not UTF-8")?,
            &hdr,
        ),
        PlyFormat::BinaryLe => parse_binary_le_body(&bytes[data_off..], &hdr),
        PlyFormat::BinaryBe => bail!("big-endian PLY not supported for read"),
    }
}

// ── Header parser ─────────────────────────────────────────────────────────────

fn find_data_offset(bytes: &[u8]) -> Option<usize> {
    const MARKER: &[u8] = b"end_header";
    bytes
        .windows(MARKER.len())
        .position(|w| w == MARKER)
        .map(|pos| {
            let after = pos + MARKER.len();
            if after < bytes.len()
                && bytes[after] == b'\r'
                && after + 1 < bytes.len()
                && bytes[after + 1] == b'\n'
            {
                after + 2
            } else if after < bytes.len() && bytes[after] == b'\n' {
                after + 1
            } else {
                after
            }
        })
}

fn parse_header(text: &str) -> Result<PlyHeader> {
    let mut lines = text.lines();
    let first = lines.next().context("PLY file is empty")?.trim();
    if first != "ply" {
        bail!("not a PLY file (first line: '{}')", first);
    }

    let mut format = PlyFormat::Ascii;
    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut vertex_props: Vec<(String, PlyType)> = Vec::new();
    let mut face_count_type = PlyType::Uchar;
    let mut face_index_type = PlyType::Int;

    #[derive(PartialEq)]
    enum Cur {
        None,
        Vertex,
        Face,
        Other,
    }
    let mut cur = Cur::None;

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with("comment") || line.starts_with("obj_info") {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }

        match toks[0] {
            "format" if toks.len() >= 2 => {
                format = match toks[1] {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::BinaryLe,
                    "binary_big_endian" => PlyFormat::BinaryBe,
                    other => bail!("unknown PLY format '{}'", other),
                };
            }
            "element" if toks.len() >= 3 => {
                let count: usize = toks[2]
                    .parse()
                    .with_context(|| format!("bad element count '{}'", toks[2]))?;
                cur = match toks[1] {
                    "vertex" => {
                        vertex_count = count;
                        Cur::Vertex
                    }
                    "face" => {
                        face_count = count;
                        Cur::Face
                    }
                    _ => Cur::Other,
                };
            }
            "property" => match cur {
                Cur::Vertex if toks.len() >= 3 && toks[1] != "list" => {
                    vertex_props.push((toks[2].to_string(), PlyType::from_str(toks[1])?));
                }
                Cur::Face if toks.len() >= 5 && toks[1] == "list" => {
                    face_count_type = PlyType::from_str(toks[2])?;
                    face_index_type = PlyType::from_str(toks[3])?;
                }
                _ => {} // skip list-on-vertex and other unknowns
            },
            "end_header" => break,
            _ => {}
        }
    }

    if vertex_count > 0 {
        let has_xyz = ["x", "y", "z"]
            .iter()
            .all(|n| vertex_props.iter().any(|(p, _)| p == n));
        if !has_xyz {
            bail!("PLY vertex element is missing x, y, or z property");
        }
    }

    Ok(PlyHeader {
        format,
        vertex_count,
        face_count,
        vertex_props,
        face_count_type,
        face_index_type,
    })
}

// ── ASCII body reader ─────────────────────────────────────────────────────────

fn parse_ascii_body(text: &str, hdr: &PlyHeader) -> Result<VtkPolyData> {
    let mut lines = text.lines();
    let xi = hdr.find_prop("x").context("no x")?;
    let yi = hdr.find_prop("y").context("no y")?;
    let zi = hdr.find_prop("z").context("no z")?;
    let has_n = hdr.has_normals();
    let (nxi, nyi, nzi) = if has_n {
        (
            hdr.find_prop("nx").unwrap(),
            hdr.find_prop("ny").unwrap(),
            hdr.find_prop("nz").unwrap(),
        )
    } else {
        (0, 0, 0)
    };

    let mut points = Vec::with_capacity(hdr.vertex_count);
    let mut normals: Option<Vec<[f32; 3]>> = if has_n {
        Some(Vec::with_capacity(hdr.vertex_count))
    } else {
        None
    };

    for v in 0..hdr.vertex_count {
        let line = lines
            .next()
            .with_context(|| format!("truncated vertex data at vertex {v}"))?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        let x = hdr
            .prop_type(xi)
            .parse_as_f32(toks.get(xi).context("x tok")?)?;
        let y = hdr
            .prop_type(yi)
            .parse_as_f32(toks.get(yi).context("y tok")?)?;
        let z = hdr
            .prop_type(zi)
            .parse_as_f32(toks.get(zi).context("z tok")?)?;
        points.push([x, y, z]);
        if let Some(ref mut ns) = normals {
            let nx = hdr
                .prop_type(nxi)
                .parse_as_f32(toks.get(nxi).context("nx tok")?)?;
            let ny = hdr
                .prop_type(nyi)
                .parse_as_f32(toks.get(nyi).context("ny tok")?)?;
            let nz = hdr
                .prop_type(nzi)
                .parse_as_f32(toks.get(nzi).context("nz tok")?)?;
            ns.push([nx, ny, nz]);
        }
    }

    let mut polygons = Vec::with_capacity(hdr.face_count);
    for f in 0..hdr.face_count {
        let line = lines
            .next()
            .with_context(|| format!("truncated face data at face {f}"))?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        let cnt = hdr
            .face_count_type
            .parse_as_u32(toks.first().context("empty face line")?)? as usize;
        if toks.len() < cnt + 1 {
            bail!("face {f}: expected {cnt} indices, got {}", toks.len() - 1);
        }
        let indices = (1..=cnt)
            .map(|i| hdr.face_index_type.parse_as_u32(toks[i]))
            .collect::<Result<Vec<u32>>>()?;
        polygons.push(indices);
    }

    build_ply_poly(points, polygons, normals)
}

// ── Binary LE body reader ─────────────────────────────────────────────────────

fn parse_binary_le_body(body: &[u8], hdr: &PlyHeader) -> Result<VtkPolyData> {
    let xi = hdr.find_prop("x").context("no x")?;
    let yi = hdr.find_prop("y").context("no y")?;
    let zi = hdr.find_prop("z").context("no z")?;
    let has_n = hdr.has_normals();
    let (nxi, nyi, nzi) = if has_n {
        (
            hdr.find_prop("nx").unwrap(),
            hdr.find_prop("ny").unwrap(),
            hdr.find_prop("nz").unwrap(),
        )
    } else {
        (0, 0, 0)
    };

    let vert_sz = hdr.vertex_byte_size();
    let mut off = 0usize;
    let mut points = Vec::with_capacity(hdr.vertex_count);
    let mut normals: Option<Vec<[f32; 3]>> = if has_n {
        Some(Vec::with_capacity(hdr.vertex_count))
    } else {
        None
    };

    for v in 0..hdr.vertex_count {
        if off + vert_sz > body.len() {
            bail!("binary PLY vertex data truncated at vertex {v}");
        }
        let x = hdr
            .prop_type(xi)
            .read_le_f32(body, off + hdr.prop_byte_offset(xi));
        let y = hdr
            .prop_type(yi)
            .read_le_f32(body, off + hdr.prop_byte_offset(yi));
        let z = hdr
            .prop_type(zi)
            .read_le_f32(body, off + hdr.prop_byte_offset(zi));
        points.push([x, y, z]);
        if let Some(ref mut ns) = normals {
            let nx = hdr
                .prop_type(nxi)
                .read_le_f32(body, off + hdr.prop_byte_offset(nxi));
            let ny = hdr
                .prop_type(nyi)
                .read_le_f32(body, off + hdr.prop_byte_offset(nyi));
            let nz = hdr
                .prop_type(nzi)
                .read_le_f32(body, off + hdr.prop_byte_offset(nzi));
            ns.push([nx, ny, nz]);
        }
        off += vert_sz;
    }

    let cnt_sz = hdr.face_count_type.byte_size();
    let idx_sz = hdr.face_index_type.byte_size();
    let mut polygons = Vec::with_capacity(hdr.face_count);

    for f in 0..hdr.face_count {
        if off + cnt_sz > body.len() {
            bail!("binary PLY face data truncated at face {f}");
        }
        let cnt = hdr.face_count_type.read_le_u32(body, off) as usize;
        off += cnt_sz;
        if off + cnt * idx_sz > body.len() {
            bail!("binary PLY face {f} index data truncated (need {cnt} indices)");
        }
        let mut indices = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            indices.push(hdr.face_index_type.read_le_u32(body, off));
            off += idx_sz;
        }
        polygons.push(indices);
    }

    build_ply_poly(points, polygons, normals)
}

// ── Shared builder ────────────────────────────────────────────────────────────

fn build_ply_poly(
    points: Vec<[f32; 3]>,
    polygons: Vec<Vec<u32>>,
    normals: Option<Vec<[f32; 3]>>,
) -> Result<VtkPolyData> {
    let mut poly = VtkPolyData {
        points,
        polygons,
        ..Default::default()
    };
    if let Some(values) = normals {
        poly.point_data
            .insert("Normals".to_string(), AttributeArray::Normals { values });
    }
    Ok(poly)
}
