//! VTK XML PolyData (.vtp) reader (ASCII inline format).

use crate::domain::vtk_data_object::VtkPolyData;
use crate::io::xml_helpers::{
    attr_usize, extract_da_content, find_section, find_tag, named_da, parse_attrs, parse_floats,
    parse_ints,
};
use anyhow::{bail, Context, Result};
use std::path::Path;

pub fn read_vtp_polydata<P: AsRef<Path>>(path: P) -> Result<VtkPolyData> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTP: {}", path.as_ref().display()))?;
    parse_vtp(&s)
}

pub(crate) fn parse_vtp(input: &str) -> Result<VtkPolyData> {
    let piece = find_tag(input, "Piece").ok_or_else(|| anyhow::anyhow!("missing <Piece>"))?;
    let n_points: usize = attr_usize(&piece, "NumberOfPoints")?;

    let points_sec =
        find_section(input, "Points").ok_or_else(|| anyhow::anyhow!("missing <Points>"))?;
    let coords: Vec<f32> = parse_floats(&extract_da_content(&points_sec));
    if coords.len() != n_points * 3 {
        bail!(
            "expected {} coord values, got {}",
            n_points * 3,
            coords.len()
        );
    }
    let points: Vec<[f32; 3]> = coords.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    let poly = VtkPolyData {
        points,
        vertices: parse_cells(input, "Verts"),
        lines: parse_cells(input, "Lines"),
        polygons: parse_cells(input, "Polys"),
        triangle_strips: parse_cells(input, "Strips"),
        point_data: find_section(input, "PointData")
            .map(|sec| parse_attrs(&sec))
            .unwrap_or_default(),
        cell_data: find_section(input, "CellData")
            .map(|sec| parse_attrs(&sec))
            .unwrap_or_default(),
    };
    Ok(poly)
}

fn parse_cells(input: &str, sname: &str) -> Vec<Vec<u32>> {
    let sec = match find_section(input, sname) {
        Some(s) => s,
        None => return vec![],
    };
    let conn_da = match named_da(&sec, "connectivity") {
        Some(s) => s,
        None => return vec![],
    };
    let offs_da = match named_da(&sec, "offsets") {
        Some(s) => s,
        None => return vec![],
    };
    let conn: Vec<u32> = parse_ints(&extract_da_content(&conn_da))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let offs: Vec<u32> = parse_ints(&extract_da_content(&offs_da))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    if offs.is_empty() {
        return vec![];
    }
    let mut cells = Vec::new();
    let mut prev = 0usize;
    for &off in &offs {
        let off = off as usize;
        if off <= conn.len() {
            cells.push(conn[prev..off].to_vec());
        }
        prev = off;
    }
    cells
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_reader.rs"]
mod tests;
