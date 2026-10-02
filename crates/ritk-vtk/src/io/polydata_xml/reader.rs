//! VTK XML PolyData (.vtp) reader (ASCII inline format).

use crate::domain::vtk_data_object::VtkPolyData;
use crate::io::xml_helpers::{
    attr_usize, find_section, find_tag, first_array_values, index_values, named_da, parse_attrs,
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
    let coords = first_array_values(&points_sec).context("<Points> coordinates")?;
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
        vertices: parse_cells(input, "Verts")?,
        lines: parse_cells(input, "Lines")?,
        polygons: parse_cells(input, "Polys")?,
        triangle_strips: parse_cells(input, "Strips")?,
        point_data: find_section(input, "PointData")
            .map(|sec| parse_attrs(&sec))
            .transpose()?
            .unwrap_or_default(),
        cell_data: find_section(input, "CellData")
            .map(|sec| parse_attrs(&sec))
            .transpose()?
            .unwrap_or_default(),
    };
    Ok(poly)
}

fn parse_cells(input: &str, sname: &str) -> Result<Vec<Vec<u32>>> {
    let Some(sec) = find_section(input, sname) else {
        return Ok(vec![]);
    };
    let (Some(conn_da), Some(offs_da)) =
        (named_da(&sec, "connectivity"), named_da(&sec, "offsets"))
    else {
        return Ok(vec![]);
    };
    let conn = index_values(&conn_da, "connectivity")?;
    let offs = index_values(&offs_da, "offsets")?;
    if offs.is_empty() {
        return Ok(vec![]);
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
    Ok(cells)
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_reader.rs"]
mod tests;
