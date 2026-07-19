//! VTU reader: `read_vtu_unstructured_grid`, `parse_vtu`.

use super::xml_helpers::{
    attr_usize, extract_da_content, find_section, find_tag, named_da, parse_attrs, parse_floats,
    parse_ints,
};
use crate::domain::vtk_data_object::{VtkCellType, VtkUnstructuredGrid};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Read a VTU XML (ASCII inline) file from disk into a [`VtkUnstructuredGrid`].
pub fn read_vtu_unstructured_grid<P: AsRef<Path>>(path: P) -> Result<VtkUnstructuredGrid> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTU: {}", path.as_ref().display()))?;
    parse_vtu(&s)
}

/// Parse an ASCII-inline VTU XML string into a [`VtkUnstructuredGrid`].
pub(crate) fn parse_vtu(input: &str) -> Result<VtkUnstructuredGrid> {
    // â”€â”€ Piece header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let piece = find_tag(input, "Piece")
        .ok_or_else(|| anyhow::anyhow!("missing <Piece> tag in VTU document"))?;
    let n_points: usize = attr_usize(&piece, "NumberOfPoints")?;
    let n_cells: usize = attr_usize(&piece, "NumberOfCells")?;

    // â”€â”€ Points â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let points_sec =
        find_section(input, "Points").ok_or_else(|| anyhow::anyhow!("missing <Points> section"))?;
    let coords = parse_floats(&extract_da_content(&points_sec));
    if coords.len() != n_points * 3 {
        bail!(
            "expected {} coord values for {} points, got {}",
            n_points * 3,
            n_points,
            coords.len()
        );
    }
    let points: Vec<[f32; 3]> = coords.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    // â”€â”€ Cells â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let cells_sec =
        find_section(input, "Cells").ok_or_else(|| anyhow::anyhow!("missing <Cells> section"))?;

    let conn_da = named_da(&cells_sec, "connectivity")
        .ok_or_else(|| anyhow::anyhow!("missing connectivity DataArray in <Cells>"))?;
    let offs_da = named_da(&cells_sec, "offsets")
        .ok_or_else(|| anyhow::anyhow!("missing offsets DataArray in <Cells>"))?;
    let types_da = named_da(&cells_sec, "types")
        .ok_or_else(|| anyhow::anyhow!("missing types DataArray in <Cells>"))?;

    let connectivity =
        parse_non_negative_u32_values(&extract_da_content(&conn_da), "connectivity")?;
    let offsets = parse_non_negative_usize_values(&extract_da_content(&offs_da), "offsets")?;
    let type_codes: Vec<i32> = parse_ints(&extract_da_content(&types_da));

    if offsets.len() != n_cells {
        bail!(
            "offsets count {} != NumberOfCells {}",
            offsets.len(),
            n_cells
        );
    }
    if type_codes.len() != n_cells {
        bail!(
            "types count {} != NumberOfCells {}",
            type_codes.len(),
            n_cells
        );
    }

    // Reconstruct cells: cell i = connectivity[offsets[i-1]..offsets[i]].
    let mut cells: Vec<Vec<u32>> = Vec::with_capacity(n_cells);
    let mut prev: usize = 0;
    for (cell_index, &off) in offsets.iter().enumerate() {
        if off < prev {
            bail!(
                "offset {} at cell {} is less than previous offset {}",
                off,
                cell_index,
                prev
            );
        }
        if off > connectivity.len() {
            bail!(
                "offset {} exceeds connectivity length {}",
                off,
                connectivity.len()
            );
        }
        cells.push(connectivity[prev..off].to_vec());
        prev = off;
    }
    if prev != connectivity.len() {
        bail!(
            "final offset {} does not consume connectivity length {}",
            prev,
            connectivity.len()
        );
    }

    let cell_types: Vec<VtkCellType> = type_codes
        .iter()
        .map(|&v| {
            let code = u8::try_from(v).with_context(|| {
                format!("types value {v} must be a non-negative UInt8 cell type code")
            })?;
            Ok(VtkCellType::try_from(code).unwrap_or_else(|_| {
                tracing::warn!(
                    code = v,
                    "unknown VTK cell type code in VTU; mapped to Vertex"
                );
                VtkCellType::Vertex
            }))
        })
        .collect::<Result<_>>()?;

    // â”€â”€ Attribute sections â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let point_data = find_section(input, "PointData")
        .map(|sec| parse_attrs(&sec))
        .unwrap_or_default();
    let cell_data = find_section(input, "CellData")
        .map(|sec| parse_attrs(&sec))
        .unwrap_or_default();

    let grid = VtkUnstructuredGrid {
        points,
        cells,
        cell_types,
        point_data,
        cell_data,
    };
    grid.validate().map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(grid)
}

fn parse_non_negative_u32_values(content: &str, field: &str) -> Result<Vec<u32>> {
    parse_ints(content)
        .into_iter()
        .map(|value| {
            u32::try_from(value)
                .with_context(|| format!("{field} value {value} must be non-negative"))
        })
        .collect()
}

fn parse_non_negative_usize_values(content: &str, field: &str) -> Result<Vec<usize>> {
    parse_ints(content)
        .into_iter()
        .map(|value| {
            usize::try_from(value)
                .with_context(|| format!("{field} value {value} must be non-negative"))
        })
        .collect()
}
