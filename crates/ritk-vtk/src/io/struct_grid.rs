//! VTK legacy STRUCTURED_GRID reader/writer.
//!
//! Structured grid G = (D, V, A_P, A_C):
//! - D = (nx, ny, nz), |V| = nx*ny*nz
//! - A_P: per-point attributes, length n_points * ncomp
//! - A_C: per-cell attributes, length n_cells * ncomp
//!
//! Reference: VTK File Formats (legacy) section 4.3, Kitware Inc.

use crate::domain::vtk_data_object::{AttributeArray, VtkStructuredGrid};
use crate::io::legacy_write_attribute::write_attribute_legacy;
use crate::io::read_helpers::{read_ascii, read_binary_be, read_line};
use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Read a VTK legacy STRUCTURED_GRID file from disk.
pub fn read_vtk_structured_grid<P: AsRef<Path>>(path: P) -> Result<VtkStructuredGrid> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open VTK structured grid: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    parse_structured_grid(&mut reader)
}

/// Write a VtkStructuredGrid to a VTK legacy ASCII file.
/// Returns Err immediately if grid.validate() fails.
pub fn write_vtk_structured_grid<P: AsRef<Path>>(path: P, grid: &VtkStructuredGrid) -> Result<()> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;
    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create VTK structured grid: {}", path.display()))?;
    let mut w = BufWriter::new(file);
    write_structured_grid(&mut w, grid)
}

fn parse_structured_grid(reader: &mut dyn BufRead) -> Result<VtkStructuredGrid> {
    let line1 = read_line(reader)?.with_context(|| "EOF before version line")?;
    if !line1.starts_with("# vtk DataFile Version") {
        bail!("not a VTK legacy file: {}", line1);
    }
    let _desc = read_line(reader)?.with_context(|| "EOF before description line")?;
    let enc_line = read_line(reader)?.with_context(|| "EOF before encoding line")?;
    let binary = match enc_line.to_ascii_uppercase().trim() {
        "ASCII" => false,
        "BINARY" => true,
        other => bail!("unsupported VTK encoding: {}", other) };
    let ds_line = read_line(reader)?.with_context(|| "EOF before DATASET line")?;
    if !ds_line.to_ascii_uppercase().contains("STRUCTURED_GRID") {
        bail!("expected DATASET STRUCTURED_GRID, got: {}", ds_line);
    }

    let mut grid = VtkStructuredGrid::default();
    let mut in_point_data = false;
    let mut pd_n: usize = 0;
    let mut cd_n: usize = 0;
    let mut pending_scalars: Option<(String, usize, bool)> = None;
    let mut pending_vectors: Option<(String, bool)> = None;
    let mut pending_normals: Option<(String, bool)> = None;

    loop {
        if let Some((name, ncomp, is_pd)) = pending_scalars.take() {
            let lt = read_line(reader)?.with_context(|| "EOF waiting for LOOKUP_TABLE")?;
            if !lt.to_ascii_uppercase().starts_with("LOOKUP_TABLE") {
                bail!("expected LOOKUP_TABLE after SCALARS, got: {}", lt);
            }
            let n = if is_pd { pd_n } else { cd_n };
            let total = n * ncomp;
            let values = if binary {
                read_binary_be::<f32>(reader, total, "f32")?
            } else {
                read_ascii::<f32>(reader, total, "f32")?
            };
            let arr = AttributeArray::Scalars {
                values,
                num_components: ncomp };
            if is_pd {
                grid.point_data.insert(name, arr);
            } else {
                grid.cell_data.insert(name, arr);
            }
            continue;
        }
        if let Some((name, is_pd)) = pending_vectors.take() {
            let n = if is_pd { pd_n } else { cd_n };
            let flat = if binary {
                read_binary_be::<f32>(reader, n * 3, "f32")?
            } else {
                read_ascii::<f32>(reader, n * 3, "f32")?
            };
            let values: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            let arr = AttributeArray::Vectors { values };
            if is_pd {
                grid.point_data.insert(name, arr);
            } else {
                grid.cell_data.insert(name, arr);
            }
            continue;
        }
        if let Some((name, is_pd)) = pending_normals.take() {
            let n = if is_pd { pd_n } else { cd_n };
            let flat = if binary {
                read_binary_be::<f32>(reader, n * 3, "f32")?
            } else {
                read_ascii::<f32>(reader, n * 3, "f32")?
            };
            let values: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            let arr = AttributeArray::Normals { values };
            if is_pd {
                grid.point_data.insert(name, arr);
            } else {
                grid.cell_data.insert(name, arr);
            }
            continue;
        }

        let line = match read_line(reader)? {
            Some(l) => l,
            None => break };
        let upper = line.to_ascii_uppercase();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        if upper.starts_with("DIMENSIONS") {
            if tokens.len() < 4 {
                bail!("DIMENSIONS malformed: {}", line);
            }
            let nx: usize = tokens[1].parse().with_context(|| "bad DIMENSIONS nx")?;
            let ny: usize = tokens[2].parse().with_context(|| "bad DIMENSIONS ny")?;
            let nz: usize = tokens[3].parse().with_context(|| "bad DIMENSIONS nz")?;
            grid.dimensions = [nx, ny, nz];
        } else if upper.starts_with("POINTS") {
            if tokens.len() < 2 {
                bail!("POINTS malformed: {}", line);
            }
            let n: usize = tokens[1].parse().with_context(|| "bad POINTS count")?;
            let is_double = tokens
                .get(2)
                .map(|s| s.eq_ignore_ascii_case("double"))
                .unwrap_or(false);
            grid.points = if binary {
                if is_double {
                    let raw = read_binary_be::<f64>(reader, n * 3, "f64")?;
                    raw.chunks_exact(3)
                        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                        .collect()
                } else {
                    let raw = read_binary_be::<f32>(reader, n * 3, "f32")?;
                    raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
                }
            } else {
                let raw = read_ascii::<f32>(reader, n * 3, "f32")?;
                raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
            };
        } else if upper.starts_with("POINT_DATA") {
            if tokens.len() < 2 {
                bail!("POINT_DATA malformed: {}", line);
            }
            pd_n = tokens[1].parse().with_context(|| "bad POINT_DATA count")?;
            in_point_data = true;
        } else if upper.starts_with("CELL_DATA") {
            if tokens.len() < 2 {
                bail!("CELL_DATA malformed: {}", line);
            }
            cd_n = tokens[1].parse().with_context(|| "bad CELL_DATA count")?;
            in_point_data = false;
        } else if upper.starts_with("SCALARS") {
            if tokens.len() < 3 {
                bail!("SCALARS malformed: {}", line);
            }
            let name = tokens[1].to_string();
            let ncomp: usize = tokens.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            pending_scalars = Some((name, ncomp, in_point_data));
        } else if upper.starts_with("VECTORS") {
            if tokens.len() < 2 {
                bail!("VECTORS malformed: {}", line);
            }
            pending_vectors = Some((tokens[1].to_string(), in_point_data));
        } else if upper.starts_with("NORMALS") {
            if tokens.len() < 2 {
                bail!("NORMALS malformed: {}", line);
            }
            pending_normals = Some((tokens[1].to_string(), in_point_data));
        } else if upper.starts_with("LOOKUP_TABLE") {
            // Standalone LOOKUP_TABLE outside SCALARS context: skip.
        }
        // Unknown keywords silently skipped for forward compatibility.
    }

    grid.validate().map_err(|e| anyhow!("{}", e))?;
    Ok(grid)
}

fn write_structured_grid(w: &mut dyn Write, grid: &VtkStructuredGrid) -> Result<()> {
    let [nx, ny, nz] = grid.dimensions;
    let np = grid.n_points();
    writeln!(w, "# vtk DataFile Version 3.0")?;
    writeln!(w, "RITK exported structured grid")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET STRUCTURED_GRID")?;
    writeln!(w, "DIMENSIONS {} {} {}", nx, ny, nz)?;
    writeln!(w, "POINTS {} float", np)?;
    for [x, y, z] in &grid.points {
        writeln!(w, "{} {} {}", x, y, z)?;
    }
    if !grid.point_data.is_empty() {
        writeln!(w, "POINT_DATA {}", np)?;
        for (name, attr) in &grid.point_data {
            write_attribute_legacy(w, name, attr)?;
        }
    }
    let nc = grid.n_cells();
    if !grid.cell_data.is_empty() {
        writeln!(w, "CELL_DATA {}", nc)?;
        for (name, attr) in &grid.cell_data {
            write_attribute_legacy(w, name, attr)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "struct_grid/tests.rs"]
mod tests;
