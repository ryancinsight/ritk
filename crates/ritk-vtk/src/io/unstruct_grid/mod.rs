//! VTK legacy UNSTRUCTURED_GRID reader/writer.
//!
//! Unstructured grid G = (V, C, T, A_P, A_C):
//! - V subset R^3, |V| = n_points
//! - C: cells (point index lists), |C| = n_cells
//! - T: VTK cell type codes, |T| = n_cells
//! - Invariants: |C|==|T|, all indices in [0, n_points).
//!
//! Reference: VTK File Formats (legacy) section 4.5, Kitware Inc.

use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
use crate::io::legacy_write_attribute::write_attribute_legacy;
use crate::io::read_helpers::{parse_cells_from_ints, read_ascii, read_binary_be, read_line};
use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Read a VTK legacy UNSTRUCTURED_GRID file from disk.
pub fn read_vtk_unstructured_grid<P: AsRef<Path>>(path: P) -> Result<VtkUnstructuredGrid> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open VTK unstructured grid: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    parse_unstructured_grid(&mut reader)
}

/// Write a VtkUnstructuredGrid to a VTK legacy ASCII file.
/// Returns Err immediately if grid.validate() fails.
pub fn write_vtk_unstructured_grid<P: AsRef<Path>>(
    path: P,
    grid: &VtkUnstructuredGrid,
) -> Result<()> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;
    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create VTK unstructured grid: {}", path.display()))?;
    let mut w = BufWriter::new(file);
    write_unstructured_grid(&mut w, grid)
}

fn parse_unstructured_grid(reader: &mut dyn BufRead) -> Result<VtkUnstructuredGrid> {
    let line1 = read_line(reader)?.with_context(|| "EOF before version line")?;
    if !line1.starts_with("# vtk DataFile Version") {
        bail!("not a VTK legacy file: {}", line1);
    }

    let _desc = read_line(reader)?.with_context(|| "EOF before description")?;

    let enc_line = read_line(reader)?.with_context(|| "EOF before encoding")?;
    let binary = match enc_line.to_ascii_uppercase().trim() {
        "ASCII" => false,
        "BINARY" => true,
        other => bail!("unsupported encoding: {}", other),
    };

    let ds_line = read_line(reader)?.with_context(|| "EOF before DATASET")?;
    if !ds_line.to_ascii_uppercase().contains("UNSTRUCTURED_GRID") {
        bail!("expected DATASET UNSTRUCTURED_GRID, got: {}", ds_line);
    }

    let mut grid = VtkUnstructuredGrid::default();
    let mut in_point_data = false;
    let mut pd_n: usize = 0;
    let mut cd_n: usize = 0;
    let mut psc: Option<(String, usize, bool)> = None;
    let mut pvc: Option<(String, bool)> = None;
    let mut pnc: Option<(String, bool)> = None;

    loop {
        if let Some((nm, nc, ip)) = psc.take() {
            let lt = read_line(reader)?.with_context(|| "EOF for LOOKUP_TABLE")?;
            if !lt.to_ascii_uppercase().starts_with("LOOKUP_TABLE") {
                bail!("expected LOOKUP_TABLE, got: {}", lt);
            }
            let n = if ip { pd_n } else { cd_n };
            let vals = if binary {
                read_binary_be::<f32>(reader, n * nc, "f32")?
            } else {
                read_ascii::<f32>(reader, n * nc, "f32")?
            };
            let arr = AttributeArray::Scalars {
                values: vals,
                num_components: nc,
            };
            if ip {
                grid.point_data.insert(nm, arr);
            } else {
                grid.cell_data.insert(nm, arr);
            }
            continue;
        }

        if let Some((nm, ip)) = pvc.take() {
            let n = if ip { pd_n } else { cd_n };
            let flat = if binary {
                read_binary_be::<f32>(reader, n * 3, "f32")?
            } else {
                read_ascii::<f32>(reader, n * 3, "f32")?
            };
            let v: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            if ip {
                grid.point_data
                    .insert(nm, AttributeArray::Vectors { values: v });
            } else {
                grid.cell_data
                    .insert(nm, AttributeArray::Vectors { values: v });
            }
            continue;
        }

        if let Some((nm, ip)) = pnc.take() {
            let n = if ip { pd_n } else { cd_n };
            let flat = if binary {
                read_binary_be::<f32>(reader, n * 3, "f32")?
            } else {
                read_ascii::<f32>(reader, n * 3, "f32")?
            };
            let v: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            if ip {
                grid.point_data
                    .insert(nm, AttributeArray::Normals { values: v });
            } else {
                grid.cell_data
                    .insert(nm, AttributeArray::Normals { values: v });
            }
            continue;
        }

        let line = match read_line(reader)? {
            Some(l) => l,
            None => break,
        };
        let upper = line.to_ascii_uppercase();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }

        if upper.starts_with("POINTS") {
            let n: usize = toks[1].parse().with_context(|| "bad POINTS count")?;
            let dbl = toks
                .get(2)
                .map(|s| s.eq_ignore_ascii_case("double"))
                .unwrap_or(false);
            grid.points = if binary {
                if dbl {
                    let r = read_binary_be::<f64>(reader, n * 3, "f64")?;
                    r.chunks_exact(3)
                        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                        .collect()
                } else {
                    let r = read_binary_be::<f32>(reader, n * 3, "f32")?;
                    r.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
                }
            } else {
                let r = read_ascii::<f32>(reader, n * 3, "f32")?;
                r.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
            };
        } else if upper.starts_with("CELLS") {
            let nc: usize = toks[1].parse().with_context(|| "bad CELLS count")?;
            let sz: usize = toks[2].parse().with_context(|| "bad CELLS size")?;
            grid.cells = if binary {
                parse_cells_from_ints(&read_binary_be::<i32>(reader, sz, "i32")?, nc)?
            } else {
                let mut cells = Vec::with_capacity(nc);
                for _ in 0..nc {
                    let cl = read_line(reader)?.with_context(|| "EOF CELLS")?;
                    let p: Vec<&str> = cl.split_whitespace().collect();
                    let cnt: usize = p[0].parse().with_context(|| format!("bad count: {}", cl))?;
                    if p.len() < 1 + cnt {
                        bail!("short cell: {}", cl);
                    }
                    cells.push(
                        (1..=cnt)
                            .map(|i| p[i].parse::<u32>().unwrap())
                            .collect::<Vec<_>>(),
                    );
                }
                cells
            };
        } else if upper.starts_with("CELL_TYPES") {
            let n: usize = toks[1].parse().with_context(|| "bad CELL_TYPES count")?;
            grid.cell_types = if binary {
                read_binary_be::<i32>(reader, n, "i32")?
                    .iter()
                    .map(|&v| {
                        VtkCellType::try_from(v as u8).unwrap_or_else(|_| {
                            tracing::warn!(
                                code = v,
                                "unknown VTK cell type code; mapped to Vertex"
                            );
                            VtkCellType::Vertex
                        })
                    })
                    .collect()
            } else {
                let mut t = Vec::with_capacity(n);
                for _ in 0..n {
                    let tl = read_line(reader)?.with_context(|| "EOF CELL_TYPES")?;
                    let code: i32 = tl
                        .trim()
                        .parse()
                        .with_context(|| format!("bad type: {}", tl.trim()))?;
                    t.push(VtkCellType::try_from(code as u8).unwrap_or_else(|_| {
                        tracing::warn!(code, "unknown VTK cell type code; mapped to Vertex");
                        VtkCellType::Vertex
                    }));
                }
                t
            };
        } else if upper.starts_with("POINT_DATA") {
            pd_n = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            in_point_data = true;
        } else if upper.starts_with("CELL_DATA") {
            cd_n = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            in_point_data = false;
        } else if upper.starts_with("SCALARS") {
            if toks.len() < 3 {
                bail!("SCALARS malformed: {}", line);
            }
            let ncomp = toks.get(3).and_then(|s| s.parse().ok()).unwrap_or(1usize);
            psc = Some((toks[1].to_string(), ncomp, in_point_data));
        } else if upper.starts_with("VECTORS") {
            pvc = Some((toks[1].to_string(), in_point_data));
        } else if upper.starts_with("NORMALS") {
            pnc = Some((toks[1].to_string(), in_point_data));
        }
        // Unknown keywords silently skipped for forward compatibility.
    }

    grid.validate().map_err(|e| anyhow!("{}", e))?;
    Ok(grid)
}

fn write_unstructured_grid(w: &mut dyn Write, grid: &VtkUnstructuredGrid) -> Result<()> {
    let np = grid.n_points();
    let nc = grid.n_cells();

    writeln!(w, "# vtk DataFile Version 3.0")?;
    writeln!(w, "RITK exported unstructured grid")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    writeln!(w, "POINTS {} float", np)?;
    for [x, y, z] in &grid.points {
        writeln!(w, "{} {} {}", x, y, z)?;
    }

    let sz: usize = grid.cells.iter().map(|c| 1 + c.len()).sum();
    writeln!(w, "CELLS {} {}", nc, sz)?;
    for cell in &grid.cells {
        let p: Vec<String> = std::iter::once(cell.len().to_string())
            .chain(cell.iter().map(|i| i.to_string()))
            .collect();
        writeln!(w, "{}", p.join(" "))?;
    }

    writeln!(w, "CELL_TYPES {}", nc)?;
    for t in &grid.cell_types {
        writeln!(w, "{}", u8::from(*t))?;
    }

    if !grid.point_data.is_empty() {
        writeln!(w, "POINT_DATA {}", np)?;
        for (name, attr) in &grid.point_data {
            write_attribute_legacy(w, name, attr)?;
        }
    }

    if !grid.cell_data.is_empty() {
        writeln!(w, "CELL_DATA {}", nc)?;
        for (name, attr) in &grid.cell_data {
            write_attribute_legacy(w, name, attr)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
