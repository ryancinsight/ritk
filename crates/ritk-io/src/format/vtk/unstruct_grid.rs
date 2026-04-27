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
use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
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
    let mut in_cell_data = false;
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
                read_binary_f32(reader, n * nc)?
            } else {
                read_ascii_f32(reader, n * nc)?
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
                read_binary_f32(reader, n * 3)?
            } else {
                read_ascii_f32(reader, n * 3)?
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
                read_binary_f32(reader, n * 3)?
            } else {
                read_ascii_f32(reader, n * 3)?
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
                .map(|s| s.to_ascii_lowercase() == "double")
                .unwrap_or(false);
            grid.points = if binary {
                if dbl {
                    let r = read_binary_f64(reader, n * 3)?;
                    r.chunks_exact(3)
                        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                        .collect()
                } else {
                    let r = read_binary_f32(reader, n * 3)?;
                    r.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
                }
            } else {
                let r = read_ascii_f32(reader, n * 3)?;
                r.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
            };
        } else if upper.starts_with("CELLS") {
            let nc: usize = toks[1].parse().with_context(|| "bad CELLS count")?;
            let sz: usize = toks[2].parse().with_context(|| "bad CELLS size")?;
            grid.cells = if binary {
                parse_cells_from_ints(&read_binary_i32(reader, sz)?, nc)?
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
                read_binary_i32(reader, n)?
                    .iter()
                    .map(|&v| {
                        VtkCellType::from_u8(v as u8).unwrap_or_else(|| {
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
                    t.push(VtkCellType::from_u8(code as u8).unwrap_or_else(|| {
                        tracing::warn!(code, "unknown VTK cell type code; mapped to Vertex");
                        VtkCellType::Vertex
                    }));
                }
                t
            };
        } else if upper.starts_with("POINT_DATA") {
            pd_n = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            in_point_data = true;
            in_cell_data = false;
        } else if upper.starts_with("CELL_DATA") {
            cd_n = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            in_cell_data = true;
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
        let _ = (in_cell_data, in_point_data);
    }
    grid.validate().map_err(|e| anyhow!("{}", e))?;
    Ok(grid)
}

fn parse_cells_from_ints(ints: &[i32], n: usize) -> Result<Vec<Vec<u32>>> {
    let mut cells = Vec::with_capacity(n);
    let mut pos = 0;
    for _ in 0..n {
        if pos >= ints.len() {
            bail!("truncated CELLS");
        }
        let c = ints[pos] as usize;
        pos += 1;
        if pos + c > ints.len() {
            bail!("cell overrun");
        }
        cells.push(ints[pos..pos + c].iter().map(|&i| i as u32).collect());
        pos += c;
    }
    Ok(cells)
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
        writeln!(w, "{}", t.to_u8())?;
    }
    if !grid.point_data.is_empty() {
        writeln!(w, "POINT_DATA {}", np)?;
        for (name, attr) in &grid.point_data {
            write_attribute(w, name, attr)?;
        }
    }
    if !grid.cell_data.is_empty() {
        writeln!(w, "CELL_DATA {}", nc)?;
        for (name, attr) in &grid.cell_data {
            write_attribute(w, name, attr)?;
        }
    }
    Ok(())
}

fn write_attribute(w: &mut dyn Write, name: &str, attr: &AttributeArray) -> Result<()> {
    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            writeln!(w, "SCALARS {} float {}", name, num_components)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for v in values {
                writeln!(w, "{}", v)?;
            }
        }
        AttributeArray::Vectors { values } => {
            writeln!(w, "VECTORS {} float", name)?;
            for [x, y, z] in values {
                writeln!(w, "{} {} {}", x, y, z)?;
            }
        }
        AttributeArray::Normals { values } => {
            writeln!(w, "NORMALS {} float", name)?;
            for [x, y, z] in values {
                writeln!(w, "{} {} {}", x, y, z)?;
            }
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(w, "TEXTURE_COORDINATES {} float {}", name, dim)?;
            for chunk in values.chunks(*dim) {
                let p: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
                writeln!(w, "{}", p.join(" "))?;
            }
        }
    }
    Ok(())
}

fn read_line(reader: &mut dyn BufRead) -> Result<Option<String>> {
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        let t = buf.trim();
        if !t.is_empty() {
            return Ok(Some(t.to_owned()));
        }
    }
}

fn read_ascii_f32(reader: &mut dyn BufRead, count: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = String::new();
    while out.len() < count {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            break;
        }
        for tok in buf.split_whitespace() {
            if out.len() >= count {
                break;
            }
            out.push(
                tok.parse::<f32>()
                    .with_context(|| format!("bad f32: {}", tok))?,
            );
        }
    }
    if out.len() != count {
        bail!("expected {} f32, got {}", count, out.len());
    }
    Ok(out)
}

fn read_binary_f32(reader: &mut dyn Read, count: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("trunc f32 (need {})", count))?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_binary_f64(reader: &mut dyn Read, count: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0u8; count * 8];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("trunc f64 (need {})", count))?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn read_binary_i32(reader: &mut dyn Read, count: usize) -> Result<Vec<i32>> {
    let mut buf = vec![0u8; count * 4];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("trunc i32 (need {})", count))?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
    use tempfile::NamedTempFile;

    #[test]
    fn test_unstructured_grid_roundtrip_tetrahedra() {
        let mut grid = VtkUnstructuredGrid::new();
        grid.points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        grid.cells = vec![vec![0u32, 1, 2, 3]];
        grid.cell_types = vec![VtkCellType::Tetra];
        grid.point_data.insert(
            "pressure".to_string(),
            AttributeArray::Scalars {
                values: vec![0.0, 1.0, 2.0, 3.0],
                num_components: 1,
            },
        );
        grid.cell_data.insert(
            "stress".to_string(),
            AttributeArray::Scalars {
                values: vec![42.0],
                num_components: 1,
            },
        );
        let tmp = NamedTempFile::new().expect("temp");
        write_vtk_unstructured_grid(tmp.path(), &grid).expect("write");
        let r = read_vtk_unstructured_grid(tmp.path()).expect("read");
        assert_eq!(r.n_points(), 4);
        assert_eq!(r.n_cells(), 1);
        assert_eq!(r.cells[0], vec![0u32, 1, 2, 3]);
        assert_eq!(r.cell_types[0], VtkCellType::Tetra);
        assert_eq!(r.cell_types[0].to_u8(), 10);
        match r.point_data.get("pressure").expect("pressure") {
            AttributeArray::Scalars { values, .. } => {
                assert_eq!(values.len(), 4);
                for i in 0..4 {
                    assert!(
                        (values[i] - i as f32).abs() < 1e-5,
                        "pressure[{}]: exp {} got {}",
                        i,
                        i as f32,
                        values[i]
                    );
                }
            }
            other => panic!("expected Scalars: {:?}", other),
        }
        match r.cell_data.get("stress").expect("stress") {
            AttributeArray::Scalars { values, .. } => {
                assert_eq!(values.len(), 1);
                assert!(
                    (values[0] - 42.0).abs() < 1e-5,
                    "stress[0]: exp 42.0 got {}",
                    values[0]
                );
            }
            other => panic!("expected Scalars: {:?}", other),
        }
    }

    #[test]
    fn test_unstructured_grid_roundtrip_multiple_cells() {
        let mut grid = VtkUnstructuredGrid::new();
        grid.points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        grid.cells = vec![vec![0u32, 1, 2], vec![1u32, 2, 3]];
        grid.cell_types = vec![VtkCellType::Triangle, VtkCellType::Triangle];
        let tmp = NamedTempFile::new().expect("temp");
        write_vtk_unstructured_grid(tmp.path(), &grid).expect("write");
        let r = read_vtk_unstructured_grid(tmp.path()).expect("read");
        assert_eq!(r.n_cells(), 2);
        assert_eq!(r.cells[0], vec![0u32, 1, 2]);
        assert_eq!(r.cells[1], vec![1u32, 2, 3]);
        assert_eq!(r.cell_types[0], VtkCellType::Triangle);
        assert_eq!(r.cell_types[1], VtkCellType::Triangle);
        assert_eq!(r.cell_types[0].to_u8(), 5);
    }

    #[test]
    fn test_unstructured_grid_validate_rejects_wrong_types_count() {
        let mut grid = VtkUnstructuredGrid::new();
        grid.points = vec![[0.0f32; 3]; 4];
        grid.cells = vec![vec![0u32, 1, 2], vec![1u32, 2, 3]];
        grid.cell_types = vec![VtkCellType::Triangle]; // 2 cells, 1 type
        let tmp = NamedTempFile::new().expect("temp");
        assert!(write_vtk_unstructured_grid(tmp.path(), &grid).is_err());
    }

    #[test]
    fn test_unstructured_grid_validate_rejects_out_of_range_index() {
        let mut grid = VtkUnstructuredGrid::new();
        grid.points = vec![[0.0f32; 3]; 3];
        grid.cells = vec![vec![0u32, 1, 99]]; // index 99 >= n_points=3
        grid.cell_types = vec![VtkCellType::Triangle];
        let tmp = NamedTempFile::new().expect("temp");
        assert!(write_vtk_unstructured_grid(tmp.path(), &grid).is_err());
    }
}
