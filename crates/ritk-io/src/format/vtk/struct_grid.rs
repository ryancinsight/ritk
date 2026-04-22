//! VTK legacy STRUCTURED_GRID reader/writer.
//!
//! Structured grid G = (D, V, A_P, A_C):
//! - D = (nx, ny, nz), |V| = nx*ny*nz
//! - A_P: per-point attributes, length n_points * ncomp
//! - A_C: per-cell attributes, length n_cells * ncomp
//!
//! Reference: VTK File Formats (legacy) section 4.3, Kitware Inc.

use crate::domain::vtk_data_object::{AttributeArray, VtkStructuredGrid};
use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
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
pub fn write_vtk_structured_grid<P: AsRef<Path>>(
    path: P,
    grid: &VtkStructuredGrid,
) -> Result<()> {
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
        other => bail!("unsupported VTK encoding: {}", other),
    };
    let ds_line = read_line(reader)?.with_context(|| "EOF before DATASET line")?;
    if !ds_line.to_ascii_uppercase().contains("STRUCTURED_GRID") {
        bail!("expected DATASET STRUCTURED_GRID, got: {}", ds_line);
    }

    let mut grid = VtkStructuredGrid::default();
    let mut in_point_data = false;
    let mut in_cell_data = false;
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
                read_binary_f32(reader, total)?
            } else {
                read_ascii_f32(reader, total)?
            };
            let arr = AttributeArray::Scalars { values, num_components: ncomp };
            if is_pd { grid.point_data.insert(name, arr); } else { grid.cell_data.insert(name, arr); }
            continue;
        }
        if let Some((name, is_pd)) = pending_vectors.take() {
            let n = if is_pd { pd_n } else { cd_n };
            let flat = if binary { read_binary_f32(reader, n * 3)? } else { read_ascii_f32(reader, n * 3)? };
            let values: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            let arr = AttributeArray::Vectors { values };
            if is_pd { grid.point_data.insert(name, arr); } else { grid.cell_data.insert(name, arr); }
            continue;
        }
        if let Some((name, is_pd)) = pending_normals.take() {
            let n = if is_pd { pd_n } else { cd_n };
            let flat = if binary { read_binary_f32(reader, n * 3)? } else { read_ascii_f32(reader, n * 3)? };
            let values: Vec<[f32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            let arr = AttributeArray::Normals { values };
            if is_pd { grid.point_data.insert(name, arr); } else { grid.cell_data.insert(name, arr); }
            continue;
        }

        let line = match read_line(reader)? { Some(l) => l, None => break };
        let upper = line.to_ascii_uppercase();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() { continue; }

        if upper.starts_with("DIMENSIONS") {
            if tokens.len() < 4 { bail!("DIMENSIONS malformed: {}", line); }
            let nx: usize = tokens[1].parse().with_context(|| "bad DIMENSIONS nx")?;
            let ny: usize = tokens[2].parse().with_context(|| "bad DIMENSIONS ny")?;
            let nz: usize = tokens[3].parse().with_context(|| "bad DIMENSIONS nz")?;
            grid.dimensions = [nx, ny, nz];
        } else if upper.starts_with("POINTS") {
            if tokens.len() < 2 { bail!("POINTS malformed: {}", line); }
            let n: usize = tokens[1].parse().with_context(|| "bad POINTS count")?;
            let is_double = tokens.get(2)
                .map(|s| s.to_ascii_lowercase() == "double")
                .unwrap_or(false);
            grid.points = if binary {
                if is_double {
                    let raw = read_binary_f64(reader, n * 3)?;
                    raw.chunks_exact(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect()
                } else {
                    let raw = read_binary_f32(reader, n * 3)?;
                    raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
                }
            } else {
                let raw = read_ascii_f32(reader, n * 3)?;
                raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
            };
        } else if upper.starts_with("POINT_DATA") {
            if tokens.len() < 2 { bail!("POINT_DATA malformed: {}", line); }
            pd_n = tokens[1].parse().with_context(|| "bad POINT_DATA count")?;
            in_point_data = true; in_cell_data = false;
        } else if upper.starts_with("CELL_DATA") {
            if tokens.len() < 2 { bail!("CELL_DATA malformed: {}", line); }
            cd_n = tokens[1].parse().with_context(|| "bad CELL_DATA count")?;
            in_cell_data = true; in_point_data = false;
        } else if upper.starts_with("SCALARS") {
            if tokens.len() < 3 { bail!("SCALARS malformed: {}", line); }
            let name = tokens[1].to_string();
            let ncomp: usize = tokens.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            pending_scalars = Some((name, ncomp, in_point_data));
        } else if upper.starts_with("VECTORS") {
            if tokens.len() < 2 { bail!("VECTORS malformed: {}", line); }
            pending_vectors = Some((tokens[1].to_string(), in_point_data));
        } else if upper.starts_with("NORMALS") {
            if tokens.len() < 2 { bail!("NORMALS malformed: {}", line); }
            pending_normals = Some((tokens[1].to_string(), in_point_data));
        } else if upper.starts_with("LOOKUP_TABLE") {
            // Standalone LOOKUP_TABLE outside SCALARS context: skip.
        }
        // Unknown keywords silently skipped for forward compatibility.
        let _ = (in_cell_data, in_point_data);
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
            write_attribute(w, name, attr)?;
        }
    }
    let nc = grid.n_cells();
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
        AttributeArray::Scalars { values, num_components } => {
            writeln!(w, "SCALARS {} float {}", name, num_components)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for v in values { writeln!(w, "{}", v)?; }
        }
        AttributeArray::Vectors { values } => {
            writeln!(w, "VECTORS {} float", name)?;
            for [x, y, z] in values { writeln!(w, "{} {} {}", x, y, z)?; }
        }
        AttributeArray::Normals { values } => {
            writeln!(w, "NORMALS {} float", name)?;
            for [x, y, z] in values { writeln!(w, "{} {} {}", x, y, z)?; }
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(w, "TEXTURE_COORDINATES {} float {}", name, dim)?;
            for chunk in values.chunks(*dim) {
                let parts: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
                writeln!(w, "{}", parts.join(" "))?;
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
        if n == 0 { return Ok(None); }
        let trimmed = buf.trim();
        if !trimmed.is_empty() { return Ok(Some(trimmed.to_owned())); }
    }
}

fn read_ascii_f32(reader: &mut dyn BufRead, count: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = String::new();
    while out.len() < count {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 { break; }
        for tok in buf.split_whitespace() {
            if out.len() >= count { break; }
            let v: f32 = tok.parse().with_context(|| format!("bad f32 token: {}", tok))?;
            out.push(v);
        }
    }
    if out.len() != count { bail!("expected {} f32 values, got {}", count, out.len()); }
    Ok(out)
}

fn read_binary_f32(reader: &mut dyn Read, count: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    reader.read_exact(&mut buf)
        .with_context(|| format!("truncated binary f32 (need {})", count))?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_binary_f64(reader: &mut dyn Read, count: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0u8; count * 8];
    reader.read_exact(&mut buf)
        .with_context(|| format!("truncated binary f64 (need {})", count))?;
    Ok(buf.chunks_exact(8)
        .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkStructuredGrid};
    use tempfile::NamedTempFile;

    #[test]
    fn test_structured_grid_roundtrip_identity() {
        // 2x3x2 grid, n_points=12, i-fastest ordering.
        let dims = [2usize, 3, 2];
        let mut points = Vec::with_capacity(12);
        for iz in 0..2usize {
            for iy in 0..3usize {
                for ix in 0..2usize {
                    points.push([ix as f32, iy as f32, iz as f32]);
                }
            }
        }
        let scalars = AttributeArray::Scalars {
            values: (0..12).map(|i| i as f32).collect(),
            num_components: 1,
        };
        let mut grid = VtkStructuredGrid::new(dims);
        grid.points = points.clone();
        grid.point_data.insert("intensity".to_string(), scalars);

        let tmp = NamedTempFile::new().expect("temp file");
        write_vtk_structured_grid(tmp.path(), &grid).expect("write");
        let result = read_vtk_structured_grid(tmp.path()).expect("read");

        assert_eq!(result.dimensions, [2, 3, 2]);
        assert_eq!(result.n_points(), 12);
        for (i, (expected, got)) in points.iter().zip(result.points.iter()).enumerate() {
            for c in 0..3 {
                assert!((expected[c] - got[c]).abs() < 1e-5,
                    "point[{}][{}]: expected {} got {}", i, c, expected[c], got[c]);
            }
        }
        match result.point_data.get("intensity").expect("intensity attribute") {
            AttributeArray::Scalars { values, num_components } => {
                assert_eq!(*num_components, 1);
                assert_eq!(values.len(), 12);
                for i in 0..12 {
                    assert!((values[i] - i as f32).abs() < 1e-5,
                        "scalar[{}]: expected {} got {}", i, i as f32, values[i]);
                }
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    #[test]
    fn test_structured_grid_validate_rejects_wrong_point_count() {
        // [2,2,2] requires 8 points; supply 5 -> validate fails -> write returns Err.
        let mut grid = VtkStructuredGrid::new([2, 2, 2]);
        grid.points = vec![[0.0f32; 3]; 5];
        let tmp = NamedTempFile::new().expect("temp file");
        let result = write_vtk_structured_grid(tmp.path(), &grid);
        assert!(result.is_err(), "write must fail when point count mismatches dimensions");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("n_points") || msg.contains("points"),
            "error message must reference point count: got {}", msg);
    }

    #[test]
    fn test_structured_grid_roundtrip_vectors() {
        // [2,2,1] -> n_points=4; VECTORS velocity with 4 entries.
        let dims = [2usize, 2, 1];
        let mut points = Vec::with_capacity(4);
        for iz in 0..1usize {
            for iy in 0..2usize {
                for ix in 0..2usize {
                    points.push([ix as f32, iy as f32, iz as f32]);
                }
            }
        }
        let expected_vecs: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let mut grid = VtkStructuredGrid::new(dims);
        grid.points = points;
        grid.point_data.insert(
            "velocity".to_string(),
            AttributeArray::Vectors { values: expected_vecs.clone() },
        );

        let tmp = NamedTempFile::new().expect("temp file");
        write_vtk_structured_grid(tmp.path(), &grid).expect("write");
        let result = read_vtk_structured_grid(tmp.path()).expect("read");

        assert_eq!(result.n_points(), 4);
        match result.point_data.get("velocity").expect("velocity attribute") {
            AttributeArray::Vectors { values } => {
                assert_eq!(values.len(), 4);
                for (i, (exp, got)) in expected_vecs.iter().zip(values.iter()).enumerate() {
                    for c in 0..3 {
                        assert!((exp[c] - got[c]).abs() < 1e-5,
                            "velocity[{}][{}]: expected {} got {}", i, c, exp[c], got[c]);
                    }
                }
            }
            other => panic!("expected Vectors, got {:?}", other),
        }
    }
}
