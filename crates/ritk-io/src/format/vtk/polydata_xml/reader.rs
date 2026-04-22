//! VTK XML PolyData (.vtp) reader (ASCII inline format).

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub fn read_vtp_polydata<P: AsRef<Path>>(path: P) -> Result<VtkPolyData> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTP: {}", path.as_ref().display()))?;
    parse_vtp(&s)
}

pub(crate) fn parse_vtp(input: &str) -> Result<VtkPolyData> {
    let piece = find_tag(input, "Piece")
        .ok_or_else(|| anyhow::anyhow!("missing <Piece>"))?;
    let n_points: usize = attr_usize(&piece, "NumberOfPoints")?;

    let points_sec = find_section(input, "Points")
        .ok_or_else(|| anyhow::anyhow!("missing <Points>"))?;
    let coords = parse_floats(&extract_content(&points_sec));
    if coords.len() != n_points * 3 {
        bail!("expected {} coord values, got {}", n_points*3, coords.len());
    }
    let points: Vec<[f32;3]> = coords.chunks_exact(3).map(|c| [c[0],c[1],c[2]]).collect();

    let mut poly = VtkPolyData::default();
    poly.points = points;
    poly.vertices        = parse_cells(input, "Verts");
    poly.lines           = parse_cells(input, "Lines");
    poly.polygons        = parse_cells(input, "Polys");
    poly.triangle_strips = parse_cells(input, "Strips");

    if let Some(sec) = find_section(input, "PointData") {
        poly.point_data = parse_attrs(&sec);
    }
    if let Some(sec) = find_section(input, "CellData") {
        poly.cell_data = parse_attrs(&sec);
    }
    Ok(poly)
}

fn find_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let start = s.find(&open)?;
    let end = s[start..].find(">")? + 1;
    Some(s[start..start+end].to_string())
}

fn find_section(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = s.find(&open)?;
    let end_pos = s[start..].find(&close)? + close.len() + start;
    if end_pos <= start { return None; }
    Some(s[start..end_pos].to_string())
}

fn attr_usize(tag: &str, name: &str) -> Result<usize> {
    let v = attr_val(tag, name).ok_or_else(|| anyhow::anyhow!("attr {} not found", name))?;
    v.parse().with_context(|| format!("parse {}", name))
}

/// Extract XML attribute value for name="value" using char::from(34u8) for double-quote.
fn attr_val(tag: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8);
    let mut pat = name.to_string();
    pat.push(char::from(61u8));  // =
    pat.push(dq);
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find(dq)?;
    Some(rest[..end].to_string())
}

fn extract_content(section: &str) -> String {
    // Find opening DataArray tag (may be at root or nested), extract data content
    let da_start = section.find("<DataArray").unwrap_or(0);
    let rest = &section[da_start..];
    let gt = rest.find(">").unwrap_or(0) + 1;
    let lt = rest[gt..].find("</").map(|p| gt+p).unwrap_or(rest.len());
    rest[gt..lt].trim().to_string()
}

fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace().filter_map(|t| t.parse().ok()).collect()
}

fn parse_ints(s: &str) -> Vec<i32> {
    s.split_whitespace().filter_map(|t| t.parse().ok()).collect()
}

fn named_da(section: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8);
    let mut np = String::from("Name=");
    np.push(dq); np.push_str(name); np.push(dq);
    let start = section.find(&np)?;
    let da_start = section[..start].rfind("<DataArray")?;
    let rest = &section[da_start..];
    let close = "</DataArray>";
    let end = rest.find(close)? + close.len();
    Some(rest[..end].to_string())
}

fn parse_cells(input: &str, sname: &str) -> Vec<Vec<u32>> {
    let sec = match find_section(input, sname) { Some(s) => s, None => return vec![] };
    let conn_da = match named_da(&sec, "connectivity") { Some(s) => s, None => return vec![] };
    let offs_da = match named_da(&sec, "offsets") { Some(s) => s, None => return vec![] };
    let conn: Vec<u32> = parse_ints(&extract_content(&conn_da)).into_iter().map(|v| v as u32).collect();
    let offs: Vec<u32> = parse_ints(&extract_content(&offs_da)).into_iter().map(|v| v as u32).collect();
    if offs.is_empty() { return vec![]; }
    let mut cells = Vec::new();
    let mut prev = 0usize;
    for &off in &offs {
        let off = off as usize;
        if off <= conn.len() { cells.push(conn[prev..off].to_vec()); }
        prev = off;
    }
    cells
}

fn parse_attrs(section: &str) -> HashMap<String, AttributeArray> {
    let mut map = HashMap::new();
    let mut rest = section;
    let close = "</DataArray>";
    loop {
        let start = match rest.find("<DataArray") { Some(s) => s, None => break };
        rest = &rest[start..];
        let te = match rest.find(">") { Some(e) => e+1, None => break };
        let tag = rest[..te].to_string();
        let name = attr_val(&tag, "Name").unwrap_or_default();
        let ncomp: usize = attr_val(&tag, "NumberOfComponents")
            .and_then(|s| s.parse().ok()).unwrap_or(1);
        let de = match rest.find(close) { Some(e) => e, None => break };
        let data = rest[te..de].trim().to_string();
        let floats = parse_floats(&data);
        if !name.is_empty() {
            let attr = match ncomp {
                3 => {
                    let v3: Vec<[f32;3]> = floats.chunks_exact(3).map(|c|[c[0],c[1],c[2]]).collect();
                    if name.to_lowercase().contains("normal") { AttributeArray::Normals { values: v3 } }
                    else { AttributeArray::Vectors { values: v3 } }
                }
                2 => AttributeArray::TextureCoords { values: floats, dim: 2 },
                n => AttributeArray::Scalars { values: floats, num_components: n },
            };
            map.insert(name, attr);
        }
        rest = &rest[de + close.len()..];
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
    use crate::format::vtk::polydata_xml::writer::write_vtp_str;

    fn triangle() -> VtkPolyData {
        let mut p = VtkPolyData::default();
        p.points = vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]];
        p.polygons = vec![vec![0,1,2]];
        p
    }

    #[test] fn test_triangle_parse() {
        let p = parse_vtp(&write_vtp_str(&triangle())).unwrap();
        assert_eq!(p.points.len(), 3);
        assert_eq!(p.polygons.len(), 1);
        assert_eq!(p.polygons[0], vec![0u32,1,2]);
        assert!((p.points[1][0]-1.0).abs()<1e-5);
    }
    #[test] fn test_empty_parse() {
        let p = parse_vtp(&write_vtp_str(&Default::default())).unwrap();
        assert_eq!(p.points.len(), 0);
        assert_eq!(p.polygons.len(), 0);
    }
    #[test] fn test_scalars_roundtrip() {
        let mut pd = triangle();
        pd.point_data.insert("pressure".to_string(),
            AttributeArray::Scalars { values: vec![1.0,2.0,3.0], num_components: 1 });
        let p = parse_vtp(&write_vtp_str(&pd)).unwrap();
        match p.point_data.get("pressure") {
            Some(AttributeArray::Scalars { values, .. }) => { assert!((values[0]-1.0).abs()<1e-4); }
            _ => panic!("not Scalars"),
        }
    }
    #[test] fn test_lines_parse() {
        let mut pd = VtkPolyData::default();
        pd.points = vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]];
        pd.lines = vec![vec![0,1,2]];
        let p = parse_vtp(&write_vtp_str(&pd)).unwrap();
        assert_eq!(p.lines.len(), 1);
        assert_eq!(p.lines[0], vec![0u32,1,2]);
    }
    #[test] fn test_vectors_roundtrip() {
        let mut pd = VtkPolyData::default();
        pd.points = vec![[0.0,0.0,0.0]];
        pd.point_data.insert("vel".to_string(),
            AttributeArray::Vectors { values: vec![[1.0,2.0,3.0]] });
        let p = parse_vtp(&write_vtp_str(&pd)).unwrap();
        match p.point_data.get("vel") {
            Some(AttributeArray::Vectors { values }) => { assert!((values[0][0]-1.0).abs()<1e-4); }
            _ => panic!("not Vectors"),
        }
    }
    #[test] fn test_missing_points_error() {
        let dq = char::from(34u8);
        let mut s = String::from("<VTKFile><PolyData><Piece NumberOfPoints=");
        s.push(dq); s.push_str("1"); s.push(dq); s.push_str("></Piece></PolyData></VTKFile>");
        assert!(parse_vtp(&s).is_err());
    }
    #[test] fn test_wrong_coord_count_error() {
        let mut pd = VtkPolyData::default();
        pd.points = vec![[0.0,0.0,0.0]];
        let s = write_vtp_str(&pd).replace("NumberOfPoints=", "NumberOfPointsBad=");
        // Force n_points=99 by injecting wrong attr
        let dq = char::from(34u8);
        let mut bad = String::from("NumberOfPoints=");
        bad.push(dq); bad.push_str("99"); bad.push(dq);
        let s2 = s + &bad;
        assert!(parse_vtp(&s2).is_err());
    }
}
