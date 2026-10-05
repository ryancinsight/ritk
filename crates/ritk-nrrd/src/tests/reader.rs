#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::fixtures::sample_value;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_core::rejection::assert_rejects;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

/// Write a minimal inline NRRD file with `MET_FLOAT`-equivalent (`float`)
/// data and the given spatial metadata.
fn write_inline_nrrd(
    path: &std::path::Path,
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    spacing: [f64; 3],
    origin: [f64; 3],
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "NRRD0004").unwrap();
    writeln!(f, "# written by ritk test helper").unwrap();
    writeln!(f, "type: float").unwrap();
    writeln!(f, "dimension: 3").unwrap();
    writeln!(f, "space: left-posterior-superior").unwrap();
    writeln!(f, "sizes: {} {} {}", nx, ny, nz).unwrap();
    writeln!(
        f,
        "space directions: ({},0,0) (0,{},0) (0,0,{})",
        spacing[0], spacing[1], spacing[2]
    )
    .unwrap();
    writeln!(f, "kinds: domain domain domain").unwrap();
    writeln!(f, "endian: little").unwrap();
    writeln!(f, "encoding: raw").unwrap();
    writeln!(
        f,
        "space origin: ({},{},{})",
        origin[0], origin[1], origin[2]
    )
    .unwrap();
    writeln!(f).unwrap(); // blank line terminates header
    for &v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn write_inline_planar_nrrd(path: &std::path::Path, data: &[f32], nx: usize, ny: usize) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("create planar NRRD fixture");
    writeln!(file, "NRRD0004").expect("write magic");
    writeln!(file, "type: float").expect("write type");
    writeln!(file, "dimension: 2").expect("write dimension");
    writeln!(file, "space: left-posterior-superior").expect("write space");
    writeln!(file, "sizes: {nx} {ny}").expect("write sizes");
    writeln!(file, "space directions: (0.5,0,0) (0,1.2,1.6)").expect("write directions");
    writeln!(file, "kinds: domain domain").expect("write kinds");
    writeln!(file, "endian: little").expect("write endian");
    writeln!(file, "encoding: raw").expect("write encoding");
    writeln!(file, "space origin: (3,4,5)").expect("write origin");
    writeln!(file).expect("terminate header");
    for &value in data {
        file.write_all(&value.to_le_bytes())
            .expect("write planar voxel");
    }
}

mod corruption;
mod geometry;
mod payload;
