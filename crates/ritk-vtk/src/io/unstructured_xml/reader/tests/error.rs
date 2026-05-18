use crate::domain::vtk_data_object::{VtkCellType, VtkUnstructuredGrid};
use crate::io::unstructured_xml::reader::{parse_vtu, read_vtu_unstructured_grid};
use crate::io::unstructured_xml::writer::write_vtu_unstructured_grid;
use tempfile::NamedTempFile;

fn tetra() -> VtkUnstructuredGrid {
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    g.cells = vec![vec![0u32, 1, 2, 3]];
    g.cell_types = vec![VtkCellType::Tetra];
    g
}

/// Build a minimal VTU XML string from raw section content for error injection.
fn minimal_vtu(np: usize, nc: usize, points: &str, conn: &str, offs: &str, types: &str) -> String {
    format!(
        "<?xml version=\"1.0\"?>\n\
         <VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n\
           <UnstructuredGrid>\n\
             <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n\
               <Points>\n\
                 <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n\
                   {}\n\
                 </DataArray>\n\
               </Points>\n\
               <Cells>\n\
                 <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n\
                   {}\n\
                 </DataArray>\n\
                 <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n\
                   {}\n\
                 </DataArray>\n\
                 <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n\
                   {}\n\
                 </DataArray>\n\
               </Cells>\n\
             </Piece>\n\
           </UnstructuredGrid>\n\
         </VTKFile>",
        np, nc, points, conn, offs, types
    )
}

#[test]
fn test_missing_piece_tag_error() {
    let input = "<?xml version=\"1.0\"?>\
                 <VTKFile><UnstructuredGrid></UnstructuredGrid></VTKFile>";
    let r = parse_vtu(input);
    assert!(r.is_err(), "missing <Piece> must return Err");
    let msg = r.unwrap_err().to_string();
    assert!(msg.contains("Piece"), "error must mention 'Piece': {}", msg);
}

#[test]
fn test_missing_cells_section_error() {
    // Valid points but no <Cells> section.
    let s = "<?xml version=\"1.0\"?>\n\
             <VTKFile type=\"UnstructuredGrid\">\n\
               <UnstructuredGrid>\n\
                 <Piece NumberOfPoints=\"0\" NumberOfCells=\"0\">\n\
                   <Points>\n\
                     <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n\
                     </DataArray>\n\
                   </Points>\n\
                 </Piece>\n\
               </UnstructuredGrid>\n\
             </VTKFile>";
    let r = parse_vtu(s);
    assert!(r.is_err(), "missing <Cells> must return Err");
    let msg = r.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("cell"),
        "error must mention 'Cells': {}",
        msg
    );
}

#[test]
fn test_wrong_coord_count_error() {
    // NumberOfPoints=4 but only 3 floats (need 12) provided.
    let s = minimal_vtu(4, 0, "0.0 0.0 0.0", "", "", "");
    let r = parse_vtu(&s);
    assert!(r.is_err(), "coord count mismatch must return Err");
    let msg = r.unwrap_err().to_string();
    // Error contains the expected count (12) or actual count (3) or "coord".
    assert!(
        msg.contains("12") || msg.contains("coord") || msg.contains("3"),
        "error must mention coord counts: {}",
        msg
    );
}

#[test]
fn test_offsets_count_mismatch_error() {
    // NumberOfCells=1 but two offsets provided.
    let s = minimal_vtu(
        4,
        1,
        "0 0 0  1 0 0  0 1 0  0 0 1",
        "0 1 2 3",
        "4 8", // two offsets for one cell
        "10",
    );
    let r = parse_vtu(&s);
    assert!(r.is_err(), "offsets count mismatch must return Err");
    let msg = r.unwrap_err().to_string();
    assert!(
        msg.contains("offsets"),
        "error must mention 'offsets': {}",
        msg
    );
}

#[test]
fn test_offset_exceeds_connectivity_error() {
    // Offset 999 exceeds connectivity length 4.
    let s = minimal_vtu(4, 1, "0 0 0  1 0 0  0 1 0  0 0 1", "0 1 2 3", "999", "10");
    let r = parse_vtu(&s);
    assert!(r.is_err(), "offset > connectivity length must return Err");
    let msg = r.unwrap_err().to_string();
    assert!(
        msg.contains("999") || msg.contains("offset") || msg.contains("connectivity"),
        "error must mention the out-of-bounds offset: {}",
        msg
    );
}

#[test]
fn test_types_count_mismatch_error() {
    // NumberOfCells=2 but only one type provided.
    let s = minimal_vtu(
        4,
        2,
        "0 0 0  1 0 0  0 1 0  0 0 1",
        "0 1 2  0 1 2 3",
        "3 7",
        "5", // one type for two cells
    );
    let r = parse_vtu(&s);
    assert!(r.is_err(), "types count mismatch must return Err");
    let msg = r.unwrap_err().to_string();
    assert!(
        msg.contains("types") || msg.contains("type"),
        "error must mention 'types': {}",
        msg
    );
}

#[test]
fn test_from_file_roundtrip() {
    let g = tetra();
    let tmp = NamedTempFile::new().expect("temp file");
    write_vtu_unstructured_grid(tmp.path(), &g).expect("write");
    let r = read_vtu_unstructured_grid(tmp.path()).expect("read");
    assert_eq!(r.n_points(), 4);
    assert_eq!(r.n_cells(), 1);
    assert_eq!(r.cells[0], vec![0u32, 1, 2, 3]);
    assert_eq!(r.cell_types[0], VtkCellType::Tetra);
}

#[test]
fn test_nonexistent_file_error() {
    let r = read_vtu_unstructured_grid("/nonexistent_dir_xyz/file.vtu");
    assert!(r.is_err(), "nonexistent path must return Err");
}
