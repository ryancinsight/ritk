//! VTK I/O module — free functions, VtkReader/VtkWriter wrappers, and sub-modules.

pub mod polydata;
pub use polydata::{read_vtk_polydata, write_vtk_polydata};
pub mod polydata_xml;
pub use polydata_xml::{read_vtp_polydata, write_vtp_polydata};
pub mod image_xml;
pub use image_xml::{
    read_vti_binary_appended, read_vti_binary_appended_bytes, read_vti_image_data,
    write_vti_binary_appended_bytes, write_vti_binary_appended_to_file, write_vti_image_data,
    write_vti_str,
};
pub mod struct_grid;
pub mod unstruct_grid;
pub use struct_grid::{read_vtk_structured_grid, write_vtk_structured_grid};
pub use unstruct_grid::{read_vtk_unstructured_grid, write_vtk_unstructured_grid};
pub mod unstructured_xml;
pub use unstructured_xml::{
    read_vtu_unstructured_grid, write_vtu_str, write_vtu_unstructured_grid,
};

pub mod obj;
pub use obj::{read_obj_mesh, write_obj_mesh};
pub mod stl;
pub use stl::{read_stl_mesh, write_stl_ascii, write_stl_binary};
pub mod ply;
pub use ply::{read_ply_mesh, write_ply_ascii, write_ply_binary_le};
pub mod gltf;
pub use gltf::write_gltf;

pub mod mesh_indexed;
pub use mesh_indexed::{
    read_obj_indexed, read_ply_indexed, read_stl_indexed, write_indexed_glb, write_indexed_obj,
    write_indexed_ply, write_indexed_stl_ascii, write_indexed_stl_binary,
};

pub(crate) mod legacy_write_attribute;
pub(crate) mod read_helpers;
pub(crate) mod xml_helpers;
pub mod xml_write_attr;

/// Reverse a three-component spatial vector between RITK `[depth, row, col]`
/// and VTK's `[X, Y, Z]` file order.
///
/// VTK legacy names the X extent first in `DIMENSIONS`, and `encode_vtk_flat`
/// takes that extent from `shape[2]`, so the file's X size is `spacing[2]` —
/// the *column* spacing. RITK's `Spacing<3>` is `[Δdepth, Δrow, Δcol]`, so the
/// two orders are reverses of one another and index-for-index transfer
/// transposes the metadata against the very axis the dimensions line names.
///
/// The map is its own inverse, so the reader and writer share it. `ORIGIN` is
/// a scanner-space position with no per-axis metadata and is never reversed.
pub(crate) const fn reverse_spatial_axes(vector: [f64; 3]) -> [f64; 3] {
    [vector[2], vector[1], vector[0]]
}

pub mod reader;
pub mod writer;

pub use reader::{read_vtk, read_vtk_flat};
pub use writer::{encode_vtk_flat, write_vtk};

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::path::Path;

/// Simple wrapper for reading VTK legacy structured-points images.
///
/// Does not implement `ritk_io::ImageReader`; that wrapper lives in `ritk-io`
/// to avoid orphan-rule violations.
pub struct VtkReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> VtkReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read a VTK legacy structured-points file at `path`.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<f32, B, 3>> {
        read_vtk(path, &self.backend)
    }
}

/// Simple wrapper for writing VTK legacy structured-points images.
///
/// Does not implement `ritk_io::ImageWriter`; that wrapper lives in `ritk-io`
/// to avoid orphan-rule violations.
pub struct VtkWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> VtkWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write a VTK legacy structured-points file to `path`.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_vtk(path, image, &self.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use ritk_spatial::{CoordinateMap, CurvilinearArray, Direction, Point, Spacing};
    use tempfile::tempdir;

    fn native_image(
        direction: Direction<3>,
        coordinate_map: CoordinateMap,
    ) -> Image<f32, SequentialBackend, 3> {
        let image = Image::from_flat_on(
            vec![1.25, -4.5],
            [1, 1, 2],
            Point::new([1.0, -2.0, 3.5]),
            Spacing::new([0.5, 0.75, 1.25]),
            direction,
            &SequentialBackend,
        )
        .expect("valid image values and dimensions");
        image
            .with_coordinate_map(coordinate_map)
            .expect("valid coordinate map for a three-dimensional image")
    }

    #[test]
    fn native_scalar_round_trip_preserves_values_and_spatial_metadata() {
        let backend = SequentialBackend;
        let shape = [2, 2, 3];
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|index| index as f32 * 0.25 - 1.0)
            .collect();
        let origin = Point::new([1.0, -2.0, 3.5]);
        let spacing = Spacing::new([0.5, 0.75, 1.25]);
        let image = Image::from_flat_on(
            values.clone(),
            shape,
            origin,
            spacing,
            Direction::identity(),
            &backend,
        )
        .expect("native image");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("roundtrip.vtk");

        VtkWriter::new(backend)
            .write(&path, &image)
            .expect("write VTK");
        let loaded = VtkReader::new(backend).read(&path).expect("read VTK");

        assert_eq!(loaded.shape(), shape);
        assert_eq!(loaded.data_slice().expect("contiguous image"), values);
        assert_eq!(*loaded.origin(), origin);
        assert_eq!(*loaded.spacing(), spacing);
        assert_eq!(*loaded.direction(), Direction::identity());
    }

    /// A hand-built VTK file's `SPACING` lands in RITK `[depth, row, col]` order.
    ///
    /// The oracle is a file this crate did not write, so the writer cannot make
    /// the reader look correct. `DIMENSIONS 4 3 2` names the X extent first, so
    /// the X size 0.5 is the *column* spacing and must arrive in `spacing[2]`.
    /// Reading the triple index-for-index puts it in `spacing[0]` — the depth
    /// slot — which no round trip can expose, because the writer and reader
    /// would transpose in the same direction.
    #[test]
    fn a_hand_built_vtk_spacing_lands_in_ritk_axis_order() {
        let backend = SequentialBackend;
        // 4 x 3 x 2 voxels = 24 big-endian f32, matching the declared type.
        let mut bytes = b"# vtk DataFile Version 3.0\nhand-built\nBINARY\n\
                          DATASET STRUCTURED_POINTS\nDIMENSIONS 4 3 2\n\
                          ORIGIN 0 0 0\nSPACING 0.5 0.75 1.25\n\
                          POINT_DATA 24\nSCALARS scalars float 1\n\
                          LOOKUP_TABLE default\n"
            .to_vec();
        for _ in 0..24 {
            bytes.extend_from_slice(&0.0f32.to_be_bytes());
        }
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("hand-built.vtk");
        std::fs::write(&path, &bytes).expect("write hand-built VTK");

        let loaded = VtkReader::new(backend).read(&path).expect("read VTK");

        assert_eq!(loaded.shape(), [2, 3, 4], "shape is [depth, row, col]");
        let spacing = loaded.spacing();
        assert!(
            (spacing[0] - 1.25).abs() < 1e-9,
            "spacing[0] is Δdepth and must be the file Z size 1.25, got {}",
            spacing[0]
        );
        assert!(
            (spacing[1] - 0.75).abs() < 1e-9,
            "spacing[1] is Δrow and must be the file Y size 0.75, got {}",
            spacing[1]
        );
        assert!(
            (spacing[2] - 0.5).abs() < 1e-9,
            "spacing[2] is Δcol and must be the file X size 0.5, got {}",
            spacing[2]
        );
    }

    /// The emitted `SPACING` line describes the same axis as `DIMENSIONS`.
    ///
    /// RITK shape `[2, 3, 4]` is emitted as `DIMENSIONS 4 3 2`, so X is the
    /// column axis and the X size must be the column spacing 0.5 — not the
    /// depth spacing 1.25. Emitting both index-for-index makes the file
    /// contradict itself: its dimensions name the column axis while its spacing
    /// gives that axis the depth pitch.
    #[test]
    fn the_emitted_spacing_line_names_the_same_axis_as_dimensions() {
        let backend = SequentialBackend;
        let image = Image::from_flat_on(
            vec![0.0f32; 24],
            [2, 3, 4],
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.25, 0.75, 0.5]),
            Direction::identity(),
            &backend,
        )
        .expect("native image");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("axes.vtk");

        VtkWriter::new(backend)
            .write(&path, &image)
            .expect("write VTK");

        let text = String::from_utf8_lossy(&std::fs::read(&path).expect("read back")).into_owned();
        assert!(
            text.contains("DIMENSIONS 4 3 2"),
            "shape [2, 3, 4] emits the column extent first, got:\n{text}"
        );
        assert!(
            text.contains("SPACING 0.5 0.75 1.25"),
            "the X size must be the column spacing 0.5, not the depth spacing 1.25, got:\n{text}"
        );
    }

    #[test]
    fn legacy_writer_rejects_non_identity_direction_before_creating_output() {
        let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let image = native_image(direction, CoordinateMap::Cartesian);
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("rotated.vtk");

        let error = VtkWriter::new(SequentialBackend)
            .write(&path, &image)
            .expect_err("legacy structured points cannot encode image direction");

        assert_eq!(
            error.to_string(),
            "legacy VTK structured points cannot preserve a non-identity direction matrix"
        );
        assert!(!path.exists(), "rejection must happen before file creation");
    }

    #[test]
    fn legacy_writer_rejects_non_cartesian_map_without_truncating_output() {
        let geometry = CurvilinearArray::centred(1.0e-4, 0.06, 0.5_f64.to_radians(), 129)
            .expect("valid curvilinear geometry");
        let image = native_image(
            Direction::identity(),
            CoordinateMap::CurvilinearArray(geometry),
        );
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("curvilinear.vtk");
        let original = b"preserve existing output";
        std::fs::write(&path, original).expect("create prior output");

        let error = VtkWriter::new(SequentialBackend)
            .write(&path, &image)
            .expect_err("legacy structured points cannot encode a non-Cartesian map");

        assert_eq!(
            error.to_string(),
            "legacy VTK structured points cannot preserve a non-Cartesian coordinate map"
        );
        assert_eq!(
            std::fs::read(&path).expect("read prior output"),
            original,
            "rejection must happen before truncating the destination"
        );
    }
}
