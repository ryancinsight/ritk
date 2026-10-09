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
mod structured_points;
pub(crate) mod xml_helpers;
pub mod xml_write_attr;

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

    const MAX_GENERATED_FILE_BYTES: usize = 4 * 1024;

    fn scalar_header(
        dimensions: [usize; 3],
        point_data_count: usize,
        encoding: &str,
        scalar_type: &str,
        component_count: Option<&str>,
    ) -> String {
        let component_field = component_count.map_or(String::new(), |count| format!(" {count}"));
        format!(
            "# vtk DataFile Version 3.0\nfixture\n{encoding}\nDATASET STRUCTURED_POINTS\nDIMENSIONS {} {} {}\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA {point_data_count}\nSCALARS scalars {scalar_type}{component_field}\nLOOKUP_TABLE default\n",
            dimensions[0], dimensions[1], dimensions[2]
        )
    }

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
        let direction = crate::domain::axis_order::vtk_image_direction();
        let image =
            Image::from_flat_on(values.clone(), shape, origin, spacing, direction, &backend)
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
        let (_, file_dims, file_origin, file_spacing) =
            read_vtk_flat(&path).expect("read VTK fields");
        assert_eq!(file_dims, [shape[2], shape[1], shape[0]]);
        assert_eq!(file_origin, [origin[0], origin[1], origin[2]]);
        assert_eq!(file_spacing, [1.25, 0.75, 0.5]);
        assert_eq!(*loaded.direction(), direction);
    }

    #[test]
    fn legacy_writer_rejects_unrepresentable_direction_before_creating_output() {
        let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let image = native_image(direction, CoordinateMap::Cartesian);
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("rotated.vtk");

        let error = VtkWriter::new(SequentialBackend)
            .write(&path, &image)
            .expect_err("legacy structured points cannot encode this direction");

        assert_eq!(
            error.to_string(),
            "legacy VTK structured points cannot preserve a direction matrix outside the VTK-aligned ZYX-to-XYZ axis order"
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

    #[test]
    fn legacy_writer_rejects_invalid_spacing_before_touching_output() {
        let mut spacing = Spacing::new([0.5, 0.75, 1.25]);
        spacing[1] = 0.0;
        let image = Image::from_flat_on(
            vec![1.25, -4.5],
            [1, 1, 2],
            Point::new([1.0, -2.0, 3.5]),
            spacing,
            crate::domain::axis_order::vtk_image_direction(),
            &SequentialBackend,
        )
        .expect("image data may carry malformed spacing for writer validation");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("invalid-spacing.vtk");
        let original = b"preserve existing output";
        std::fs::write(&path, original).expect("create prior output");

        let error = VtkWriter::new(SequentialBackend)
            .write(&path, &image)
            .expect_err("legacy VTK spacing must be finite and positive");

        assert_eq!(
            error.to_string(),
            "legacy VTK structured points requires finite, positive SPACING at axis 1; got 0"
        );
        assert_eq!(
            std::fs::read(&path).expect("read prior output"),
            original,
            "rejection must happen before truncating the destination"
        );

        let absent = directory.path().join("invalid-spacing-new.vtk");
        let error = VtkWriter::new(SequentialBackend)
            .write(&absent, &image)
            .expect_err("invalid spacing must fail for a new destination");
        assert_eq!(
            error.to_string(),
            "legacy VTK structured points requires finite, positive SPACING at axis 1; got 0"
        );
        assert!(
            !absent.exists(),
            "rejection must happen before file creation"
        );
    }

    #[test]
    fn legacy_writer_rejects_zero_dimensions_before_touching_output() {
        let image = Image::from_flat_on(
            Vec::new(),
            [0, 1, 1],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            crate::domain::axis_order::vtk_image_direction(),
            &SequentialBackend,
        )
        .expect("empty tensors can represent zero-sized image dimensions");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("zero-dimensions.vtk");
        let original = b"preserve existing output";
        std::fs::write(&path, original).expect("create prior output");

        let error = VtkWriter::new(SequentialBackend)
            .write(&path, &image)
            .expect_err("legacy structured points dimensions must be positive");

        assert_eq!(
            error.to_string(),
            "legacy VTK structured points dimensions must be positive, got [1, 1, 0]"
        );
        assert_eq!(
            std::fs::read(&path).expect("read prior output"),
            original,
            "rejection must happen before truncating the destination"
        );

        let absent = directory.path().join("zero-dimensions-new.vtk");
        let error = VtkWriter::new(SequentialBackend)
            .write(&absent, &image)
            .expect_err("zero-sized dimensions must fail for new destinations");
        assert_eq!(
            error.to_string(),
            "legacy VTK structured points dimensions must be positive, got [1, 1, 0]"
        );
        assert!(
            !absent.exists(),
            "rejection must happen before file creation"
        );
    }

    #[test]
    fn legacy_reader_returns_an_error_for_nonpositive_spacing() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("invalid-spacing.vtk");
        let header = b"# vtk DataFile Version 3.0\nfixture\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 1 1 1\nORIGIN 0 0 0\nSPACING 1 0 1\nPOINT_DATA 1\nSCALARS scalars float 1\nLOOKUP_TABLE default\n";
        std::fs::write(&path, header).expect("write malformed-spacing header");

        let flat_error = read_vtk_flat(&path)
            .expect_err("flat reader validates spacing before reading a payload");
        assert!(
            format!("{flat_error:#}")
                .contains("legacy VTK structured points requires finite, positive SPACING"),
            "unexpected flat reader error: {flat_error:#}"
        );

        let error = VtkReader::new(SequentialBackend)
            .read(&path)
            .expect_err("nonpositive spacing is malformed geometry");

        assert!(
            format!("{error:#}")
                .contains("VTK SPACING components must be finite and strictly positive"),
            "unexpected parser error: {error:#}"
        );
    }

    #[test]
    fn legacy_reader_rejects_zero_dimensions_before_reading_payload() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("zero-dimensions.vtk");
        let header = b"# vtk DataFile Version 3.0\nfixture\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 0 1 1\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA 0\nSCALARS scalars float 1\nLOOKUP_TABLE default\n";
        std::fs::write(&path, header).expect("write zero-dimension header");

        let error =
            read_vtk_flat(&path).expect_err("zero dimensions are invalid before payload decoding");

        assert!(
            format!("{error:#}")
                .contains("legacy VTK structured points dimensions must be positive"),
            "unexpected reader error: {error:#}"
        );
    }

    #[test]
    fn legacy_reader_rejects_unrepresentable_f32_count_before_payload_read() {
        let maximum = usize::try_from(isize::MAX).expect("positive pointer maximum fits usize");
        let count = maximum
            .checked_div(std::mem::size_of::<f32>())
            .expect("f32 has a nonzero size")
            .checked_add(1)
            .expect("one value beyond the allocation limit fits usize");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("unrepresentable-sample-count.vtk");
        let header = scalar_header([count, 1, 1], count, "ASCII", "float", Some("1"));
        std::fs::write(&path, header).expect("write unrepresentable sample-count header");

        let error = read_vtk_flat(&path).expect_err("unrepresentable Vec layout must be rejected");

        assert!(
            format!("{error:#}").contains("cannot fit a Vec<f32> allocation"),
            "unexpected parser error: {error:#}"
        );
    }

    #[test]
    fn legacy_reader_rejects_unrepresentable_binary_payload_before_read() {
        let maximum = usize::try_from(isize::MAX).expect("positive pointer maximum fits usize");
        let count = maximum
            .checked_div(std::mem::size_of::<f64>())
            .expect("f64 has a nonzero size")
            .checked_add(1)
            .expect("one value beyond the allocation limit fits usize");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("unrepresentable-binary-payload.vtk");
        let header = scalar_header([count, 1, 1], count, "BINARY", "double", Some("1"));
        std::fs::write(&path, header).expect("write unrepresentable binary header");

        let error =
            read_vtk_flat(&path).expect_err("unrepresentable payload layout must be rejected");

        assert!(
            format!("{error:#}").contains("cannot fit a Vec<u8> allocation"),
            "unexpected parser error: {error:#}"
        );
    }

    #[test]
    fn legacy_reader_rejects_invalid_or_multicomponent_scalars() {
        for (component_count, expected_error) in [
            ("0", "unsupported VTK SCALARS component count"),
            ("2", "unsupported VTK SCALARS component count"),
            ("many", "bad SCALARS component count"),
        ] {
            let directory = tempdir().expect("temporary directory");
            let path = directory.path().join("invalid-scalar-components.vtk");
            let mut contents = scalar_header([2, 1, 1], 2, "ASCII", "float", Some(component_count));
            contents.push_str("10 11 20 21\n");
            std::fs::write(&path, contents).expect("write scalar component-count fixture");

            let error = read_vtk_flat(&path)
                .expect_err("unsupported scalar components must not be truncated");

            assert!(
                format!("{error:#}").contains(expected_error),
                "unexpected parser error for component count {component_count}: {error:#}"
            );
        }
    }

    #[test]
    fn legacy_reader_rejects_scalar_keywords_with_trailing_text() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("invalid-scalar-keyword.vtk");
        let contents = scalar_header([2, 1, 1], 2, "ASCII", "float", Some("1"))
            .replace("SCALARS scalars", "SCALARS_BROKEN scalars")
            + "10 20\n";
        std::fs::write(&path, contents).expect("write invalid scalar keyword fixture");

        let error =
            read_vtk_flat(&path).expect_err("a prefixed keyword must not be accepted as SCALARS");

        assert!(
            format!("{error:#}").contains("VTK header missing SCALARS"),
            "unexpected parser error: {error:#}"
        );
    }

    #[test]
    fn legacy_reader_defaults_omitted_scalar_component_count_to_one() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("implicit-scalar-component.vtk");
        let mut contents = scalar_header([2, 1, 1], 2, "ASCII", "float", None);
        contents.push_str("10 20\n");
        std::fs::write(&path, contents).expect("write single-component fixture");

        let (values, dimensions, origin, spacing) =
            read_vtk_flat(&path).expect("omitted component count defaults to one");

        assert_eq!(values, [10.0, 20.0]);
        assert_eq!(dimensions, [2, 1, 1]);
        assert_eq!(origin, [0.0; 3]);
        assert_eq!(spacing, [1.0; 3]);
    }

    proptest::proptest! {
        #[test]
        fn legacy_reader_handles_bounded_arbitrary_bytes_without_panicking(
            bytes in proptest::collection::vec(
                proptest::prelude::any::<u8>(),
                0..=MAX_GENERATED_FILE_BYTES
            )
        ) {
            let directory = tempdir().expect("temporary directory");
            let path = directory.path().join("arbitrary.vtk");
            std::fs::write(&path, bytes).expect("write generated parser input");

            if let Ok((values, dimensions, _origin, spacing)) = read_vtk_flat(&path) {
                let sample_count = dimensions
                    .iter()
                    .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension));
                assert!(dimensions.iter().all(|dimension| *dimension > 0));
                assert_eq!(sample_count, Some(values.len()));
                assert!(spacing.iter().all(|value| value.is_finite() && *value > 0.0));
            }
        }
    }
}
