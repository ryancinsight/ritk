DQ = chr(34)
src = open("crates/ritk-cli/src/commands/mod.rs").read()
# Replace DICOM error arm with real call
old_arm = ("""        "dicom" => Err(anyhow!(""" + "\n" +
    """            "DICOM output is not supported: ritk-io currently provides read-only series loading. \\""" + "\n" +
    """             "Convert to NIfTI, MetaImage, or NRRD instead.""" + "\n" +
    """        )),""")
new_arm = ("""        """ + DQ + """dicom""" + DQ + """ => ritk_io::write_dicom_series::<Backend, _>(path, image)""" + "\n" +
    """            .with_context(|| format!(""" + DQ + """Failed to write DICOM series to: {}""" + DQ + """, path.display())),""")
if old_arm in src:
    src = src.replace(old_arm, new_arm)
    print("DICOM arm replaced")
else:
    print("WARNING: old arm not found, trying fallback")
    import re
    pattern = r"""        "dicom" => Err\(anyhow!\([^)]+\)\),"""
    replacement = new_arm
    src2 = re.sub(pattern, replacement, src, flags=re.DOTALL)
    if src2 != src:
        src = src2
        print("DICOM arm replaced via regex")
    else:
        print("WARNING: Could not replace DICOM arm")
# Remove the test_write_image_dicom_returns_err test
old_test = src[src.find("    #[test]\n    fn test_write_image_dicom_returns_err"):src.find("    #[test]\n    fn test_write_image_vtk_succeeds")]
if old_test:
    src = src.replace(old_test, "")
    print("old DICOM test removed")
# Add new DICOM test before the VTK test
vtk_test_marker = "    #[test]\n    fn test_write_image_vtk_succeeds"
new_test = ("""    #[test]""" + "\n" +
    """    fn test_write_image_dicom_creates_directory() {""" + "\n" +
    """        use burn::tensor::{Shape, Tensor, TensorData};""" + "\n" +
    """        use ritk_core::image::Image;""" + "\n" +
    """        use ritk_core::spatial::{Direction, Point, Spacing};""" + "\n" +
    """        let dir = tempfile::tempdir().unwrap();""" + "\n" +
    """        let out_path = dir.path().join(""" + DQ + """dicom_series""" + DQ + """);""" + "\n" +
    """        let device: <Backend as BurnBackend>::Device = Default::default();""" + "\n" +
    """        let td = TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));""" + "\n" +
    """        let tensor = Tensor::<Backend, 3>::from_data(td, &device);""" + "\n" +
    """        let image = Image::new(tensor, Point::new([0.0; 3]),""" + "\n" +
    """            Spacing::new([1.0; 3]), Direction::identity());""" + "\n" +
    """        let result = write_image(&out_path, &image, """ + DQ + """dicom""" + DQ + """);""" + "\n" +
    """        assert!(result.is_ok(), """ + DQ + """DICOM write must succeed: {:?}""" + DQ + """, result.err());""" + "\n" +
    """        assert!(out_path.is_dir(), """ + DQ + """DICOM output directory must exist""" + DQ + """);""" + "\n" +
    """    }""" + "\n\n")
src = src.replace(vtk_test_marker, new_test + vtk_test_marker)
open("crates/ritk-cli/src/commands/mod.rs", "w").write(src)
print("CLI mod.rs updated")
