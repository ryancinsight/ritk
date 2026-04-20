path = r"crates/ritk-cli/src/commands/segment.rs"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

last_brace = content.rfind("\n}")
assert last_brace != -1

part_a = """
    // -- Shape-detection: phi signed-distance helper -------------------------

    fn make_phi_sphere(dims: [usize; 3], center: [f64; 3], radius: f64) -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let mut data = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dist = ((iz as f64 - center[0]).powi(2)
                        + (iy as f64 - center[1]).powi(2)
                        + (ix as f64 - center[2]).powi(2))
                    .sqrt();
                    data[iz * ny * nx + iy * nx + ix] = (dist - radius) as f32;
                }
            }
        }
        let td = TensorData::new(data, Shape::new(dims));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // -- shape-detection tests ----------------------------------------------

    #[test]
    fn test_segment_shape_detection_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "shape-detection");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        assert!(output.exists(), "output file must exist");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "shape must be preserved");
    }

    #[test]
    fn test_segment_shape_detection_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "shape-detection");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
        }
    }

    #[test]
    fn test_segment_shape_detection_missing_phi_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let args = default_args(input.clone(), output.clone(), "shape-detection");
        let result = run(args);
        assert!(result.is_err(), "--initial-phi missing must produce error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("--initial-phi"), "error must mention --initial-phi, got: {msg}");
    }
"""

content = content[:last_brace] + part_a + "\n}"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Part A done. {len(content)} chars, {content.count(chr(10))} lines")
