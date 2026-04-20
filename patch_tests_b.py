path = r"crates/ritk-cli/src/commands/segment.rs"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

last_brace = content.rfind("\n}")
assert last_brace != -1

part_b = """
    // -- threshold-level-set tests -----------------------------------------

    #[test]
    fn test_segment_threshold_level_set_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(5.0);
        args.upper_threshold = Some(250.0);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        assert!(output.exists(), "output file must exist");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "shape must be preserved");
    }

    #[test]
    fn test_segment_threshold_level_set_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(5.0);
        args.upper_threshold = Some(250.0);
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
    fn test_segment_threshold_level_set_missing_phi_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.lower_threshold = Some(5.0);
        args.upper_threshold = Some(250.0);
        let result = run(args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--initial-phi"));
    }
"""

content = content[:last_brace] + part_b + "\n}"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Part B done. {len(content)} chars, {content.count(chr(10))} lines")
