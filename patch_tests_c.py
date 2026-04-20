path = r"crates/ritk-cli/src/commands/segment.rs"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

last_brace = content.rfind("\n}")
assert last_brace != -1

part_c = """
    #[test]
    fn test_segment_threshold_level_set_missing_lower_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0)).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.upper_threshold = Some(250.0);
        let result = run(args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--lower-threshold"));
    }

    #[test]
    fn test_segment_threshold_level_set_missing_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0)).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(5.0);
        let result = run(args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--upper-threshold"));
    }

    #[test]
    fn test_segment_threshold_level_set_lower_gt_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0)).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(250.0);
        args.upper_threshold = Some(5.0);
        let result = run(args);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must be") && msg.contains("<="),
            "error must explain bound constraint, got: {msg}"
        );
    }
"""

content = content[:last_brace] + part_c + "\n}"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Part C done. {len(content)} chars, {content.count(chr(10))} lines")
