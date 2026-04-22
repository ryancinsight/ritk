import pathlib

p = pathlib.Path('crates/ritk-cli/src/commands/resample.rs')
content = p.read_text(encoding='utf-8')

old = '''    #[test]
    fn test_resample_half_spacing_doubles_size() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nii");
        let output = dir.path().join("out.nii");
        write_test_nifti(&input, vec![1.0f32; 64], [4, 4, 4], [2.0, 2.0, 2.0]);
        let args = ResampleArgs {
            input: input.clone(), output: output.clone(),
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "linear".to_string(),
        };
        run(args).unwrap();
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let loaded = ritk_io::read_nifti::<Backend, _>(&output, &dev).unwrap();
        // E_d = 4 * 2.0 = 8; new size = round(8/1.0) = 8
        assert_eq!(loaded.shape(), [8, 8, 8], "halving spacing doubles voxel count");
    }'''

new = '''    #[test]
    fn test_resample_half_spacing_doubles_size() {
        // Uses NRRD: NIfTI writer does not persist sform/pixdim, so spacing is lost on
        // NIfTI round-trip.  NRRD preserves all spatial metadata.
        //
        // Physical extent: E_d = 4 * 2.0 = 8.0
        // New size: N_d = round(E_d / 1.0) = 8
        let dir = tempdir().unwrap();
        let input = dir.path().join("in.nrrd");
        let output = dir.path().join("out.nrrd");
        let dev: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 64], Shape::new([4, 4, 4]));
        let t = Tensor::<Backend, 3>::from_data(td, &dev);
        let img = Image::new(
            t,
            Point::new([0.0; 3]),
            Spacing::new([2.0, 2.0, 2.0]),
            Direction::identity(),
        );
        ritk_io::write_nrrd::<Backend, _>(&input, &img).expect("write_nrrd must succeed");
        let args = ResampleArgs {
            input: input.clone(), output: output.clone(),
            spacing: "1.0,1.0,1.0".to_string(), interpolation: "linear".to_string(),
        };
        run(args).unwrap();
        let loaded = ritk_io::read_nrrd::<Backend, _>(&output, &dev).unwrap();
        assert_eq!(loaded.shape(), [8, 8, 8], "halving spacing must double voxel count");
    }'''

if old in content:
    content = content.replace(old, new)
    p.write_text(content, encoding='utf-8')
    print("patched ok")
else:
    print("ERROR: old text not found")
    print("snippet:", repr(content[content.find("test_resample_half"):content.find("test_resample_half")+200]))
