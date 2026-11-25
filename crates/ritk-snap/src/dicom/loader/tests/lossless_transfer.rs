//! Cross-encoder comparison using the four MR_small Part 10 files bundled by pydicom.

use super::super::{load_volume_from_bytes, load_volume_from_path};
use crate::presentation::PresentationFrame;
use crate::render::{NamedColorMap, WindowLevel};
use std::path::PathBuf;

#[test]
#[ignore = "requires RITK_PYDICOM_TEST_FILES pointing to pydicom/data/test_files"]
fn independent_lossless_mr_encodings_preserve_every_voxel_and_displayed_sample() {
    let fixtures =
        PathBuf::from(std::env::var_os("RITK_PYDICOM_TEST_FILES").expect("pydicom fixtures"));
    // These are alternative encodings of one acquisition. The file loader
    // discovers sibling instances, so give each encoding its own series folder.
    let load_file = |filename: &str, bytes: &[u8]| {
        let series = tempfile::tempdir().expect("isolated series folder");
        let instance = series.path().join(filename);
        std::fs::write(&instance, bytes).expect("stage the packaged instance");
        load_volume_from_path(&instance).expect("load isolated series")
    };
    let raw = load_file(
        "MR_small.dcm",
        &std::fs::read(fixtures.join("MR_small.dcm")).expect("uncompressed reference"),
    );
    assert_eq!(raw.shape, [1, 64, 64]);
    assert_eq!(raw.data[..4], [905., 1019., 1227., 1259.]);
    let render = |volume: &crate::LoadedVolume| {
        PresentationFrame::from_slice(
            volume,
            0,
            0,
            WindowLevel::new(600., 1200.),
            NamedColorMap::Grayscale,
        )
        .expect("render MR frame")
    };
    let reference = render(&raw);
    for filename in [
        "MR_small_RLE.dcm",
        "MR_small_jpeg_ls_lossless.dcm",
        "MR_small_jp2klossless.dcm",
    ] {
        let path = fixtures.join(filename);
        let bytes = std::fs::read(&path).expect("packaged compressed fixture");
        let from_file = load_file(filename, &bytes);
        let from_bytes = load_volume_from_bytes(filename, &bytes).expect("read compressed bytes");
        for volume in [from_file, from_bytes] {
            assert_eq!(volume.shape, raw.shape, "{filename}");
            assert_eq!(volume.spacing, raw.spacing, "{filename}");
            assert_eq!(volume.direction, raw.direction, "{filename}");
            assert_eq!(volume.origin, raw.origin, "{filename}");
            assert_eq!(volume.data.as_slice(), raw.data.as_slice(), "{filename}");
            assert_eq!(render(&volume), reference, "{filename}");
        }
    }
}
