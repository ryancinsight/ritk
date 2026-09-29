use super::*;
use crate::dicom::loader::tests::fixtures;
use ritk_io::{DicomObjectModel, DicomObjectNode, DicomTag};
use std::path::Path;
use tempfile::tempdir;

fn write_preview(root: &Path, uid: &str, modality: &str, size: [u16; 2], pixels: &[u8]) {
    assert_eq!(pixels.len(), usize::from(size[0]) * usize::from(size[1]));
    let mut model = DicomObjectModel::new();
    for (group, element, vr, value) in [
        (0x0008, 0x0016, "UI", "1.2.840.10008.5.1.4.1.1.7"),
        (0x0008, 0x0018, "UI", uid),
        (0x0008, 0x0060, "CS", modality),
        (0x0020, 0x000D, "UI", "2.25.928"),
        (0x0020, 0x000E, "UI", uid),
        (0x0020, 0x0032, "DS", "0\\0\\0"),
        (0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0"),
        (0x0028, 0x0004, "CS", "MONOCHROME2"),
        (0x0028, 0x0030, "DS", "1\\1"),
        (0x0018, 0x0050, "DS", "1"),
        (0x0028, 0x1050, "DS", "127.5"),
        (0x0028, 0x1051, "DS", "255"),
        (0x0028, 0x1056, "CS", "LINEAR_EXACT"),
    ] {
        model.insert(DicomObjectNode::text(
            DicomTag::new(group, element),
            vr,
            value,
        ));
    }
    for (element, value) in [
        (0x0002, 1),
        (0x0010, size[1]),
        (0x0011, size[0]),
        (0x0100, 8),
        (0x0101, 8),
        (0x0102, 7),
        (0x0103, 0),
    ] {
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, element),
            "US",
            value,
        ));
    }
    model.insert(DicomObjectNode::bytes(
        DicomTag::new(0x7FE0, 0x0010),
        "OB",
        pixels.to_vec(),
    ));
    let path = root.join(format!("{uid}.dcm"));
    ritk_io::write_dicom_object(&model, &path).expect("write preview instance");
}

fn grayscale(pixels: &[u8]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|&value| [value, value, value, 255])
        .collect()
}

#[test]
fn unopened_series_decode_lazily_and_depend_on_instance_pixels() {
    for modality in ["CT", "MR"] {
        let root = tempdir().expect("study root");
        let uid = "2.25.928.1";
        write_preview(root.path(), uid, modality, [4, 1], &[0, 64, 128, 255]);
        let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
        let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
        assert_eq!(browser.thumbnails.len(), 0);
        // A replacement after catalog construction must supply the preview.
        // Eager decode would instead retain the original ramp.
        write_preview(root.path(), uid, modality, [4, 1], &[255, 128, 64, 0]);
        let frame = browser.thumbnail(0).expect("decoded unopened series");
        assert_eq!((frame.width(), frame.height()), (4, 1));
        assert_eq!(frame.rgba(), grayscale(&[255, 128, 64, 0]));
        assert_eq!(
            browser.cached_thumbnail(0).expect("cached frame").rgba(),
            grayscale(&[255, 128, 64, 0])
        );
        assert_eq!(browser.thumbnail(1), None);
        assert_eq!(browser.thumbnails.len(), 1);
    }
}

#[test]
fn preview_preserves_dicom_rescale_voi_and_inversion() {
    for (photometric, expected) in [
        ("MONOCHROME2", [0, 64, 191, 255]),
        ("MONOCHROME1", [255, 191, 64, 0]),
    ] {
        let root = tempdir().expect("study root");
        fixtures::write_grayscale_presentation(root.path(), photometric, Some("LINEAR_EXACT"))
            .expect("write grayscale instance");
        let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
        let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
        // Modality values [-30,-10,10,30], center 0 and width 40 give
        // clipped intensities [0,1/4,3/4,1], rounded to 8-bit grayscale.
        assert_eq!(
            browser.thumbnail(0).expect("grayscale preview").rgba(),
            grayscale(&expected)
        );
    }
}

#[test]
fn preview_reads_only_first_instance() {
    let root = tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
    let paths = &browser.choice(0).expect("series").acquisition.file_paths;
    let volume = load_volume_from_dicom_instance(&paths[0]).expect("first instance");
    let mut app = SnapApp::default();
    app.load_volume(volume, "Synthetic preview.".into());
    let expected = PresentationFrame::from_slice(
        app.loaded.as_ref().expect("volume"),
        0,
        0,
        super::super::frame::window_level_for_app(&app),
        app.colormap,
    )
    .expect("reference frame");
    for path in paths.iter().skip(1) {
        std::fs::remove_file(path).expect("remove unused instance");
    }
    let actual = browser.thumbnail(0).expect("first instance preview");
    assert_eq!(actual, &expected);
}

#[test]
fn preview_keeps_pet_modality_colormap() {
    use iris::color::{ColorMap, Normalized};

    let root = tempdir().expect("study root");
    let pixels = [0, 64, 128, 255];
    write_preview(root.path(), "2.25.928.1", "PT", [4, 1], &pixels);
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
    let expected: Vec<u8> = pixels
        .into_iter()
        .flat_map(|value| {
            crate::render::NamedColorMap::Hot
                .sample(Normalized::from_u8(value))
                .to_rgba8()
        })
        .collect();
    let actual = browser.thumbnail(0).expect("PET preview");
    assert_eq!(actual.rgba(), expected);
    assert!(actual
        .rgba()
        .chunks_exact(4)
        .any(|pixel| pixel[0] != pixel[1] || pixel[1] != pixel[2]));
}

#[test]
fn unsupported_voi_is_unavailable_instead_of_a_sentinel_image() {
    let root = tempdir().expect("study root");
    fixtures::write_grayscale_presentation(root.path(), "MONOCHROME2", Some("POLYNOMIAL"))
        .expect("write unsupported VOI instance");
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
    assert_eq!(browser.thumbnail(0), None);
    assert_eq!(browser.thumbnails.len(), 1);
    assert_eq!(browser.thumbnails[0].index, 0);
}

#[test]
fn thumbnails_bound_dimensions_and_evict_least_recently_requested_frames() {
    let root = tempdir().expect("study root");
    let pixels: Vec<u8> = (0..80).flat_map(|_| 0..128).collect();
    for index in 0..27 {
        write_preview(
            root.path(),
            &format!("2.25.928.{index}"),
            "MR",
            [128, 80],
            &pixels,
        );
    }
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
    for index in 0..26 {
        let frame = browser.thumbnail(index).expect("preview");
        assert_eq!((frame.width(), frame.height()), (64, 40));
        let expected: Vec<u8> = (0..40)
            .flat_map(|_| (0..64).flat_map(|x| [x * 2, x * 2, x * 2, 255]))
            .collect();
        assert_eq!(frame.rgba(), expected);
        assert_eq!(frame.display_spacing().values(), [2.0, 2.0]);
    }
    assert_eq!(browser.thumbnails.len(), 26);
    assert_eq!(browser.thumbnail(0).expect("refresh oldest").width(), 64);
    assert_eq!(browser.thumbnail(26).expect("new preview").width(), 64);
    assert_eq!(browser.cached_thumbnail(1), None);
    assert_eq!(
        browser
            .cached_thumbnail(0)
            .expect("recent preview retained")
            .width(),
        64
    );
    let uid = browser
        .choice(1)
        .expect("evicted series")
        .acquisition
        .series_instance_uid()
        .to_owned();
    write_preview(root.path(), &uid, "MR", [4, 1], &[0, 255, 255, 0]);
    assert_eq!(
        browser.thumbnail(1).expect("reload evicted preview").rgba(),
        grayscale(&[0, 255, 255, 0])
    );
    assert_eq!(browser.thumbnails.len(), 26);
    let retained_bytes: usize = browser
        .thumbnails
        .iter()
        .filter_map(|item| item.frame.as_ref())
        .map(|frame| frame.rgba().len())
        .sum();
    assert!(retained_bytes <= 26 * 64 * 64 * 4);
}

#[test]
fn unavailable_previews_are_cached_and_retried_after_eviction() {
    let root = tempdir().expect("study root");
    for index in 0..27 {
        write_preview(
            root.path(),
            &format!("2.25.928.{index}"),
            "CT",
            [2, 1],
            &[0, 255],
        );
    }
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");
    let choice = browser.choice(0).expect("first series");
    let uid = choice.acquisition.series_instance_uid().to_owned();
    let path = choice.acquisition.file_paths[0].clone();
    std::fs::write(&path, b"invalid DICOM").expect("replace unreadable instance");
    assert_eq!(browser.thumbnail(0), None);
    assert_eq!(browser.thumbnails.len(), 1);
    write_preview(root.path(), &uid, "CT", [2, 1], &[255, 0]);
    assert_eq!(browser.thumbnail(0), None);
    for index in 1..27 {
        assert_eq!(
            browser.thumbnail(index).expect("available series").rgba(),
            grayscale(&[0, 255])
        );
        assert!(browser.thumbnails.len() <= 26);
    }
    assert_eq!(
        browser.thumbnail(0).expect("retry after eviction").rgba(),
        grayscale(&[255, 0])
    );
    assert_eq!(browser.thumbnails.len(), 26);
}

#[test]
fn browser_retains_every_series_and_selects_by_uid() {
    let root = tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    fixtures::write_study(root.path(), "MR", "2.25.20260905002").expect("write MR series");
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");

    let browser = SeriesBrowser::from_tree(&tree, Some("2.25.20260905002")).expect("browser");

    assert_eq!(browser.len(), 2);
    assert_eq!(browser.study_count(), 1);
    assert_eq!(tree.patients.len(), 1);
    assert_eq!(tree.patients[0].patient_name, "FIXTURE^PATIENT");
    assert_eq!(tree.patients[0].studies.len(), 1);
    assert_eq!(
        tree.patients[0].studies[0].study_uid.as_deref(),
        Some("2.25.20260905")
    );
    assert_eq!(
        tree.patients[0].studies[0].study_date.as_deref(),
        Some("20260905")
    );
    assert_eq!(browser.active_index(), 1);
    assert_eq!(
        browser.choice(0).expect("first series").modality.as_ref(),
        "CT"
    );
    assert_eq!(
        browser.choice(1).expect("second series").modality.as_ref(),
        "MR"
    );
    assert_eq!(
        browser
            .choice(1)
            .expect("second series")
            .description
            .as_ref(),
        "Series 2"
    );
    assert_eq!(browser.choice(1).expect("second series").image_count, 3);
}

#[test]
fn browser_rejects_a_series_uid_outside_the_discovered_study() {
    let root = tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");

    let result = SeriesBrowser::from_tree(&tree, Some("2.25.999"));

    assert_eq!(
        result.err().map(|error| error.to_string()),
        Some("selected SeriesInstanceUID is absent from study".into())
    );
}

#[test]
fn active_series_stays_visible_after_selection_and_scroll_clamps() {
    let root = tempdir().expect("study root");
    for index in 1..=4 {
        fixtures::write_study(root.path(), "MR", &format!("2.25.2026090500{index}"))
            .expect("write series");
    }
    let tree = crate::dicom::loader::scan_folder_for_series(root.path()).expect("scan study");
    let mut browser = SeriesBrowser::from_tree(&tree, None).expect("browser");

    assert!(browser.scroll_series(2, 2));
    assert_eq!(browser.first_visible(), 2);
    assert!(browser.set_active(0));
    assert_eq!(browser.first_visible(), 0);
    assert!(browser.set_active(3));
    assert_eq!(browser.first_visible(), 3);
    assert!(browser.scroll_series(1, 2));
    assert_eq!(browser.first_visible(), 2);
    assert!(browser.scroll_series(-2, 2));
    assert_eq!(browser.first_visible(), 0);
}
