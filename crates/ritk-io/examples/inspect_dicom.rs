use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom_core::header::Header;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: inspect_dicom <file.dcm>");
    let obj = open_file(&path)?;

    println!("=== DICOM Header Dump ===");
    println!("Transfer Syntax: {:?}", obj.meta().transfer_syntax());

    for &tag in &[
        tags::PATIENT_ID,
        tags::STUDY_INSTANCE_UID,
        tags::SERIES_INSTANCE_UID,
        tags::SOP_INSTANCE_UID,
        tags::MODALITY,
        tags::IMAGE_ORIENTATION_PATIENT,
        tags::IMAGE_POSITION_PATIENT,
        tags::PIXEL_SPACING,
        tags::SLICE_THICKNESS,
        tags::SLICE_LOCATION,
        tags::ROWS,
        tags::COLUMNS,
        tags::BITS_ALLOCATED,
        tags::BITS_STORED,
        tags::HIGH_BIT,
        tags::SAMPLES_PER_PIXEL,
        tags::PHOTOMETRIC_INTERPRETATION,
        tags::RESCALE_INTERCEPT,
        tags::RESCALE_SLOPE,
        tags::WINDOW_CENTER,
        tags::WINDOW_WIDTH,
        tags::SERIES_DESCRIPTION,
        tags::STUDY_DESCRIPTION,
    ] {
        if let Ok(elem) = obj.element(tag) {
            let val = elem.to_str().unwrap_or_else(|_| "<error>".into());
            println!("  {:?}: {}", tag, val);
        } else {
            println!("  {:?}: <missing>", tag);
        }
    }

    // Also dump first 20 elements in order
    println!("\n=== First 20 elements ===");
    for (i, elem) in obj.iter().take(20).enumerate() {
        let tag = elem.tag();
        let val = elem.to_str().unwrap_or_else(|_| "<error>".into());
        println!("  [{:2}] {:?} = {}", i, tag, val);
    }

    Ok(())
}
