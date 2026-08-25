//! PACS-safe DICOM anonymization with export verification.
//!
//! This example demonstrates the recommended export pipeline for
//! de-identified studies bound for a PACS: anonymize every DICOM file in an
//! input directory with the `Enhanced` profile (full PS 3.15 Annex E
//! confidentiality, deterministic UID replacement, private-tag removal), then
//! run the export gate so corrupted or leaking metadata fails closed instead
//! of shipping.
//!
//! The export gate checks, per file:
//! - the file re-parses as a conformant Part 10 DICOM object;
//! - Study/Series/SOPInstance/FrameOfReference UIDs are present and conformant;
//! - Rows/Columns/BitsAllocated match the PixelData payload;
//! - none of the `prohibited_values` (the raw identifiers that were scrubbed)
//!   appear anywhere in the object.
//!
//! And across the directory:
//! - exactly one StudyInstanceUID and one SeriesInstanceUID for the set;
//! - every SOPInstanceUID is unique.
//!
//! Usage:
//!   cargo run -p ritk-io --example anonymize_pacs_export -- <input_dir> <output_dir>
//!
//! Example:
//!   cargo run -p ritk-io --example anonymize_pacs_export -- "raw/study_001" "anon/study_001"
#![expect(clippy::print_stderr, reason = "ratchet RITK-LINT-1")]
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use ritk_io::format::dicom::anonymize::verify::{verify_dicom_directory, VerifyOptions};
use ritk_io::format::dicom::{
    anonymize_dicom_directory, AnonymizationProfile, AnonymizeOptions, AnonymizeStats,
    CleaningPolicy,
};
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: {} <input_dir> <output_dir>\nExample: {} \"raw/study_001\" \"anon/study_001\"",
            args[0], args[0]
        );
        std::process::exit(1);
    }
    let input_dir = &args[1];
    let output_dir = &args[2];

    // ── 1. Anonymize ───────────────────────────────────────────────────────
    // The Enhanced profile removes/replaces every PS 3.15 Annex E attribute,
    // deterministically remaps all UIDs (referentially consistent within the
    // batch), clears date/time/description fields, and removes private tags.
    // Pixel data is cleaned too so visual PHI does not survive.
    let options = AnonymizeOptions {
        profile: AnonymizationProfile::Enhanced,
        patient_name: "ANONYMOUS".to_owned(),
        patient_id: "ANON001".to_owned(),
        uid_salt: env::var("RITK_ANON_SALT").unwrap_or_else(|_| "ritk-example-salt".to_owned()),
        clean_pixel_data: CleaningPolicy::Clean,
        clean_private_tags: CleaningPolicy::Clean,
    };

    println!("Anonymizing {input_dir} -> {output_dir} (Enhanced profile)");
    let AnonymizeStats {
        file_count,
        success_count,
        error_count,
        errors,
    } = anonymize_dicom_directory(input_dir, output_dir, &options)?;
    println!("  files={file_count} ok={success_count} errors={error_count}");
    for (path, message) in &errors {
        eprintln!("  error {path:?}: {message}");
    }
    if error_count > 0 {
        anyhow::bail!("{error_count} file(s) failed to anonymize; export aborted");
    }

    // ── 2. Verify the export ────────────────────────────────────────────────
    // The standard PACS anonymization failure is silent metadata corruption:
    // a divergent Study/Series UID breaks reassembly at the destination, a
    // leftover identifier re-identifies the patient, and a geometry mismatch
    // corrupts quantitative analysis. The gate catches all of these.
    let verify_options = VerifyOptions {
        require_uids: true,
        validate_uid_format: true,
        validate_pixel_geometry: true,
        // Belt-and-suspenders: the exact raw identifiers the pipeline expects
        // to have scrubbed. Add any that your source studies carry.
        prohibited_values: vec!["Doe^John".to_owned(), "PAT001".to_owned()],
    };

    println!("Verifying export metadata integrity in {output_dir}");
    let report = verify_dicom_directory(output_dir, &verify_options)?;
    println!(
        "  verified={} clean={} failing={}",
        report.file_count, report.clean_count, report.failing_count
    );

    if !report.is_clean() {
        for file in &report.files {
            eprintln!("  FAIL {}", file.path.display());
            for issue in &file.issues {
                eprintln!("    - {issue}");
            }
        }
        anyhow::bail!(
            "export verification failed: {} of {} files have metadata defects",
            report.failing_count,
            report.file_count
        );
    }

    println!("Export verified clean: {success_count} instances ready for PACS ingest.");
    Ok(())
}
