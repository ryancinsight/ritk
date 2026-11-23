//! DICOM header de-identification and structural export checks.
//!
//! This example uses the `Enhanced` header profile to replace selected
//! attributes and UIDs and remove private tags, then checks export structure.
//! It opts into destructive zero-fill of native top-level Pixel Data. It does
//! not detect or selectively redact identifying image content. A file whose
//! Pixel Data cannot be cleared (for example, an encapsulated frame) is
//! reported as a failure and no output is written for that instance.
//!
//! These checks do not establish de-identification for sharing with a PACS
//! or conformance with PS 3.15 Annex E.
//!
//! The export gate checks, per file:
//! - the file can be reopened by the Part 10 parser;
//! - Study, Series, and SOP Instance UIDs are present and properly formatted,
//!   and a Frame of Reference UID is checked when present;
//! - an object holding pixel data carries Rows, Columns, and Bits Allocated,
//!   and native Pixel Data holds at least one frame of that size;
//! - the configured `prohibited_values` are not found in metadata.
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
    // The Enhanced profile applies header rules, consistent UID remapping,
    // and private-tag removal. Explicit native zero-fill below destroys the
    // image samples; use Skip when the image must remain usable and assess
    // visual privacy separately. Encapsulated Pixel Data is not cleared.
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
        anyhow::bail!(
            "{error_count} file(s) failed; do not use this batch. Successful files may remain"
        );
    }

    // ── 2. Verify the export ────────────────────────────────────────────────
    // Check the UID set, known prohibited values, and pixel geometry.
    // This is not a privacy review or a test of identifying image content.
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

    println!("Structural checks passed for {success_count} instances.");
    Ok(())
}
