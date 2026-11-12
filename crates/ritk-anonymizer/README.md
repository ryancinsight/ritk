# RITK DICOM Metadata Editor

Windows-only local desktop application using RITK for DICOM processing and the
Métis WebView2 host for the form. The installed Microsoft WebView2 runtime is
required. It is a metadata **pseudonymization** tool, not an anonymity certificate.

## Run

```powershell
cargo run --locked -p ritk-anonymizer --release
```

Select a folder containing only DICOM instances (a DICOMDIR index is skipped),
then select the parent for a **new** `RITK-anonymized` folder. Existing output is
never overwritten. Enter replacement fields, acknowledge the privacy-review
notice and export. Blank optional fields clear their corresponding tags.
The source files remain unchanged. Export runs off the window thread, rejects
repeated commands while busy, and waits for an in-flight export if the window
closes. Cancel closes the editor before export; it does not interrupt an active
batch. A crash can leave an `INCOMPLETE` marker: do not use such output.

| Field | DICOM tag | Value representation |
|---|---|---|
| Patient ID | (0010,0020) | LO |
| Patient name | (0010,0010) | PN |
| Sex | (0010,0040) | CS: M, F, O or empty |
| Age in years | (0010,1010) | AS: three digits + Y |
| Birth date | (0010,0030) | DA |
| Height in cm | (0010,1020) | DS converted to metres |
| Weight in kg | (0010,1030) | DS |
| Study date | (0008,0020) | DA |
| Study time | (0008,0030) | TM |
| Study ID | (0020,0010) | SH |
| Accession number | (0008,0050) | SH |

Text replacement fields currently accept printable ASCII only. Dates and times
are validated by the backend, not only by the form. Export applies RITK's
Enhanced metadata profile, then the explicit replacement values. Each batch
uses a fresh random UID salt; identical UIDs within a batch remain consistent.
Outputs use neutral filenames and are reparsed to check replacement values,
SOP identity and unchanged pixel content. The original directory layout and
DICOMDIR are not copied.

## Privacy limits

Pixel data is preserved. Burned-in text, recognizable anatomy, residual standard
attributes and values deliberately entered into this form may identify people.
The application marks Patient Identity Removed as NO rather than asserting full
PS 3.15 conformance. No study is uploaded and no network service is started.
Do not share exports until they have undergone the required privacy review.

## Acceptance checks

- `cargo test -p ritk-io --lib format::dicom::anonymize::editor`
- `cargo check --locked -p ritk-anonymizer --target x86_64-pc-windows-msvc`
- On Windows: select/cancel both dialogs, edit all fields, reject invalid values,
  export synthetic nested instances, reopen outputs, verify original hashes,
  reject an existing destination, repeat Export, close during a batch and inspect
  completion. Check keyboard focus, scrolling and resize at 125%/150% display DPI.

A macOS check cannot qualify the Windows-only host. Native Windows acceptance
must pass before delivering a usable Windows build.

The PR-triggered or manually dispatched `DICOM editor Windows candidate` workflow builds an
unreleased Windows artifact and runs backend tests on Windows. It performs no
release, deployment or package publication. Its executable still requires native
interaction/privacy acceptance before use with clinical data.
