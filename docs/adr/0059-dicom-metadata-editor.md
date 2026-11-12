# ADR 0059: DICOM metadata editor with copy-only export

- Status: Proposed

## Context

The requested fields mirror the existing DICOM Editor 1.0.0.7 screenshots.
RITK owns tag validation, profile application and serialization. Métis supplies
the Windows WebView2 host and native directory selection. The packaged page
contains no remote resources and cannot submit arbitrary paths: the backend
retains only paths selected with native dialogs. No patient values from the
reference photographs are test fixtures or defaults.

The app creates a new destination, never updates sources, uses neutral output
names and fresh batch UID salt, reparses every output, and preserves the pixel
payload. A worker keeps export off the UI thread; all host exits join it so
closing does not interrupt a partially written batch. Repeated export commands
are rejected while one worker is active. Failed batches remove their owned
files; failure to clean leaves an incomplete marker and an explicit error.

Operator-entered demographics, residual standard metadata, burned-in text and
recognizable anatomy prevent a general anonymity claim. The UI calls this
metadata pseudonymization, and the data set marks PatientIdentityRemoved NO.
A dedicated all-content deidentification profile and pixel review workflow is
not silently implied by this editor.

Verification combines field validation and synthetic serialized DICOM tests,
source-hash preservation, rejection/cleanup tests, a Windows compile check and
native Windows UI acceptance. A macOS build is insufficient to qualify the
Windows host; no production release occurs before that acceptance.
