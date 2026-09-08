# ritk-dicom

DICOM image I/O and DIMSE networking for [RITK](https://github.com/ryancinsight/ritk).

Single source of truth for DICOM transfer-syntax classification and pixel-codec
contracts. `dicom-rs` supplies the dataset, metadata, and external-codec adapter
layer; the pixel codecs themselves are RITK-native (`ritk-codecs`).

Native decode covers uncompressed little-endian pixels, RLE Lossless, grayscale
JPEG Baseline / Extended / Lossless, grayscale JPEG-LS, JPEG 2000, and JPEG XL.
No supported DICOM pixel path requires a C or C++ codec library.

The crate also provides DIMSE association handling for PACS SCU/SCP workflows.

Untrusted Part 10 input can be checked before object materialization with
`validate_part10` or the `BoundedDicomParseBackend` helpers. They apply the
Atlas `ParseBudget` byte, structural-element, and nesting ceilings while
preserving the existing dicom-rs object contract:

```rust
use ritk_dicom::{parse_bytes_with_budget, DicomRsBackend, ParseBudget};

let object = parse_bytes_with_budget::<DicomRsBackend>(&bytes, &ParseBudget::DEFAULT)?;
# Ok::<(), anyhow::Error>(())
```

The structural preflight covers implicit little-endian, explicit little-endian
and explicit big-endian datasets, including encapsulated pixel fragments. The
current dicom-rs lock does not enable a deflate dataset decoder; deflated input
is rejected with an explicit error until a bounded decoder is available.

## Usage

```toml
[dependencies]
ritk-dicom = "0.2.0"
```
