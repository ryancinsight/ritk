# ritk-dicom

DICOM image I/O and DIMSE networking for [RITK](https://github.com/ryancinsight/ritk).

Single source of truth for DICOM transfer-syntax classification and pixel-codec
contracts. `dicom-rs` supplies the dataset, metadata, and external-codec adapter
layer; the pixel codecs themselves are RITK-native (`ritk-codecs`).

Native decode covers uncompressed little-endian pixels, RLE Lossless, grayscale
JPEG Baseline / Extended / Lossless, grayscale JPEG-LS, JPEG 2000, and JPEG XL.
No supported DICOM pixel path requires a C or C++ codec library.

The crate also provides DIMSE association handling for PACS SCU/SCP workflows.

## Usage

```toml
[dependencies]
ritk-dicom = "0.2.0"
```
