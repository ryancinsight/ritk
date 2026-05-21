# Migration Guide: Zero-Disk DICOM Loading (Sprint 287–288)

This guide covers the two breaking pre-1.0 changes introduced in Sprint 287–288
and documents the new zero-disk DICOM loading public API.

## Breaking Changes

### 1. `DicomSliceMetadata.part10_bytes: Option<Vec<u8>>` (Sprint 287)

A new field was added to `DicomSliceMetadata` to carry in-memory Part 10 bytes
for zero-disk pixel decoding. This affects:

- **Struct literal construction**: If you construct `DicomSliceMetadata` manually,
  add `part10_bytes: None` for file-based loading, or `part10_bytes: Some(bytes)`
  for in-memory loading.

```rust
// Before (Sprint 286):
let meta = DicomSliceMetadata {
    path: PathBuf::from("/data/ct/001.dcm"),
    bits_allocated: 16,
    // ... other fields ...
    // part10_bytes did not exist
};

// After (Sprint 287+):
let meta = DicomSliceMetadata {
    path: PathBuf::from("/data/ct/001.dcm"),
    bits_allocated: 16,
    // ... other fields ...
    part10_bytes: None, // file-based: decode from path
};
```

- **`Default` impl**: Already updated. `DicomSliceMetadata::default()` sets
  `part10_bytes: None`.

### 2. `PacsConfig.auto_load_received: bool` (Sprint 287)

New field controls whether SCP-received instances are automatically loaded into
the viewer. Affects:

- **Struct literal construction**: Add `auto_load_received: true` (or your
  preferred default).

```rust
// Before (Sprint 286):
let config = PacsConfig {
    calling_ae_title: "MYAPP".to_owned(),
    called_ae_title: "PACS".to_owned(),
    host: "10.0.0.1".to_owned(),
    port: 104,
    // ... other fields ...
    // auto_load_received did not exist
};

// After (Sprint 287+):
let config = PacsConfig {
    calling_ae_title: "MYAPP".to_owned(),
    called_ae_title: "PACS".to_owned(),
    host: "10.0.0.1".to_owned(),
    port: 104,
    // ... other fields ...
    auto_load_received: true, // auto-load SCP instances on receive
};
```

### 3. `PacsConfig.auto_load_limit: u32` (Sprint 288)

New field limits the number of instances that will be auto-loaded. When pending
instances exceed this limit, auto-load is suppressed and the user must click
"Load Received". Affects:

- **Struct literal construction**: Add `auto_load_limit: 512`.

```rust
let config = PacsConfig {
    // ... other fields ...
    auto_load_received: true,
    auto_load_limit: 512, // suppress auto-load above 512 instances
};
```

## New Public API

### `scan_dicom_instances` (Sprint 287)

Scans in-memory SCP-received `StoredInstance` values, producing a
`DicomSeriesInfo` with `part10_bytes` attached to each slice.

```rust
use ritk_io::{scan_dicom_instances, StoredInstance};

let instances: Vec<StoredInstance> = /* ... from SCP ... */;
let series = scan_dicom_instances(&instances)?;
// series.metadata.slices[i].part10_bytes is Some(...)
```

### `scan_dicom_part10_bytes` (Sprint 288)

Scans in-memory DICOM Part 10 byte payloads (e.g., from drag-and-drop),
producing a `DicomSeriesInfo` with `part10_bytes` attached.

```rust
use ritk_io::scan_dicom_part10_bytes;

let files: Vec<(&str, &[u8])> = vec![
    ("slice001.dcm", &dicom_bytes_1),
    ("slice002.dcm", &dicom_bytes_2),
];
let series = scan_dicom_part10_bytes(&files)?;
```

### `load_dicom_from_series` (Sprint 287)

Loads a DICOM series from a pre-scanned `DicomSeriesInfo`. Dispatches pixel
decode via `part10_bytes`: `Some(bytes)` → in-memory decode, `None` → file I/O.

```rust
use ritk_io::load_dicom_from_series;

let (image, metadata) = load_dicom_from_series::<NdArray<f32>>(series, &device)?;
```

### `load_dicom_color_from_series` (Sprint 288)

Color counterpart of `load_dicom_from_series`. Loads an RGB DICOM series from
a pre-scanned `DicomSeriesInfo`, producing an `RgbVolume` with shape
`[depth, rows, cols, 3]`.

```rust
use ritk_io::load_dicom_color_from_series;

let (rgb_volume, metadata) = load_dicom_color_from_series::<NdArray<f32>>(series, &device)?;
```

## Zero-Disk Loading Paths

All DICOM loading paths now support in-memory bytes without writing temporary
files to disk:

| Source | Scanner | Loader | part10_bytes |
|---|---|---|---|
| Directory on disk | `scan_dicom_directory` | `load_dicom_series_with_metadata` | `None` |
| SCP-received instances | `scan_dicom_instances` | `load_dicom_from_series` | `Some(...)` |
| Dropped byte payloads | `scan_dicom_part10_bytes` | `load_dicom_from_series` | `Some(...)` |
| RGB from directory | `scan_dicom_directory` | `load_dicom_color_series` | `None` |
| RGB from SCP/dropped | `scan_dicom_instances` / `scan_dicom_part10_bytes` | `load_dicom_color_from_series` | `Some(...)` |

## Removed Functions

- `ritk-snap::dicom::loader::bytes::create_unique_temp_subdir` — no longer
  needed; all DICOM loading paths are now zero-disk.
- `ritk-snap::dicom::loader::bytes::sanitize_temp_filename` — no longer
  needed.

If you were using these internally, you will need to provide your own
implementations or switch to the zero-disk loading APIs.
