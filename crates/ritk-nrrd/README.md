# ritk-nrrd

`ritk-nrrd` reads and writes NRRD images. It preserves spatial metadata and
acquisition-axis ordering for the existing compute-image API, and its stored
sample reader preserves all ten fixed-width integer and IEEE 754 sample types,
accepts either binary byte order, and retains physical metadata and coordinate
maps through `ritk-image-io`. It writes raw payloads in little-endian order.
Binary reads and writes preserve floating-point payload bits; ASCII values are
parsed from text into their declared sample type.

The reader bounds one header to 16 MiB and 65,536 retained entries across
standard fields, comments, and custom key/value records. Repeated custom keys
count as separate records. Oversized headers return a typed error before
payload allocation.
Stored readers also accept `ImageReadBudget`, which limits encoded bytes,
decoded sample bytes, and series volume count before allocating sample data.
The default byte ceilings are 1 GiB and the default series limit is 65,536.

Readers support `raw`, `ascii` (`text`, `txt`), and `gzip` (`gz`) encodings.
ASCII tokens are whitespace-delimited and limited to 128 bytes per sample.
Detached data uses one relative file name; absolute paths and parent traversal
are rejected. `read_nrrd_stored` preserves stored samples, while `read_nrrd`
provides compute-ready `f32` images.

The native image API remains convenient for processing. Use
`read_nrrd_stored` and `read_nrrd_stored_series` when a format conversion must
retain the original sample representation and exact payload values.
The stored writer streams samples through a buffered file writer without
creating another volume-sized encoded payload.

`read_nrrd_header` reads metadata without decoding samples. Its `NrrdHeader`
exposes canonical standard fields, the format version, comment lines, the
effective custom key/value map, and every decoded custom record in source
order. Repeated keys remain inspectable even though the effective map follows
NRRD's last-value rule.

```rust,no_run
use ritk_image_io::ImageReadBudget;
use ritk_nrrd::{read_nrrd_document, write_nrrd_document};

let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
write_nrrd_document("output.nrrd", &document)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Documents with retained metadata

`NrrdDocument::new(series, comments, records)` constructs a document directly
from a `ritk_image_io::StoredSeries`, comment strings, and ordered `(String,
String)` custom key/value pairs. `series()`, `comments()`, and `records()`
provide borrowed access. Construction checks the same metadata, header limits,
series consistency, physical geometry, and identity calibration as writing.

```rust
use ritk_image_io::StoredSeries;
use ritk_nrrd::{NrrdDocument, NrrdDocumentError};

fn document_with_provenance(series: StoredSeries) -> Result<NrrdDocument, NrrdDocumentError> {
    NrrdDocument::new(
        series,
        vec!["# imported scan".to_owned()],
        vec![
            ("source".to_owned(), "scanner".to_owned()),
            ("source".to_owned(), "reviewed".to_owned()),
        ],
    )
}
```

Retained comments keep their leading `#` and their order. Custom records keep
all repeated keys and their order; an effective key/value map uses the last
value. The writer emits generated fields, then retained comments, then custom
records, so source interleaving and header bytes are not preserved. It
regenerates the sample type, shape, spatial fields, acquisition fields, and
coordinate-map and diffusion records from the typed series, with an attached
raw little-endian payload. The document reader removes the writer's generated
banner so successive writes do not accumulate it.

`read_nrrd_document` and the constructor return
`NrrdDocumentError::UnsupportedField` for metadata the document cannot retain,
including standard `content` and `labels` fields. Caller-supplied structural
or reserved generated records are rejected. Use `read_nrrd_header` to inspect
unsupported header fields; document conversion does not silently discard them.
`write_nrrd_document` completes validation before creating or truncating the
destination. Validation errors leave an existing destination unchanged; an I/O
failure after opening the file can leave partial output.

Both construction and the write guard run the shared stored-series preflight.
`NrrdStoredSeriesTarget` implements `ritk_image_io::ConversionAdapter`, and
`NrrdDocument::from_stored_series` runs it before a document exists. A source
whose sample type, geometry, calibration, axis, or coordinate map NRRD cannot
represent is reported as a typed `NrrdStoredSeriesRejection` scoped to the
series or to the exact volume, or as a `ConversionCapabilityReport` when the
gap is a declared feature category rather than an input-dependent value.

The [NRRD format manual](https://ryancinsight.github.io/ritk/nrrd_format.html)
documents axis ordering, spatial metadata, acquisition series, payload rules,
and the stored-sample writer's calibration limits.
