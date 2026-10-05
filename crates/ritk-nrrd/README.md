# ritk-nrrd

`ritk-nrrd` reads and writes NRRD images and complete parsed NRRD documents. The
compute-image API preserves spatial metadata and acquisition-axis ordering;
the stored-volume API retains the ten fixed-width integer and IEEE 754 sample
types plus the physical metadata represented by `ritk-image-io`. The document
API additionally retains comments, standard fields, repeated custom records,
file-axis sizes, declared sample type, byte order, and decoded samples when a
field does not fit the stored-volume model. Binary sample bits remain exact;
ASCII samples decode to their declared fixed-width representation.

The reader bounds one header to 16 MiB and 65,536 standard-field and
key/value records. Repeated custom keys count as separate records. Oversized
headers return a typed error before payload allocation.
Stored readers also accept `ImageReadBudget`, which limits encoded bytes,
decoded sample bytes, and series volume count before allocating sample data.
The default byte ceilings are 1 GiB and the default series limit is 65,536.

Readers support `raw`, `ascii` (`text`, `txt`), and `gzip` (`gz`) encodings.
ASCII tokens are whitespace-delimited and limited to 128 bytes per sample.
Detached data uses one relative file name; absolute paths and parent traversal
are rejected. `read_nrrd_stored` preserves stored samples, while `read_nrrd`
provides compute-ready `f32` images.

The native image API remains convenient for processing. Use
`read_nrrd_stored` and `read_nrrd_stored_series` when the target format can
represent the shared volume metadata. Use `read_nrrd_document` and
`write_nrrd_document` when conversion must retain NRRD fields or records that
the shared volume model cannot represent. Document output keeps semantic
fields, comments, custom records, sample values, and binary sample bits while
normalizing storage to an inline raw payload; it does not preserve the source
encoding or detached-file packaging. Both stored and document writers
preflight before replacing an existing destination.

```rust,no_run
use ritk_image_io::ImageReadBudget;
use ritk_nrrd::{read_nrrd_stored, write_nrrd_stored};

let volume = read_nrrd_stored("input.nrrd", ImageReadBudget::DEFAULT)?;
write_nrrd_stored("output.nrrd", &volume)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Preserve NRRD-only fields and repeated custom records with the document API:

```rust,no_run
use ritk_image_io::ImageReadBudget;
use ritk_nrrd::{read_nrrd_document, write_nrrd_document};

let document = read_nrrd_document("input.nrrd", ImageReadBudget::DEFAULT)?;
write_nrrd_document("output.nrrd", &document)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The [NRRD format manual](https://ryancinsight.github.io/ritk/nrrd_format.html)
documents axis ordering, spatial metadata, acquisition series, payload rules,
stored-sample calibration limits, and the document-retention API.
