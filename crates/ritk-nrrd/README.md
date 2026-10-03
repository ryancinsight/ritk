# ritk-nrrd

`ritk-nrrd` reads and writes NRRD images. It preserves spatial metadata and
acquisition-axis ordering for the existing compute-image API, and its stored
sample reader preserves all ten fixed-width integer and IEEE 754 sample types,
accepts either binary byte order, and retains physical metadata and coordinate
maps through `ritk-image-io`. It writes raw payloads in little-endian order.
Binary reads and writes preserve floating-point payload bits; ASCII values are
parsed from text into their declared sample type.

The reader bounds one header to 16 MiB and 65,536 distinct fields and
key/value pairs. Oversized headers return a typed error before payload
allocation.
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

```rust,no_run
use ritk_image_io::ImageReadBudget;
use ritk_nrrd::read_nrrd_stored;

let volume = read_nrrd_stored("input.nrrd", ImageReadBudget::DEFAULT)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The [NRRD format manual](https://ryancinsight.github.io/ritk/nrrd_format.html)
documents axis ordering, spatial metadata, acquisition series, payload rules,
and the stored-sample writer's calibration limits.
