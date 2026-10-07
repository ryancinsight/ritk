# ritk-image-io

ritk-image-io defines the typed, geometry-aware volume exchanged by RITK
format adapters. It keeps fixed-width stored samples separate from Coeus
compute scalars and carries calibration as metadata without silently applying
it. Spatial metadata uses patient LPS coordinates and millimeters; each format
adapter converts its source basis and units at the file boundary.
Stored-volume construction preserves every finite positive spacing and rejects
direction-times-spacing components that overflow or underflow to zero, keeping
format writers from emitting unrepresentable physical axes.
`IntensityUnit` carries the source label for calibrated values as uninterpreted
text. Conversion targets declare whether they can retain this label; a target
without that representation receives a typed loss before its writer runs.

`ImageReadBudget` provides encoded-byte, decoded-byte, and series-volume
ceilings to format readers. Its default is 1 GiB for encoded and decoded
payloads and 65,536 volumes; applications may construct smaller or larger
limits for their data and memory policy.

~~~rust
use ritk_codecs::SampleBuffer;
use ritk_image::ImageMetadata;
use ritk_image_io::{IntensityCalibration, StoredVolume};
use ritk_spatial::CoordinateMap;

let volume = StoredVolume::new(
    [1, 1, 2],
    SampleBuffer::from_samples(vec![16_777_u16, 16_778]),
    ImageMetadata::default_for_shape([1, 1, 2]),
    CoordinateMap::Cartesian,
    IntensityCalibration::Identity,
)?;
assert_eq!(volume.shape(), [1, 1, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())
~~~

Format adapters own validation. Capability reports list feature categories and
scoped metadata losses. `prepare_conversion` rejects reported losses, then asks
the selected target adapter to validate cross-volume values and format limits
and produce a target-owned plan. Its `PreparedConversion` keeps the plan tied to
the exact immutable series checked. Source readers supply losses for metadata
that the shared model cannot retain. Preparation takes no destination path, so
a writer receives a witness only after the full target contract passes.
