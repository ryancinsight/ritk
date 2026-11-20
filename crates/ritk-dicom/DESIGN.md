# Native DICOM dataset cutover

The dataset is a sorted map of unique tags. Nested items own datasets. Each
binary primitive and each fragment can share the captured input allocation.
Numeric values use host representations and have an explicit byte order at
the wire boundary. The writer emits the standard padding byte for the VR
and rejects an unrepresentable length. Part 10 metadata uses Explicit VR
Little Endian, regardless of the dataset transfer syntax.

One bounded traversal validates headers and container lengths, visits raw
spans for catalog selection, and builds the mutable dataset. Implicit VR
uses the PS3.6 tag and VR registry. The qualified representations are
resolved with their dataset context. The writer preserves defined and undefined item framing and recalculates
defined byte lengths. DICOMDIR record offsets are absolute addresses used
by the reader. The generic writer does not relocate directory records
when an edit moves them.

Transfer syntax owns the stream encoding. For compressed Pixel Data the
offset table points relative to the first fragment item tag. Frame
selection uses those offsets and lends a contiguous fragment when possible.
For big-endian output, word swaps use one 8 KiB buffer per primitive.
Codec adapters consume encoded frame bytes and apply the modality transform
once. Validation covers ordered tag values, nested spans, endian words,
code-set overrides, declared sample layout, and pixel frame offsets.

A fork of the external object model would keep two independently evolving
dataset implementations. A reader-only native object would leave the
writer and C-STORE boundary dependent on that model. This cutover gives
RITK one dataset and file representation across ingest, repair, export,
and upload. Independent external decoders remain useful as test oracles.
