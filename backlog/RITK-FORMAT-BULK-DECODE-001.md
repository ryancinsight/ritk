<a id="RITK-FORMAT-BULK-DECODE-001"></a>

## RITK-FORMAT-BULK-DECODE-001 — Decode sample buffers through one bulk path — todo
- outcome: format readers and writers convert whole sample buffers through one bulk byte-order path, choosing the byte order once per buffer, instead of per-type `chunks_exact` loops.
- acceptance: the ritk-vtk binary scalar reader, ritk-mif's float writer, the ritk-nifti and ritk-analyze voxel decoders and the JPEG 2000 QCD step sizes use one shared bulk decode and encode, with the same output bytes and values; each crate's tests pass unchanged, and an instruction-count or pinned run shows no regression on each reader's decode loop.
- scope: `crates/ritk-codecs/src/byte_decode.rs`, `crates/ritk-vtk/src/io/reader.rs`, `crates/ritk-mif/src/writer.rs`, `crates/ritk-nifti/src/header/types.rs`, `crates/ritk-analyze/src/reader.rs`, `crates/ritk-codecs/src/jpeg_2000/codestream.rs`
- next: profile these call sites and route the required bulk operations through the locked `EndianScalar` API; keep any conversion to `f32` explicit at the image boundary.
- basis: fb1ff642dcc2ce722780dc9b051cb0796897ca25
- status: todo
- needs: none
- priority: tightening
