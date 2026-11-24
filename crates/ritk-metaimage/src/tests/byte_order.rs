//! MetaIO accepts both byte-order field names. Its BinaryDataByteOrderMSB
//! field takes precedence when both are present, independently of line order.
use std::io::Write;

use anyhow::Result;
use coeus_core::SequentialBackend;
use flate2::{write::ZlibEncoder, Compression};

#[test]
fn aliases_decode_identical_signed_samples_in_all_packagings() -> Result<()> {
    let root = tempfile::tempdir()?;
    let expected = [-513_i16, 258, i16::MIN, i16::MAX];
    for (fields, big) in [
        ("ElementByteOrderMSB = True", true),
        ("ElementByteOrderMSB = False", false),
        (
            "BinaryDataByteOrderMSB = True\nElementByteOrderMSB = False",
            true,
        ),
        (
            "ElementByteOrderMSB = True\nBinaryDataByteOrderMSB = False",
            false,
        ),
    ] {
        let raw: Vec<_> = expected
            .iter()
            .flat_map(|s| {
                if big {
                    s.to_be_bytes()
                } else {
                    s.to_le_bytes()
                }
            })
            .collect();
        for compressed in [false, true] {
            let payload = if compressed {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(&raw)?;
                encoder.finish()?
            } else {
                raw.clone()
            };
            for detached in [false, true] {
                let path = root
                    .path()
                    .join(if detached { "scan.mhd" } else { "scan.mha" });
                let data_file = if detached { "scan.raw" } else { "LOCAL" };
                let mut file = std::fs::File::create(&path)?;
                writeln!(file, "ObjectType = Image\nNDims = 3\nDimSize = 2 2 1\nOffset = 2 3 5\nElementSpacing = 1 2 4\nElementType = MET_SHORT\n{fields}\nCompressedData = {compressed}\nElementDataFile = {data_file}")?;
                if detached {
                    std::fs::write(root.path().join(data_file), &payload)?;
                } else {
                    file.write_all(&payload)?;
                }
                drop(file);
                let image = crate::read_metaimage(&path, &SequentialBackend)?;
                assert_eq!(image.shape(), [1, 2, 2]);
                assert_eq!(
                    image.data_slice().expect("host pixels"),
                    expected.map(f32::from)
                );
            }
        }
    }
    Ok(())
}
