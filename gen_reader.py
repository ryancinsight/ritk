import re
src = open("crates/ritk-io/src/format/dicom/reader.rs").read()
DQ = chr(34)
# Replace read_slice_pixels with dicom-aware version
old = src[src.index("fn read_slice_pixels"):src.index("fn is_likely_dicom_file")]
new_fn = ("""fn read_slice_pixels(slice: &DicomSliceMetadata) -> Result<Vec<f32>> {""" + "\n" +
    """    use dicom::core::Tag;""" + "\n" +
    """    use dicom::object::open_file;""" + "\n" +
    """    if let Ok(obj) = open_file(&slice.path) {""" + "\n" +
    """        if let Ok(pixel_elem) = obj.element(Tag(0x7FE0, 0x0010)) {""" + "\n" +
    """            let pv = pixel_elem.value();""" + "\n" +
    """            if let Ok(bytes) = pv.to_bytes() {""" + "\n" +
    """                if bytes.len() >= 2 {""" + "\n" +
    """                    let data: Vec<f32> = bytes.chunks_exact(2)""" + "\n" +
    """                        .map(|c| {""" + "\n" +
    """                            let raw = u16::from_le_bytes([c[0], c[1]]);""" + "\n" +
    """                            raw as f32 * slice.rescale_slope + slice.rescale_intercept""" + "\n" +
    """                        })""" + "\n" +
    """                        .collect();""" + "\n" +
    """                    if !data.is_empty() { return Ok(data); }""" + "\n" +
    """                }""" + "\n" +
    """            }""" + "\n" +
    """        }""" + "\n" +
    """    }""" + "\n" +
    """    let bytes = std::fs::read(&slice.path)""" + "\n" +
    """        .with_context(|| format!(""" + DQ + """failed to read DICOM slice {:?}""" + DQ + """, slice.path))?;""" + "\n" +
    """    if bytes.is_empty() { bail!(""" + DQ + """DICOM slice file is empty""" + DQ + """); }""" + "\n" +
    """    let data: Vec<f32> = bytes.chunks_exact(2)""" + "\n" +
    """        .map(|c| { let raw = u16::from_le_bytes([c[0], c[1]]); raw as f32 * slice.rescale_slope + slice.rescale_intercept })""" + "\n" +
    """        .collect();""" + "\n" +
    """    if data.is_empty() { bail!(""" + DQ + """DICOM slice contained no decodable pixel data""" + DQ + """); }""" + "\n" +
    """    Ok(data)""" + "\n" +
    """}""" + "\n\n")
src = src.replace(old, new_fn)
open("crates/ritk-io/src/format/dicom/reader.rs", "w").write(src)
print("reader.rs updated")
