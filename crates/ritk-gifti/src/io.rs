//! Reading and writing whole documents.

use std::io::{Read, Write};

use quick_xml::escape::escape;

use crate::decode::encode;
use crate::{DataArray, DataEncoding, GiftiError, GiftiImage, MetaData};

impl GiftiImage {
    /// Read a GIFTI document.
    ///
    /// # Errors
    ///
    /// [`GiftiError::Io`] for a read failure or text that is not UTF-8;
    /// [`GiftiError::Xml`] for malformed XML; [`GiftiError::Structure`] for a
    /// document that breaks the DTD, declares an unreasonable shape, or whose
    /// `NumberOfDataArrays` disagrees with its content; [`GiftiError::Data`]
    /// for a payload that does not decode to its declared shape;
    /// [`GiftiError::Unsupported`] for a version other than 1.0 or data held
    /// in an external file.
    pub fn read(reader: impl Read) -> Result<Self, GiftiError> {
        crate::read::read(reader)
    }

    /// Write the document with every data array in `encoding`.
    ///
    /// Arrays are written in row-major order with little-endian binary, the
    /// forms every GIFTI reader accepts; a column-major array is transposed
    /// on the way out, so reading the output back gives equal values in
    /// row-major order.
    ///
    /// # Errors
    ///
    /// [`GiftiError::Io`] on write failure.
    pub fn write(&self, mut writer: impl Write, encoding: DataEncoding) -> Result<(), GiftiError> {
        let writer = &mut writer;
        writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
        writeln!(
            writer,
            r#"<GIFTI Version="1.0" NumberOfDataArrays="{}">"#,
            self.arrays.len()
        )?;
        write_metadata(writer, &self.metadata)?;
        if !self.labels.is_empty() {
            writeln!(writer, "<LabelTable>")?;
            for label in &self.labels {
                write!(writer, r#"<Label Key="{}""#, label.key())?;
                if let Some([red, green, blue, alpha]) = label.rgba() {
                    write!(
                        writer,
                        r#" Red="{red}" Green="{green}" Blue="{blue}" Alpha="{alpha}""#
                    )?;
                }
                writeln!(writer, ">{}</Label>", escape(label.name()))?;
            }
            writeln!(writer, "</LabelTable>")?;
        }
        for array in &self.arrays {
            write_array(writer, array, encoding)?;
        }
        writeln!(writer, "</GIFTI>")?;
        Ok(())
    }
}

fn write_metadata(writer: &mut impl Write, metadata: &MetaData) -> Result<(), GiftiError> {
    if metadata.entries().is_empty() {
        return Ok(());
    }
    writeln!(writer, "<MetaData>")?;
    for (name, value) in metadata.entries() {
        writeln!(
            writer,
            "<MD><Name>{}</Name><Value>{}</Value></MD>",
            escape(name.as_str()),
            escape(value.as_str())
        )?;
    }
    writeln!(writer, "</MetaData>")?;
    Ok(())
}

fn write_array(
    writer: &mut impl Write,
    array: &DataArray,
    encoding: DataEncoding,
) -> Result<(), GiftiError> {
    let data = array.row_major();
    write!(
        writer,
        r#"<DataArray Intent="{}" DataType="{}" ArrayIndexingOrder="RowMajorOrder" Dimensionality="{}""#,
        escape(array.intent().name()),
        data.type_name(),
        array.dims().len()
    )?;
    for (axis, extent) in array.dims().iter().enumerate() {
        write!(writer, r#" Dim{axis}="{extent}""#)?;
    }
    let encoding_name = match encoding {
        DataEncoding::Ascii => "ASCII",
        DataEncoding::Base64Binary => "Base64Binary",
        DataEncoding::GZipBase64Binary => "GZipBase64Binary",
    };
    writeln!(
        writer,
        r#" Encoding="{encoding_name}" Endian="LittleEndian">"#
    )?;
    write_metadata(writer, array.metadata())?;
    for transform in array.transforms() {
        writeln!(writer, "<CoordinateSystemTransformMatrix>")?;
        writeln!(
            writer,
            "<DataSpace>{}</DataSpace>",
            escape(transform.data_space.as_str())
        )?;
        writeln!(
            writer,
            "<TransformedSpace>{}</TransformedSpace>",
            escape(transform.transformed_space.as_str())
        )?;
        let values: Vec<String> = transform
            .matrix
            .iter()
            .flatten()
            .map(ToString::to_string)
            .collect();
        writeln!(writer, "<MatrixData>{}</MatrixData>", values.join(" "))?;
        writeln!(writer, "</CoordinateSystemTransformMatrix>")?;
    }
    writeln!(writer, "<Data>{}</Data>", encode(&data, encoding)?)?;
    writeln!(writer, "</DataArray>")?;
    Ok(())
}
