use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("D:/atlas/repos/ritk/test_data/2_skull_ct/DICOM/I103").expect("open failed");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("read failed");

    // Search for JPEG SOI
    let mut jpeg_start = None;
    for i in 0..data.len() - 1 {
        if data[i] == 0xFF && data[i+1] == 0xD8 {
            jpeg_start = Some(i);
            break;
        }
    }

    let Some(start) = jpeg_start else {
        println!("No JPEG SOI found!");
        return;
    };

    println!("JPEG SOI found at offset {}", start);
    let jpeg_data = &data[start..];

    // Decode using jpeg-decoder crate
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    match decoder.decode() {
        Ok(pixels) => {
            println!("jpeg-decoder SUCCESS! Pixel count: {}", pixels.len());
            let metadata = decoder.info().unwrap();
            println!("Info: {:?}", metadata);
            // Print first 20 pixels
            match pixels {
                jpeg_decoder::PixelFormat::L8(vec) => {
                    println!("First 20 pixels (L8): {:?}", &vec[..std::cmp::min(20, vec.len())]);
                }
                jpeg_decoder::PixelFormat::L16(vec) => {
                    println!("First 20 pixels (L16): {:?}", &vec[..std::cmp::min(20, vec.len())]);
                }
                _ => println!("Other format"),
            }
        }
        Err(e) => {
            println!("jpeg-decoder FAILED: {:?}", e);
        }
    }
}
