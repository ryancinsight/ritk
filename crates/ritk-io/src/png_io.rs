use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Read a single grayscale PNG into an Image<B, 3> with shape [1, height, width].
pub fn read_png_to_image<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let img = image::open(path)
        .with_context(|| format!("Failed to open PNG: {}", path.display()))?
        .to_luma8();

    let (width, height) = img.dimensions();
    let pixels: Vec<f32> = img.iter().map(|&v| v as f32).collect();

    let shape = Shape::new([1, height as usize, width as usize]);
    let data = TensorData::new(pixels, shape);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Read a series of PNG files from a directory into a 3D Image [depth, height, width].
///
/// PNGs are sorted by filename (natural sort for numbered files).
pub fn read_png_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let dir = path.as_ref();

    // Collect all PNG files
    let mut png_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();

    if png_files.is_empty() {
        anyhow::bail!("No PNG files found in {}", dir.display());
    }

    // Natural sort by filename numbers
    png_files.sort_by(|a, b| {
        let a_name = a.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let b_name = b.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        natural_cmp(a_name, b_name)
    });

    println!(
        "Loading {} PNG files from {}",
        png_files.len(),
        dir.display()
    );

    // Load first image to get dimensions
    let first_img = image::open(&png_files[0])
        .with_context(|| format!("Failed to open PNG: {}", png_files[0].display()))?
        .to_luma8();
    let (width, height) = first_img.dimensions();

    // Load all slices
    let mut all_pixels: Vec<f32> =
        Vec::with_capacity(png_files.len() * height as usize * width as usize);

    for file in &png_files {
        let img = image::open(file)
            .with_context(|| format!("Failed to open PNG: {}", file.display()))?
            .to_luma8();

        let (w, h) = img.dimensions();
        if w != width || h != height {
            anyhow::bail!(
                "PNG size mismatch: {} is {}x{} but expected {}x{}",
                file.display(),
                w,
                h,
                width,
                height
            );
        }

        all_pixels.extend(img.iter().map(|&v| v as f32));
    }

    let depth = png_files.len();
    let shape = Shape::new([depth, height as usize, width as usize]);
    let data = TensorData::new(all_pixels, shape);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    println!(
        "Loaded {} slices of {}x{} (total {} voxels)",
        depth,
        width,
        height,
        depth * height as usize * width as usize,
    );

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Natural string comparison that handles embedded numbers.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();

    loop {
        match (a_chars.peek(), b_chars.peek()) {
            (Some(&ac), Some(&bc)) if ac.is_ascii_digit() && bc.is_ascii_digit() => {
                // Parse full number from both
                let a_num: String = a_chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let b_num: String = b_chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let a_val: u64 = a_num.parse().unwrap_or(0);
                let b_val: u64 = b_num.parse().unwrap_or(0);
                match a_val.cmp(&b_val) {
                    std::cmp::Ordering::Equal => {
                        // Advance both by the same amount
                        let len = a_num.len().min(b_num.len());
                        for _ in 0..len {
                            a_chars.next();
                            b_chars.next();
                        }
                    }
                    ord => return ord,
                }
            }
            (Some(&ac), Some(&bc)) => match ac.cmp(&bc) {
                std::cmp::Ordering::Equal => {
                    a_chars.next();
                    b_chars.next();
                }
                ord => return ord,
            },
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (None, None) => return std::cmp::Ordering::Equal,
        }
    }
}
