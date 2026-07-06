use anyhow::{Context, Result};
use ritk_image::tensor::backend::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::{Path, PathBuf};

mod color;

pub use color::{
    read_png_color_series, read_png_color_to_volume, PngColorReader, PngColorSeriesReader,
};

/// Read a single grayscale PNG into an `Image<B, 3>` with shape `[1, height, width]`.
pub fn read_png_to_image<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let (pixels, dims) = decode_png_single(path.as_ref())?;
    image_from_flat_pixels(pixels, dims[0], dims[1], dims[2], device)
}


/// Decode a single grayscale PNG into row-major `f32` pixels and `[1, h, w]` dims.
fn decode_png_single(path: &Path) -> Result<(Vec<f32>, [usize; 3])> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open PNG: {}", path.display()))?
        .to_luma8();
    let (width, height) = img.dimensions();
    let pixels: Vec<f32> = img.iter().map(|&v| v as f32).collect();
    Ok((pixels, [1, height as usize, width as usize]))
}

/// Read a directory of PNG files into a 3-D image with shape `[depth, height, width]`.
///
/// PNGs are sorted by natural filename order before stacking.
pub fn read_png_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let (pixels, dims) = decode_png_series(path.as_ref())?;
    image_from_flat_pixels(pixels, dims[0], dims[1], dims[2], device)
}


/// Decode a sorted PNG series into row-major `f32` pixels and `[depth, h, w]` dims.
fn decode_png_series(dir: &Path) -> Result<(Vec<f32>, [usize; 3])> {
    let png_files = sorted_png_files(dir)?;

    let first_img = image::open(&png_files[0])
        .with_context(|| format!("Failed to open PNG: {}", png_files[0].display()))?
        .to_luma8();
    let (width, height) = first_img.dimensions();

    // Bound the upfront reservation: the reserved length is `file_count ×
    // height × width`, all derived from untrusted inputs (directory listing +
    // the first PNG's decoded dimensions). A directory whose first image is
    // large amplifies the reservation by the file count even though later
    // mismatched files make the loop bail. `saturating_mul` avoids an
    // overflow wrapping to a tiny capacity; `bounded_capacity` caps the eager
    // reservation while the Vec still grows to its true length as pages append.
    let reserve = png_files
        .len()
        .saturating_mul(height as usize)
        .saturating_mul(width as usize);
    let mut all_pixels: Vec<f32> = Vec::with_capacity(ritk_core::io_bounds::bounded_capacity(
        reserve,
        std::mem::size_of::<f32>(),
    ));
    append_png_pixels(&mut all_pixels, &first_img);

    for file in &png_files[1..] {
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

        append_png_pixels(&mut all_pixels, &img);
    }

    Ok((
        all_pixels,
        [png_files.len(), height as usize, width as usize],
    ))
}

pub(crate) fn sorted_png_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut png_files: Vec<PathBuf> = std::fs::read_dir(dir)
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

    png_files.sort_by(|a, b| {
        let a_name = a.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let b_name = b.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        natural_cmp(a_name, b_name)
    });

    Ok(png_files)
}

fn append_png_pixels(all_pixels: &mut Vec<f32>, image: &image::GrayImage) {
    all_pixels.extend(image.iter().map(|&v| v as f32));
}

fn image_from_flat_pixels<B: Backend>(
    pixels: Vec<f32>,
    depth: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let expected = depth
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .context("PNG image shape overflow")?;
    if pixels.len() != expected {
        anyhow::bail!(
            "PNG pixel count {} does not match shape [{}, {}, {}]",
            pixels.len(),
            depth,
            height,
            width
        );
    }

    let shape = Shape::new([depth, height, width]);
    let data = TensorData::new(pixels, shape);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    Ok(Image::new(tensor, origin, spacing, direction))
}


/// DIP boundary for standard PNG single slices.
pub struct PngReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_png_to_image(path, &self.device)
    }
}

/// DIP boundary for PNG sequential volumes.
pub struct PngSeriesReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngSeriesReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_png_series(path, &self.device)
    }
}

/// Natural string comparison that handles embedded unsigned decimal runs.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();

    loop {
        match (a_chars.peek(), b_chars.peek()) {
            (Some(&ac), Some(&bc)) if ac.is_ascii_digit() && bc.is_ascii_digit() => {
                let a_num: String = a_chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let b_num: String = b_chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let a_val: u64 = a_num.parse().unwrap_or(0);
                let b_val: u64 = b_num.parse().unwrap_or(0);
                match a_val.cmp(&b_val) {
                    std::cmp::Ordering::Equal => {
                        let ord = a_num.len().cmp(&b_num.len());
                        for _ in 0..a_num.len() {
                            a_chars.next();
                        }
                        for _ in 0..b_num.len() {
                            b_chars.next();
                        }
                        if ord != std::cmp::Ordering::Equal {
                            return ord;
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

#[cfg(test)]
mod tests {
    use super::{natural_cmp, read_png_series, read_png_to_image};
    use ritk_image::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use std::cmp::Ordering;
    use std::path::Path;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        let image = image::GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match pixel count");
        image.save(path).expect("test PNG write must succeed");
    }

    fn tensor_values(image: &ritk_core::image::Image<TestBackend, 3>) -> Vec<f32> {
        image.data_slice().into_owned()
    }

    #[test]
    fn read_png_to_image_preserves_shape_values_and_default_metadata() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("slice.png");
        write_gray_png(&path, 3, 2, &[10, 20, 30, 40, 50, 60]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_png_to_image::<TestBackend, _>(&path, &device)?;

        assert_eq!(image.shape(), [1, 2, 3]);
        assert_eq!(
            tensor_values(&image),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        );
        assert_eq!(
            [image.origin()[0], image.origin()[1], image.origin()[2]],
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            [image.spacing()[0], image.spacing()[1], image.spacing()[2]],
            [1.0, 1.0, 1.0]
        );
        assert_eq!(*image.direction(), ritk_spatial::Direction::<3>::identity());

        Ok(())
    }

    #[test]
    fn read_png_series_natural_sorts_and_stacks_slices() -> anyhow::Result<()> {
        let dir = tempdir()?;
        write_gray_png(&dir.path().join("slice10.png"), 2, 1, &[10, 11]);
        write_gray_png(&dir.path().join("slice2.png"), 2, 1, &[2, 3]);
        write_gray_png(&dir.path().join("slice1.png"), 2, 1, &[1, 4]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_png_series::<TestBackend, _>(dir.path(), &device)?;

        assert_eq!(image.shape(), [3, 1, 2]);
        assert_eq!(tensor_values(&image), vec![1.0, 4.0, 2.0, 3.0, 10.0, 11.0]);

        Ok(())
    }

    #[test]
    fn read_png_series_rejects_dimension_mismatch() -> anyhow::Result<()> {
        let dir = tempdir()?;
        write_gray_png(&dir.path().join("slice1.png"), 2, 1, &[1, 2]);
        write_gray_png(&dir.path().join("slice2.png"), 1, 1, &[3]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_png_series::<TestBackend, _>(dir.path(), &device);
        let msg = match result {
            Ok(_) => panic!("mismatched PNG dimensions must fail"),
            Err(err) => format!("{err:?}"),
        };

        assert!(
            msg.contains("PNG size mismatch"),
            "error must name the dimension mismatch, got: {msg}"
        );

        Ok(())
    }

    #[test]
    fn natural_cmp_orders_embedded_numbers_by_numeric_value() {
        assert_eq!(natural_cmp("slice2", "slice10"), Ordering::Less);
        assert_eq!(natural_cmp("slice10", "slice2"), Ordering::Greater);
        assert_eq!(natural_cmp("slice2", "slice02"), Ordering::Less);
    }

    #[test]
    fn native_read_png_matches_burn_single_and_series() -> anyhow::Result<()> {
        use coeus_core::SequentialBackend;

        let dir = tempdir()?;
        let single = dir.path().join("slice.png");
        write_gray_png(&single, 3, 2, &[10, 20, 30, 40, 50, 60]);
        let device: <TestBackend as Backend>::Device = Default::default();

        let burn = read_png_to_image::<TestBackend, _>(&single, &device)?;
        let coeus = crate::native::read_png_to_image(&single, &SequentialBackend)?;
        assert_eq!(coeus.shape(), burn.shape());
        assert_eq!(coeus.data_slice()?, tensor_values(&burn).as_slice());

        let series_dir = tempdir()?;
        write_gray_png(&series_dir.path().join("s1.png"), 2, 1, &[1, 4]);
        write_gray_png(&series_dir.path().join("s2.png"), 2, 1, &[2, 3]);
        let burn_series = read_png_series::<TestBackend, _>(series_dir.path(), &device)?;
        let coeus_series = crate::native::read_png_series(series_dir.path(), &SequentialBackend)?;
        assert_eq!(coeus_series.shape(), burn_series.shape());
        assert_eq!(
            coeus_series.data_slice()?,
            tensor_values(&burn_series).as_slice()
        );
        Ok(())
    }
}

/// Atlas-native-substrate entry points (transitional module: plain
/// end-state names, disambiguated from the Burn functions by module
/// path only; folds away when the Burn path is deleted — ADR 0002 A1).
pub mod native {
    #[allow(unused_imports)]
    use super::*;

    /// Build a Coeus-backed grayscale image from flat pixels and `[depth, h, w]`
    /// dims with default spatial metadata. The Coeus constructor validates the
    /// shape product against `pixels.len()`.
    fn image_from_flat_pixels<B: coeus_core::ComputeBackend>(
        pixels: Vec<f32>,
        dims: [usize; 3],
        backend: &B,
    ) -> Result<ritk_image::native::Image<f32, B, 3>> {
        ritk_image::native::Image::from_flat_on(
            pixels,
            dims,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
            backend,
        )
    }

    /// Read a single grayscale PNG into a Coeus-backed `Image` on `backend`.
    ///
    /// The Atlas-tensor counterpart to [`read_png_to_image`], sharing `decode_png_single`.
    pub fn read_png_to_image<B, P>(
        path: P,
        backend: &B,
    ) -> Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        P: AsRef<Path>,
    {
        let (pixels, dims) = decode_png_single(path.as_ref())?;
        image_from_flat_pixels(pixels, dims, backend)
    }

    /// Read a directory of PNG files into a Coeus-backed `[depth, height, width]` image.
    ///
    /// The Atlas-tensor counterpart to [`read_png_series`], sharing `decode_png_series`.
    pub fn read_png_series<B, P>(
        path: P,
        backend: &B,
    ) -> Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        P: AsRef<Path>,
    {
        let (pixels, dims) = decode_png_series(path.as_ref())?;
        image_from_flat_pixels(pixels, dims, backend)
    }
}
