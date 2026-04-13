//! `ritk segment` — image segmentation command.
//!
//! Applies one of the following segmentation algorithms to a 3-D medical image:
//!
//! | Method               | Algorithm                                      |
//! |----------------------|------------------------------------------------|
//! | `otsu`               | Single-threshold Otsu (maximises between-class variance) |
//! | `multi-otsu`         | Multi-class Otsu (K−1 thresholds, K classes)   |
//! | `connected-threshold`| BFS flood-fill region growing from a seed voxel|
//!
//! # Output
//! - `otsu`: binary mask (0.0 / 1.0) + printed threshold value.
//! - `multi-otsu`: label image (0.0, 1.0, …, K−1.0) + printed threshold list.
//! - `connected-threshold`: binary mask (0.0 / 1.0) + foreground voxel count.

use anyhow::{anyhow, Context, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use ritk_core::segmentation::{
    connected_threshold, multi_otsu_threshold, otsu_threshold, MultiOtsuThreshold, OtsuThreshold,
};

use super::{read_image, write_image_inferred, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `segment` subcommand.
#[derive(Args, Debug)]
pub struct SegmentArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output mask / label image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Segmentation method.
    ///
    /// Accepted values: `otsu`, `multi-otsu`, `connected-threshold`.
    #[arg(long, value_name = "METHOD")]
    pub method: String,

    // ── Multi-Otsu ────────────────────────────────────────────────────────
    /// Number of intensity classes for `multi-otsu`.
    ///
    /// Must be ≥ 2.  Produces `classes − 1` threshold values.
    #[arg(long, default_value = "3", value_name = "INT")]
    pub classes: usize,

    // ── Connected-threshold ───────────────────────────────────────────────
    /// Inclusive lower intensity bound for `connected-threshold`.
    #[arg(long, value_name = "FLOAT")]
    pub lower: Option<f32>,

    /// Inclusive upper intensity bound for `connected-threshold`.
    #[arg(long, value_name = "FLOAT")]
    pub upper: Option<f32>,

    /// Seed voxel for `connected-threshold` in `Z,Y,X` index order.
    ///
    /// Example: `--seed 4,5,6` sets z=4, y=5, x=6.
    #[arg(long, value_name = "Z,Y,X")]
    pub seed: Option<String>,
}

// ── Seed parsing ──────────────────────────────────────────────────────────────

/// Parse a `"Z,Y,X"` string into a `[usize; 3]` seed voxel index.
///
/// # Errors
/// Returns an error when the string does not contain exactly three
/// comma-separated non-negative integer tokens.
fn parse_seed(s: &str) -> Result<[usize; 3]> {
    let parts: Vec<&str> = s.splitn(4, ',').collect();
    if parts.len() != 3 {
        return Err(anyhow!(
            "Seed must be provided as Z,Y,X (three comma-separated integers), got: '{s}'"
        ));
    }
    let z = parts[0]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid Z component '{}' in seed '{s}'", parts[0]))?;
    let y = parts[1]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid Y component '{}' in seed '{s}'", parts[1]))?;
    let x = parts[2]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid X component '{}' in seed '{s}'", parts[2]))?;
    Ok([z, y, x])
}

// ── Foreground count helper ───────────────────────────────────────────────────

/// Count the number of voxels with value > 0.5 in `image`.
///
/// Suitable for binary (0.0 / 1.0) masks produced by Otsu and
/// connected-threshold segmentation.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
fn count_foreground(image: &ritk_core::image::Image<Backend, 3>) -> usize {
    let td = image.data().clone().into_data();
    let slice = td
        .as_slice::<f32>()
        .expect("segmentation output must contain f32 data");
    slice.iter().filter(|&&v| v > 0.5).count()
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `segment` subcommand.
///
/// Dispatches to the appropriate segmentation algorithm based on `args.method`,
/// writes the output mask / label image, and prints a one-line summary.
///
/// # Errors
/// Returns an error when:
/// - The input image cannot be read.
/// - A required argument for the chosen method is missing or malformed.
/// - The output image cannot be written.
/// - An unknown method name is supplied.
pub fn run(args: SegmentArgs) -> Result<()> {
    info!(
        input  = %args.input.display(),
        output = %args.output.display(),
        method = %args.method,
        "segment: starting"
    );

    match args.method.as_str() {
        "otsu" => run_otsu(&args),
        "multi-otsu" => run_multi_otsu(&args),
        "connected-threshold" => run_connected_threshold(&args),
        other => Err(anyhow!(
            "Unknown segmentation method '{other}'. \
             Supported methods: otsu, multi-otsu, connected-threshold."
        )),
    }
}

// ── Otsu thresholding ─────────────────────────────────────────────────────────

/// Apply single-threshold Otsu segmentation.
///
/// Computes the optimal threshold t* that maximises between-class variance,
/// then maps voxels ≥ t* to 1.0 (foreground) and voxels < t* to 0.0
/// (background).
fn run_otsu(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    // Compute threshold first (for reporting) then produce the binary mask.
    let threshold = otsu_threshold::<Backend, 3>(&image);
    let mask = OtsuThreshold::new().apply(&image);

    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        input      = %args.input.display(),
        threshold  = threshold,
        foreground = n_foreground,
        "segment: otsu complete"
    );

    Ok(())
}

// ── Multi-Otsu thresholding ───────────────────────────────────────────────────

/// Apply multi-class Otsu segmentation with `args.classes` intensity classes.
///
/// Computes K−1 optimal thresholds and maps each voxel to the class label
/// (0.0, 1.0, …, K−1.0) whose intensity interval it falls into.
fn run_multi_otsu(args: &SegmentArgs) -> Result<()> {
    if args.classes < 2 {
        return Err(anyhow!(
            "--classes must be ≥ 2 for multi-otsu, got {}",
            args.classes
        ));
    }

    let image = read_image(&args.input)?;

    // Compute thresholds for reporting.
    let thresholds = multi_otsu_threshold::<Backend, 3>(&image, args.classes);
    let labeled = MultiOtsuThreshold::new(args.classes).apply(&image);

    // Count non-background (class > 0) voxels for the summary line.
    let n_labeled = count_foreground(&labeled);

    write_image_inferred(&args.output, &labeled)?;

    // Format threshold list as "[T1, T2, …]".
    let thresh_str: Vec<String> = thresholds.iter().map(|t| format!("{t:.4}")).collect();
    println!(
        "Segmented {}: found {} labeled voxels / thresholds=[{}]",
        args.input.display(),
        n_labeled,
        thresh_str.join(", "),
    );

    info!(
        input      = %args.input.display(),
        classes    = args.classes,
        thresholds = ?thresholds,
        labeled    = n_labeled,
        "segment: multi-otsu complete"
    );

    Ok(())
}

// ── Connected-threshold region growing ───────────────────────────────────────

/// Apply connected-threshold BFS region growing from a user-specified seed.
///
/// Voxels reachable from `seed` whose intensities lie in `[lower, upper]`
/// are set to 1.0 (foreground); all others are set to 0.0 (background).
///
/// # Argument validation
/// `--lower`, `--upper`, and `--seed` are all required for this method.
fn run_connected_threshold(args: &SegmentArgs) -> Result<()> {
    let lower = args
        .lower
        .ok_or_else(|| anyhow!("--lower is required for the connected-threshold method"))?;
    let upper = args
        .upper
        .ok_or_else(|| anyhow!("--upper is required for the connected-threshold method"))?;
    let seed_str = args.seed.as_deref().ok_or_else(|| {
        anyhow!("--seed is required for the connected-threshold method (format: Z,Y,X)")
    })?;

    if lower > upper {
        return Err(anyhow!("--lower ({lower}) must be ≤ --upper ({upper})"));
    }

    let seed = parse_seed(seed_str).with_context(|| {
        format!("Failed to parse --seed '{seed_str}' (expected Z,Y,X integer format)")
    })?;

    let image = read_image(&args.input)?;

    // Validate seed bounds against image shape before calling the kernel so
    // that any out-of-bounds error is surfaced as a user-friendly message
    // rather than a panic from the core implementation.
    let shape = image.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(anyhow!(
            "Seed [{},{},{}] is out of bounds for image shape [{}×{}×{}]",
            seed[0],
            seed[1],
            seed[2],
            shape[0],
            shape[1],
            shape[2],
        ));
    }

    let mask = connected_threshold::<Backend>(&image, seed, lower, upper);
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}])",
        args.input.display(),
        n_foreground,
        seed[0],
        seed[1],
        seed[2],
        lower,
        upper,
    );

    info!(
        input      = %args.input.display(),
        seed       = ?seed,
        lower      = lower,
        upper      = upper,
        foreground = n_foreground,
        "segment: connected-threshold complete"
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    // ── Test image factories ──────────────────────────────────────────────────

    /// Build a 4×4×4 bimodal image.
    ///
    /// The first half of voxels (flat indices 0–31) have intensity 20.0;
    /// the second half (32–63) have intensity 200.0.
    /// The analytically correct Otsu threshold lies between 20.0 and 200.0.
    fn make_bimodal_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64)
            .map(|i| if i < 32 { 20.0_f32 } else { 200.0_f32 })
            .collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 6×6×6 trimodal image for multi-Otsu tests.
    ///
    /// Voxels are split into three equal groups with intensities 30, 130, 230.
    fn make_trimodal_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let n = 6 * 6 * 6; // 216
        let values: Vec<f32> = (0..n)
            .map(|i| {
                if i < n / 3 {
                    30.0_f32
                } else if i < 2 * n / 3 {
                    130.0_f32
                } else {
                    230.0_f32
                }
            })
            .collect();
        let td = TensorData::new(values, Shape::new([6, 6, 6]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 5×5×5 image with a high-intensity sphere at the centre.
    ///
    /// Centre voxel (2,2,2) and its 6 face-adjacent neighbours have intensity
    /// 200.0; all other voxels have intensity 10.0.
    fn make_sphere_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let (nz, ny, nx) = (5usize, 5usize, 5usize);
        let mut values = vec![10.0_f32; nz * ny * nx];
        let high_indices: &[(usize, usize, usize)] = &[
            (2, 2, 2), // centre
            (1, 2, 2), // −Z
            (3, 2, 2), // +Z
            (2, 1, 2), // −Y
            (2, 3, 2), // +Y
            (2, 2, 1), // −X
            (2, 2, 3), // +X
        ];
        for &(z, y, x) in high_indices {
            values[z * ny * nx + y * nx + x] = 200.0;
        }
        let td = TensorData::new(values, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // ── Positive: Otsu creates binary output file ─────────────────────────────

    /// Otsu segmentation must produce a file with the correct shape.
    #[test]
    fn test_segment_otsu_creates_output_file_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "otsu".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
        })
        .unwrap();

        assert!(output.exists(), "output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [4, 4, 4], "output shape must match input");
    }

    // ── Positive: Otsu output is strictly binary ──────────────────────────────

    /// Every voxel in the Otsu output mask must be exactly 0.0 or 1.0.
    #[test]
    fn test_segment_otsu_output_is_strictly_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("mask.mha");

        ritk_io::write_metaimage(&input, &make_bimodal_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "otsu".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
        })
        .unwrap();

        let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Otsu output must be strictly binary (0.0 or 1.0), got {v}"
            );
        }
    }

    // ── Positive: Otsu threshold is between the two modes ─────────────────────

    /// For a bimodal image with modes at 20 and 200, the Otsu threshold must
    /// lie strictly between 20 and 200.
    #[test]
    fn test_segment_otsu_threshold_between_modes() {
        let image = make_bimodal_image();
        let threshold = otsu_threshold::<Backend, 3>(&image);
        assert!(
            threshold > 20.0 && threshold < 200.0,
            "Otsu threshold {threshold} must lie between the two modes (20, 200)"
        );
    }

    // ── Positive: Otsu foreground count matches high-intensity voxels ─────────

    /// In the bimodal image the high-intensity half (32 voxels at 200.0)
    /// must become the foreground class.
    #[test]
    fn test_segment_otsu_foreground_count_equals_high_mode_voxels() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "otsu".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
        })
        .unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        // The bimodal image has exactly 32 voxels at 200.0.
        assert_eq!(
            foreground, 32,
            "Otsu must label exactly 32 high-intensity voxels as foreground"
        );
    }

    // ── Positive: Multi-Otsu creates labeled output ────────────────────────────

    /// Multi-Otsu with 3 classes must create an output file with the correct shape.
    #[test]
    fn test_segment_multi_otsu_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("labels.nii");

        ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "multi-otsu".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
        })
        .unwrap();

        assert!(output.exists(), "output label image must be created");
        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(labels.shape(), [6, 6, 6], "label shape must match input");
    }

    // ── Positive: Multi-Otsu labels are in valid set ───────────────────────────

    /// For K=3 classes, every voxel label must be in {0.0, 1.0, 2.0}.
    #[test]
    fn test_segment_multi_otsu_labels_in_valid_set() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("labels.mha");

        ritk_io::write_metaimage(&input, &make_trimodal_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "multi-otsu".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
        })
        .unwrap();

        let labels = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let td = labels.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        let valid = [0.0_f32, 1.0_f32, 2.0_f32];
        for &v in values {
            assert!(
                valid.iter().any(|&vv| (v - vv).abs() < 1e-4),
                "label value {v} is not in the valid set {{0, 1, 2}} for K=3"
            );
        }
    }

    // ── Positive: Multi-Otsu returns K-1 thresholds ───────────────────────────

    /// For K=3 classes, `multi_otsu_threshold` must return exactly 2 thresholds,
    /// both lying within the image's intensity range.
    #[test]
    fn test_segment_multi_otsu_returns_k_minus_1_thresholds() {
        let image = make_trimodal_image();
        let thresholds = multi_otsu_threshold::<Backend, 3>(&image, 3);
        assert_eq!(
            thresholds.len(),
            2,
            "K=3 must produce exactly 2 thresholds, got {:?}",
            thresholds
        );
        for &t in &thresholds {
            assert!(
                t >= 30.0 && t <= 230.0,
                "threshold {t} must lie within the image intensity range [30, 230]"
            );
        }
        // Thresholds must be strictly increasing.
        assert!(
            thresholds[0] < thresholds[1],
            "thresholds must be strictly increasing: {:?}",
            thresholds
        );
    }

    // ── Positive: Connected-threshold grows sphere region ─────────────────────

    /// Seeding at the centre of the sphere must grow exactly the 7 high-intensity
    /// voxels (centre + 6 face-adjacent neighbours).
    #[test]
    fn test_segment_connected_threshold_grows_sphere_from_centre_seed() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("2,2,2".to_string()),
        })
        .unwrap();

        assert!(output.exists(), "output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        assert_eq!(
            foreground, 7,
            "connected-threshold from centre seed must grow exactly 7 sphere voxels"
        );
    }

    // ── Positive: Connected-threshold output is strictly binary ───────────────

    #[test]
    fn test_segment_connected_threshold_output_is_strictly_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("mask.mha");

        ritk_io::write_metaimage(&input, &make_sphere_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("2,2,2".to_string()),
        })
        .unwrap();

        let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "connected-threshold output must be strictly binary, got {v}"
            );
        }
    }

    // ── Negative: unknown method returns descriptive error ────────────────────

    #[test]
    fn test_segment_unknown_method_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "watershed".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
        });

        assert!(result.is_err(), "unknown method must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown segmentation method 'watershed'"),
            "error must name the unsupported method, got: {msg}"
        );
    }

    // ── Negative: connected-threshold missing --lower ─────────────────────────

    #[test]
    fn test_segment_connected_threshold_missing_lower_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: None, // deliberately omitted
            upper: Some(255.0),
            seed: Some("2,2,2".to_string()),
        });

        assert!(result.is_err(), "missing --lower must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--lower is required"),
            "error must name the missing argument, got: {msg}"
        );
    }

    // ── Negative: connected-threshold missing --upper ─────────────────────────

    #[test]
    fn test_segment_connected_threshold_missing_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: None, // deliberately omitted
            seed: Some("2,2,2".to_string()),
        });

        assert!(result.is_err(), "missing --upper must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--upper is required"),
            "error must name the missing argument, got: {msg}"
        );
    }

    // ── Negative: connected-threshold missing --seed ──────────────────────────

    #[test]
    fn test_segment_connected_threshold_missing_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: None, // deliberately omitted
        });

        assert!(result.is_err(), "missing --seed must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--seed is required"),
            "error must name the missing argument, got: {msg}"
        );
    }

    // ── Negative: connected-threshold lower > upper ───────────────────────────

    #[test]
    fn test_segment_connected_threshold_lower_gt_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(255.0),
            upper: Some(100.0), // lower > upper
            seed: Some("2,2,2".to_string()),
        });

        assert!(result.is_err(), "lower > upper must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must be \u{2264}")
                || msg.contains("must be <=")
                || msg.contains('\u{2264}'),
            "error must explain the bound constraint, got: {msg}"
        );
    }

    // ── Negative: out-of-bounds seed returns error ────────────────────────────

    #[test]
    fn test_segment_connected_threshold_out_of_bounds_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("99,99,99".to_string()), // far out of [5,5,5] bounds
        });

        assert!(result.is_err(), "out-of-bounds seed must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of bounds"),
            "error must explain the bounds problem, got: {msg}"
        );
    }

    // ── Negative: malformed seed string returns error ─────────────────────────

    #[test]
    fn test_segment_malformed_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("2,2".to_string()), // only two components
        });

        assert!(result.is_err(), "malformed seed must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Z,Y,X"),
            "error must explain the expected format, got: {msg}"
        );
    }

    // ── Negative: multi-otsu classes < 2 returns error ───────────────────────

    #[test]
    fn test_segment_multi_otsu_classes_lt_2_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "multi-otsu".to_string(),
            classes: 1, // invalid
            lower: None,
            upper: None,
            seed: None,
        });

        assert!(result.is_err(), "classes < 2 must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("≥ 2"),
            "error must state the minimum class count, got: {msg}"
        );
    }

    // ── Boundary: parse_seed correct output ───────────────────────────────────

    #[test]
    fn test_parse_seed_valid_input() {
        let seed = parse_seed("4,5,6").unwrap();
        assert_eq!(seed, [4, 5, 6]);
    }

    #[test]
    fn test_parse_seed_with_spaces() {
        let seed = parse_seed("1, 2, 3").unwrap();
        assert_eq!(seed, [1, 2, 3]);
    }

    #[test]
    fn test_parse_seed_too_few_components_returns_error() {
        assert!(parse_seed("1,2").is_err());
    }

    #[test]
    fn test_parse_seed_non_numeric_component_returns_error() {
        assert!(parse_seed("1,two,3").is_err());
    }

    #[test]
    fn test_parse_seed_negative_component_returns_error() {
        // Negative values cannot be parsed as usize.
        assert!(parse_seed("1,-2,3").is_err());
    }
}
