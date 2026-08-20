//! Parcellate a synthetic subject from three labelled atlases.
//!
//! The example exists to make the atlas-propagation chapter's claims runnable
//! and checkable. Everything is synthesised, so the correct parcellation is
//! known exactly and the example reports Dice against it rather than asserting
//! that the pipeline merely returned something.
//!
//! # What it demonstrates
//!
//! Each atlas is the subject's anatomy displaced by a different whole-voxel
//! shift, which is the situation atlas propagation exists for: the anatomy
//! corresponds but the coordinates do not, and the registration has to recover
//! the correspondence before the labels mean anything. One atlas additionally
//! mislabels a structure, so majority voting has something to outvote and the
//! agreement map has something to report — a fused result that never disagreed
//! would say nothing about fusion.
//!
//! A second run with none mislabelled is the control. The only thing separating
//! those atlases is each one's own registration error, which is where the two
//! fusion rules' assumptions part company.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p ritk-registration --example book_parcellation
//! ```

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use coeus_core::SequentialBackend;

use ritk_image::Image;
use ritk_registration::{
    parcellate_with_atlas_set, AtlasParcellationConfig, LabelFusion, LabelledAtlas,
};
use ritk_spatial::{Direction, Point, Spacing};

type Backend = SequentialBackend;

/// Volume shape, outermost axis first. Unequal on every axis: a cubic volume on
/// an isotropic grid cannot expose an axis-order defect, because reversing the
/// axes leaves every array the same length and every voxel the same size.
const SHAPE: [usize; 3] = [16, 20, 24];
/// Voxel size in mm, in the same axis order — anisotropic for the same reason.
const SPACING: [f64; 3] = [2.0, 1.5, 1.0];

/// Background intensity, and the intensity of each labelled structure.
const BACKGROUND_INTENSITY: f32 = 10.0;
const STRUCTURE_INTENSITY: [f32; 3] = [90.0, 140.0, 60.0];

/// A short registration schedule. The displacement to recover is one to two
/// voxels, so a long schedule would spend its iterations confirming a
/// correspondence already found — and the example carries a runtime budget.
const ITERATIONS: [usize; 3] = [20, 12, 6];

/// Slice, along the outermost axis, drawn in the figure.
const FIGURE_SLICE: usize = SHAPE[0] / 2;

fn voxels() -> usize {
    SHAPE[0] * SHAPE[1] * SHAPE[2]
}

fn at(i: usize, j: usize, k: usize) -> usize {
    (i * SHAPE[1] + j) * SHAPE[2] + k
}

/// The three structures, as half-open index ranges per axis.
///
/// They differ in size, position, and aspect ratio so that a parcellation which
/// swapped two of them, or shifted all of them together, is visible in the Dice
/// scores rather than averaging out. Their ranges on the outermost axis overlap
/// so that one slice shows all three — a figure that caught only one structure
/// would be a picture of the slice choice rather than of the result.
fn structures() -> [[std::ops::Range<usize>; 3]; 3] {
    [
        [3..9, 3..8, 3..10],
        [4..11, 3..8, 14..21],
        [5..12, 12..18, 6..17],
    ]
}

/// The subject's true parcellation: label `n + 1` inside structure `n`.
fn ground_truth() -> Vec<u32> {
    let mut labels = vec![0_u32; voxels()];
    for (index, ranges) in structures().into_iter().enumerate() {
        let label = u32::try_from(index).expect("three structures fit in u32") + 1;
        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                for k in ranges[2].clone() {
                    labels[at(i, j, k)] = label;
                }
            }
        }
    }
    labels
}

/// An intensity volume built from a label volume, one intensity per structure.
///
/// The registration matches intensities, so the structures must be
/// distinguishable by intensity as well as by position — three identical bright
/// blocks would let a registration slide one onto another with no penalty.
fn intensity_from(labels: &[u32]) -> Vec<f32> {
    labels
        .iter()
        .map(|label| match label {
            0 => BACKGROUND_INTENSITY,
            n => STRUCTURE_INTENSITY[(*n as usize - 1) % STRUCTURE_INTENSITY.len()],
        })
        .collect()
}

/// Shift a volume by whole voxels, filling what moves in from outside with the
/// background label.
///
/// Whole voxels rather than a subvoxel warp because a label volume cannot be
/// interpolated: shifting by half a voxel would require inventing labels at the
/// boundary, which is the very thing the chapter says never to do.
fn shifted(labels: &[u32], shift: [isize; 3]) -> Vec<u32> {
    let mut out = vec![0_u32; voxels()];
    let extent = SHAPE.map(|n| isize::try_from(n).expect("a test extent fits in isize"));
    for i in 0..SHAPE[0] {
        for j in 0..SHAPE[1] {
            for k in 0..SHAPE[2] {
                let source = [
                    isize::try_from(i).expect("index fits") - shift[0],
                    isize::try_from(j).expect("index fits") - shift[1],
                    isize::try_from(k).expect("index fits") - shift[2],
                ];
                if source
                    .iter()
                    .zip(extent)
                    .any(|(value, bound)| *value < 0 || *value >= bound)
                {
                    continue;
                }
                let [si, sj, sk] = source.map(|value| {
                    usize::try_from(value).expect("bounds were checked immediately above")
                });
                out[at(i, j, k)] = labels[at(si, sj, sk)];
            }
        }
    }
    out
}

/// Build an image on the subject's grid from a flat volume.
fn image(values: Vec<f32>) -> Result<Image<f32, Backend, 3>> {
    let device = Backend::default();
    let tensor = ritk_image::tensor::Tensor::<f32, Backend>::from_slice_on(SHAPE, &values, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new(SPACING),
        Direction::identity(),
    )
    .context("building an image on the example grid")
}

/// Dice overlap of one label between a result and the truth.
///
/// Dice rather than a raw voxel count because the structures differ in size:
/// counting agreements would let the largest structure hide a total failure on
/// the smallest.
fn dice(result: &[u32], truth: &[u32], label: u32) -> f64 {
    let mut intersection = 0_usize;
    let mut in_result = 0_usize;
    let mut in_truth = 0_usize;
    for (left, right) in result.iter().zip(truth) {
        if *left == label {
            in_result += 1;
        }
        if *right == label {
            in_truth += 1;
        }
        if *left == label && *right == label {
            intersection += 1;
        }
    }
    let total = in_result + in_truth;
    if total == 0 {
        return 1.0;
    }
    2.0 * intersection as f64 / total as f64
}

/// One atlas: a displaced copy of the anatomy, and the labels drawn on it.
struct Atlas {
    /// Structure geometry, from which the intensity volume is built.
    anatomy: Vec<u32>,
    /// The labels this atlas asserts — the same volume unless it is wrong.
    labels: Vec<u32>,
    shift: [isize; 3],
}

/// The three atlases: each displaced differently, and the third mislabelled.
///
/// The third atlas has the *same anatomy* as the others; only its label volume
/// swaps 2 and 3. That distinction is the whole point. Swapping its intensities
/// too would make it a different brain, which a registration is entitled to
/// match poorly — and then the fusion would be choosing between atlases that
/// genuinely disagree about anatomy rather than about naming. Mislabelling is
/// the case fusion exists for: every atlas fits the subject equally well, and
/// only the labels are in dispute.
fn atlases(truth: &[u32]) -> [Atlas; 3] {
    let swap = |labels: &[u32]| -> Vec<u32> {
        labels
            .iter()
            .map(|label| match label {
                2 => 3,
                3 => 2,
                other => *other,
            })
            .collect()
    };
    let displaced = |shift: [isize; 3]| shifted(truth, shift);
    [
        Atlas {
            anatomy: displaced([1, 0, -1]),
            labels: displaced([1, 0, -1]),
            shift: [1, 0, -1],
        },
        Atlas {
            anatomy: displaced([-1, 1, 1]),
            labels: displaced([-1, 1, 1]),
            shift: [-1, 1, 1],
        },
        Atlas {
            anatomy: displaced([0, -1, 2]),
            labels: swap(&displaced([0, -1, 2])),
            shift: [0, -1, 2],
        },
    ]
}

fn main() -> Result<()> {
    let truth = ground_truth();
    let subject = image(intensity_from(&truth))?;

    let atlas_volumes = atlases(&truth);
    let labelled: Vec<LabelledAtlas> = atlas_volumes
        .iter()
        .map(|atlas| LabelledAtlas {
            intensity: intensity_from(&atlas.anatomy),
            labels: atlas.labels.clone(),
            region_names: vec![
                (1, "anterior".to_owned()),
                (2, "posterior".to_owned()),
                (3, "lateral".to_owned()),
            ],
        })
        .collect();

    let mut config = AtlasParcellationConfig::default();
    config.registration.num_levels = ITERATIONS.len();
    config.registration.iterations_per_level = ITERATIONS.to_vec();

    let majority = parcellate_with_atlas_set(&subject, &labelled, &config)
        .context("parcellating by majority vote")?;

    config.fusion = LabelFusion::JointLabelFusion(Default::default());
    let joint = parcellate_with_atlas_set(&subject, &labelled, &config)
        .context("parcellating by joint label fusion")?;

    // Interchangeable atlases are the case where weighting has nothing to work
    // with. All three fit the subject equally well, so joint fusion's weights
    // come out near-equal and it ends up following whichever single atlas
    // matches best locally — inheriting that one atlas's registration error,
    // where voting averages three independent ones. Removing the mislabelling
    // is what isolates that, since nothing else is then in dispute.
    let interchangeable: Vec<LabelledAtlas> = atlas_volumes
        .iter()
        .map(|atlas| LabelledAtlas {
            intensity: intensity_from(&atlas.anatomy),
            labels: atlas.anatomy.clone(),
            region_names: Vec::new(),
        })
        .collect();
    config.fusion = LabelFusion::MajorityVote;
    let control_majority = parcellate_with_atlas_set(&subject, &interchangeable, &config)
        .context("parcellating interchangeable atlases by majority vote")?;
    config.fusion = LabelFusion::JointLabelFusion(Default::default());
    let control_joint = parcellate_with_atlas_set(&subject, &interchangeable, &config)
        .context("parcellating interchangeable atlases by joint label fusion")?;

    report(&majority, &joint, &truth, &atlas_volumes);
    report_control(&control_majority, &control_joint, &truth);

    let figure = render(&truth, majority.parcellation.labels(), &majority.agreement);
    let path = figure_path();
    std::fs::write(&path, figure).with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {}", path.display());

    Ok(())
}

/// Print what the run actually measured, against the known answer.
fn report(
    majority: &ritk_registration::ParcellationResult,
    joint: &ritk_registration::ParcellationResult,
    truth: &[u32],
    atlas_volumes: &[Atlas],
) {
    println!(
        "subject grid {SHAPE:?} at {SPACING:?} mm, {} voxels",
        voxels()
    );

    println!("\natlases, before any registration:");
    for (index, atlas) in atlas_volumes.iter().enumerate() {
        let scores: Vec<String> = (1..=3)
            .map(|label| format!("{:.2}", dice(&atlas.labels, truth, label)))
            .collect();
        println!(
            "  {index}: shifted by {:?} voxels, Dice vs truth [{}]",
            atlas.shift,
            scores.join(", ")
        );
    }

    for (name, result) in [("majority vote", majority), ("joint fusion", joint)] {
        println!("\n{name}:");
        let scores: Vec<String> = (1..=3)
            .map(|label| format!("{:.2}", dice(result.parcellation.labels(), truth, label)))
            .collect();
        println!("  Dice vs truth [{}]", scores.join(", "));
        println!(
            "  {} regions, mean agreement {:.3}",
            result.parcellation.region_count(),
            mean(&result.agreement)
        );
        let quality: Vec<String> = result
            .registration_quality
            .iter()
            .map(|value| format!("{value:.4}"))
            .collect();
        println!(
            "  final cross-correlation per atlas [{}]",
            quality.join(", ")
        );
    }

    // Structures 2 and 3 are the ones the third atlas votes against, so their
    // agreement must be lower than structure 1's. If it is not, the map is
    // reporting something other than the vote.
    println!("\nmean agreement inside each structure, majority vote:");
    for (index, ranges) in structures().into_iter().enumerate() {
        let mut inside = Vec::new();
        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                for k in ranges[2].clone() {
                    inside.push(majority.agreement[at(i, j, k)]);
                }
            }
        }
        println!("  structure {}: {:.3}", index + 1, mean(&inside));
    }
}

/// Report the interchangeable-atlas control.
///
/// No atlas is mislabelled here, so nothing is in dispute except each atlas's
/// own registration error. This is where the two rules' assumptions separate:
/// voting averages three independent errors, while weighting concentrates on
/// whichever atlas matches best locally and inherits that one's error. Neither
/// is wrong — they answer different questions, and the example exists so the
/// difference is a number rather than a claim.
fn report_control(
    majority: &ritk_registration::ParcellationResult,
    joint: &ritk_registration::ParcellationResult,
    truth: &[u32],
) {
    println!();
    println!("control — interchangeable atlases, none mislabelled:");
    for (name, result) in [("majority vote", majority), ("joint fusion", joint)] {
        let scores: Vec<String> = (1..=3)
            .map(|label| format!("{:.2}", dice(result.parcellation.labels(), truth, label)))
            .collect();
        println!("  {name}: Dice vs truth [{}]", scores.join(", "));
    }
    println!("  Voting wins when the atlases are interchangeable, because there is");
    println!("  then no local quality difference for the weights to exploit.");
}

fn mean(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
}

// ── Figure ───────────────────────────────────────────────────────────────

/// Colour per label. Background is a dark neutral so the structures read
/// against it without implying a fourth region.
const LABEL_COLOUR: [&str; 4] = ["#20242c", "#e4572e", "#17bebb", "#ffc914"];
/// Side of one voxel in the figure, in SVG units.
const CELL: usize = 8;

/// Render the mid-slice as truth, result, and agreement panels.
///
/// A figure rather than only numbers because the failure this pipeline has is
/// spatial: a parcellation can score well overall while being wrong along one
/// boundary, and a Dice column cannot show which boundary.
fn render(truth: &[u32], result: &[u32], agreement: &[f32]) -> String {
    let (rows, columns) = (SHAPE[1], SHAPE[2]);
    let panel = columns * CELL;
    let gap = CELL * 2;
    let margin = CELL;
    let width = panel * 3 + gap * 2 + margin * 2;
    let height = rows * CELL + 24 + margin;

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"#
    );
    let _ = write!(
        svg,
        r##"<rect width="{width}" height="{height}" fill="#11131a"/>"##
    );

    for (index, title) in ["ground truth", "majority vote", "agreement"]
        .into_iter()
        .enumerate()
    {
        let x0 = margin + index * (panel + gap);
        let _ = write!(
            svg,
            r##"<text x="{x0}" y="14" fill="#c9d1d9" font-family="sans-serif" font-size="11">{title}</text>"##
        );
        for j in 0..rows {
            for k in 0..columns {
                let voxel = at(FIGURE_SLICE, j, k);
                let fill = match index {
                    0 => LABEL_COLOUR[truth[voxel] as usize % LABEL_COLOUR.len()].to_owned(),
                    1 => LABEL_COLOUR[result[voxel] as usize % LABEL_COLOUR.len()].to_owned(),
                    _ => grey(agreement[voxel]),
                };
                let x = x0 + k * CELL;
                let y = 20 + j * CELL;
                let _ = write!(
                    svg,
                    r#"<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{fill}"/>"#
                );
            }
        }
    }
    svg.push_str("</svg>\n");
    svg
}

/// Grey level for an agreement value in `[0, 1]`.
fn grey(value: f32) -> String {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped into [0, 1] immediately before the cast"
    )]
    let level = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{level:02x}{level:02x}{level:02x}")
}

fn figure_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/book/figures/atlas_parcellation.svg")
}
