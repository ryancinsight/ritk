//! `parcellate` subcommand group — labelling a subject brain by region.
//!
//! Registration, label warping, and fusion belong to
//! [`ritk_registration::parcellation`]; this module parses arguments, reads the
//! inputs, and writes the label volume. It holds no parcellation logic of its
//! own.
//!
//! # Where this sits
//!
//! It closes the middle of the connectomics pipeline. `tract dti` produces
//! streamlines and `tract connectome` consumes a label volume, but nothing
//! produced that volume from the command line:
//!
//! ```text
//! ritk parcellate atlas  --subject T1 --atlas ... --output parc.nii
//! ritk tract dti         --dwi ...             --output tracks.tck
//! ritk tract connectome  --tractogram tracks.tck --labels parc.nii --output matrix.json
//! ```
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: ritk-cli is the application output layer"
)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ritk_parcellation::storage::label_from_stored;
use ritk_registration::{
    parcellate_with_atlas_set, AtlasParcellationConfig, LabelFusion, LabelledAtlas,
};
use tracing::info;

use super::{read_image, write_image, Backend};

/// Label a subject brain by anatomical region.
#[derive(clap::Args, Debug)]
pub struct ParcellateArgs {
    #[command(subcommand)]
    pub command: ParcellateCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum ParcellateCommand {
    /// Parcellate by deforming labelled atlases onto the subject.
    Atlas(AtlasArgs),
}

/// Parcellate a subject from one or more labelled atlases.
///
/// Each atlas is registered to the subject independently, its labels warped
/// through the recovered deformation, and the results fused. Every atlas must
/// already lie on the subject's grid — resample it first, since a registration
/// cannot recover a resampling.
///
/// # Why more than one atlas
///
/// A single atlas transfers not only its anatomy but its idiosyncrasies, and
/// wherever the registration is locally wrong the labels are locally wrong with
/// no signal that they are. Fusing several independently labelled brains means
/// an error has to be shared by a majority to survive, and `--agreement` writes
/// out where they disagreed.
#[derive(clap::Args, Debug)]
pub struct AtlasArgs {
    /// Subject image to parcellate. Supplies both the intensity the
    /// registration matches and the grid the result lands on.
    #[arg(long)]
    pub subject: PathBuf,

    /// Atlas intensity image. Repeat for a multi-atlas run; pairs positionally
    /// with `--atlas-labels`.
    #[arg(long = "atlas-intensity", required = true)]
    pub atlas_intensity: Vec<PathBuf>,

    /// Atlas label volume. Repeat once per `--atlas-intensity`, in the same
    /// order.
    #[arg(long = "atlas-labels", required = true)]
    pub atlas_labels: Vec<PathBuf>,

    /// Parcellation output — a label volume on the subject's grid.
    #[arg(long)]
    pub output: PathBuf,

    /// Per-voxel agreement output.
    ///
    /// The fraction of atlases that voted for the winning label, in `[0, 1]`.
    /// Low values mark where the result is a coin toss between neighbouring
    /// parcels — usually the boundaries, which is exactly where a connectome's
    /// streamline endpoints land. A single atlas has nothing to disagree with,
    /// so every voxel reads 1.
    #[arg(long)]
    pub agreement: Option<PathBuf>,

    /// How several atlases' votes are combined.
    #[arg(long, value_enum, default_value_t = Fusion::Majority)]
    pub fusion: Fusion,

    /// Registration iterations per resolution level, coarsest first.
    ///
    /// Defaults to `40,20,10`. Fewer is faster and leaves more of the
    /// deformation unclosed; the agreement map and the reported
    /// cross-correlation are what say whether it converged.
    #[arg(long, value_delimiter = ',')]
    pub iterations: Option<Vec<usize>>,
}

/// How multiple atlases' labels are combined.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum Fusion {
    /// The label most atlases agree on; ties to the smaller label.
    ///
    /// Treats every atlas as equally trustworthy everywhere — right when they
    /// are interchangeable, wrong when some registered better than others in a
    /// particular region.
    Majority,
    /// Weighted voting, each atlas weighted by how well its warped intensities
    /// match the subject locally.
    ///
    /// Lets a well-registered atlas outvote a poorly registered one in the
    /// region where that is true, at the cost of a patch comparison and a small
    /// dense solve per voxel.
    Joint,
}

/// Execute `parcellate`.
///
/// # Errors
///
/// Propagates argument, IO, registration, and fusion failures.
pub fn run(args: ParcellateArgs) -> Result<()> {
    match args.command {
        ParcellateCommand::Atlas(args) => atlas(args),
    }
}

/// Execute `parcellate atlas`.
fn atlas(args: AtlasArgs) -> Result<()> {
    anyhow::ensure!(
        args.atlas_intensity.len() == args.atlas_labels.len(),
        "each atlas needs both an intensity and a label volume: got {} intensities and {} label volumes",
        args.atlas_intensity.len(),
        args.atlas_labels.len()
    );

    info!(
        "parcellate atlas: subject={} atlases={} output={}",
        args.subject.display(),
        args.atlas_intensity.len(),
        args.output.display()
    );

    let subject =
        read_image(&args.subject).with_context(|| format!("reading {}", args.subject.display()))?;
    let voxels = subject.shape().iter().product::<usize>();

    let atlases = args
        .atlas_intensity
        .iter()
        .zip(&args.atlas_labels)
        .map(|(intensity_path, labels_path)| read_atlas(intensity_path, labels_path, voxels))
        .collect::<Result<Vec<_>>>()?;

    let mut config = AtlasParcellationConfig::default();
    if let Some(iterations) = args.iterations {
        anyhow::ensure!(
            !iterations.is_empty(),
            "--iterations needs at least one level"
        );
        config.registration.num_levels = iterations.len();
        config.registration.iterations_per_level = iterations;
    }
    config.fusion = match args.fusion {
        Fusion::Majority => LabelFusion::MajorityVote,
        Fusion::Joint => LabelFusion::JointLabelFusion(Default::default()),
    };

    let result = parcellate_with_atlas_set(&subject, &atlases, &config)
        .context("parcellating the subject")?;

    write_labels(&args.output, &result.parcellation, &subject)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "wrote {}: {} regions over {voxels} voxels",
        args.output.display(),
        result.parcellation.region_count()
    );
    // The registration quality is reported unconditionally: a value far below
    // the others marks an atlas that did not register, whose labels are then
    // noise in the vote rather than a second opinion.
    for (index, quality) in result.registration_quality.iter().enumerate() {
        println!("  atlas {index}: final cross-correlation {quality:.4}");
    }
    report_agreement(&result.agreement);

    if let Some(path) = &args.agreement {
        write_agreement(path, &result.agreement, &subject)
            .with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    Ok(())
}

/// Read one atlas and check it covers the subject's grid.
fn read_atlas(intensity: &Path, labels: &Path, voxels: usize) -> Result<LabelledAtlas> {
    let intensity_image =
        read_image(intensity).with_context(|| format!("reading {}", intensity.display()))?;
    let label_image =
        read_image(labels).with_context(|| format!("reading {}", labels.display()))?;

    let intensity_data = intensity_image
        .data_slice()
        .context("atlas intensity volume is not contiguous in host memory")?;
    let label_data = label_image
        .data_slice()
        .context("atlas label volume is not contiguous in host memory")?;

    anyhow::ensure!(
        intensity_data.len() == voxels && label_data.len() == voxels,
        "atlas {} / {} must lie on the subject's grid ({voxels} voxels), got {} and {}. \
         Resample it first — a registration cannot recover a resampling.",
        intensity.display(),
        labels.display(),
        intensity_data.len(),
        label_data.len()
    );

    Ok(LabelledAtlas {
        intensity: intensity_data.to_vec(),
        labels: label_data.iter().copied().map(label_from_stored).collect(),
        region_names: Vec::new(),
    })
}

/// Write the parcellation as a label volume on the subject's grid.
fn write_labels(
    path: &Path,
    parcellation: &ritk_parcellation::Parcellation,
    subject: &ritk_image::Image<f32, Backend, 3>,
) -> Result<()> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "NIfTI stores geometry in f32, which is the format's own precision"
    )]
    let narrow = |values: [f64; 3]| values.map(|value| value as f32);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "NIfTI stores geometry in f32, which is the format's own precision"
    )]
    let direction = subject.direction().to_row_major().map(|value| value as f32);

    ritk_nifti::write_nifti_labels(
        path,
        parcellation.labels(),
        subject.shape(),
        narrow(subject.origin().to_array()),
        narrow(subject.spacing().to_array()),
        direction,
    )
    .map_err(|error| anyhow::anyhow!("{error}"))
}

/// Write the agreement map as an ordinary image on the subject's grid.
fn write_agreement(
    path: &Path,
    agreement: &[f32],
    subject: &ritk_image::Image<f32, Backend, 3>,
) -> Result<()> {
    let device = Backend::default();
    let tensor = ritk_image::tensor::Tensor::<f32, Backend>::from_slice_on(
        subject.shape(),
        agreement,
        &device,
    );
    let image = ritk_image::Image::new(
        tensor,
        *subject.origin(),
        *subject.spacing(),
        *subject.direction(),
    )?;
    let format = super::infer_format(path)
        .with_context(|| format!("cannot infer an image format from {}", path.display()))?;
    write_image(path, &image, format)
}

/// Print how much of the volume the atlases agreed on.
fn report_agreement(agreement: &[f32]) {
    if agreement.is_empty() {
        return;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "voxel counts stay far below f64's exact-integer range"
    )]
    let total = agreement.len() as f64;
    let unanimous = agreement.iter().filter(|value| **value >= 1.0).count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "voxel counts stay far below f64's exact-integer range"
    )]
    let share = unanimous as f64 / total;
    println!("  {:.1}% of voxels unanimous", 100.0 * share);
}

#[cfg(test)]
mod tests;
