//! `tract connectome` — reduce a tractogram and a parcellation to a graph.
//!
//! The construction and every measure belong to [`ritk_connectome`]; this module
//! parses arguments, reads the two inputs, and writes the result. It holds no
//! graph logic of its own.
//!
//! # What the two inputs must share
//!
//! A streamline endpoint is looked up in the parcellation by *physical*
//! position, so both files must express that position the same way. Streamlines
//! written by `tract dti` are already in their reference image's physical frame,
//! and a label volume read here carries its own affine, so two volumes that were
//! acquired or resampled onto the same grid agree by construction. Two that were
//! not do not, and nothing in either file says so — a label volume from a
//! different session will produce a full, plausible, and entirely wrong
//! connectome. Registering the parcellation onto the diffusion reference first
//! is the caller's responsibility.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use gaia::Polyline;
use ritk_connectome::{
    build_connectivity_matrix, ConnectomeConfig, EdgeWeighting, EndpointAssignment, GraphMeasures,
};
use ritk_parcellation::Parcellation;
use ritk_registration::parcellation_from_labels;
use tracing::info;

/// Build a region connectome from streamlines and a parcellation.
#[derive(clap::Args, Debug)]
pub struct ConnectomeArgs {
    /// Streamlines in the parcellation's physical frame (`.tck` or `.trk`).
    #[arg(long)]
    pub tractogram: PathBuf,

    /// Parcellation label volume, as any natively readable image format.
    #[arg(long)]
    pub labels: PathBuf,

    /// Endpoint assignment radius in mm.
    ///
    /// Zero assigns an endpoint only to the region it lands in, which discards
    /// every streamline terminating in white matter — ordinarily most of a
    /// tractogram, since tracking stops at the grey/white boundary while a
    /// cortical parcellation labels only grey matter. A few millimetres
    /// recovers those; too many reach across a sulcus into a parcel no fibre
    /// entered.
    #[arg(long, default_value_t = 2.0, value_parser = parse_radius)]
    pub assignment_radius: f64,

    /// What an edge weight counts.
    #[arg(long, value_enum, default_value_t = Weighting::Count)]
    pub weighting: Weighting,

    /// Connectivity matrix output (JSON).
    #[arg(long)]
    pub output: PathBuf,

    /// Graph measures output (JSON). Omit to skip computing them.
    #[arg(long)]
    pub measures: Option<PathBuf>,
}

/// Parse a search radius, rejecting anything that is not a usable distance.
///
/// Clap's `value_parser` range support does not cover `f64`, and without a
/// check a negative radius would fall through to terminal assignment — silently
/// giving a caller who asked for a wide search the narrowest one there is.
fn parse_radius(value: &str) -> Result<f64, String> {
    let radius: f64 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a number"))?;
    if !radius.is_finite() || radius < 0.0 {
        return Err(format!(
            "assignment radius must be finite and nonnegative, got `{value}`"
        ));
    }
    Ok(radius)
}

/// Edge weighting, as a command-line choice.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum Weighting {
    /// Number of streamlines connecting the two regions.
    Count,
    /// Sum of reciprocal streamline lengths, which divides out the length
    /// dependence of tracking.
    InverseLength,
    /// Streamline count over the summed region volumes, which divides out the
    /// region-size dependence.
    InverseNodeVolume,
    /// Mean length of the connecting streamlines — a description of the
    /// pathway rather than a count of it.
    MeanLength,
}

impl From<Weighting> for EdgeWeighting {
    fn from(value: Weighting) -> Self {
        match value {
            Weighting::Count => Self::StreamlineCount,
            Weighting::InverseLength => Self::InverseLength,
            Weighting::InverseNodeVolume => Self::InverseNodeVolume,
            Weighting::MeanLength => Self::MeanLength,
        }
    }
}

/// Execute `tract connectome`.
///
/// # Errors
///
/// Propagates argument, IO, parcellation, and construction failures.
pub fn run(args: ConnectomeArgs) -> Result<()> {
    info!(
        "tract connectome: tractogram={} labels={} output={}",
        args.tractogram.display(),
        args.labels.display(),
        args.output.display()
    );

    let parcellation = read_parcellation(&args.labels)?;
    let streamlines = read_streamlines(&args.tractogram)?;
    info!(
        "read {} regions and {} streamlines",
        parcellation.region_count(),
        streamlines.len()
    );

    let config = ConnectomeConfig::new()
        .with_assignment(if args.assignment_radius > 0.0 {
            EndpointAssignment::RadialSearch {
                radius_mm: args.assignment_radius,
            }
        } else {
            EndpointAssignment::Terminal
        })
        .with_weighting(args.weighting.into());

    let matrix = build_connectivity_matrix(&parcellation, &streamlines, &config)
        .context("building the connectivity matrix")?;

    write_json(
        &args.output,
        &matrix.to_json().context("encoding the matrix")?,
    )
    .with_context(|| format!("writing {}", args.output.display()))?;

    let accounting = matrix.accounting();
    // The accounting is reported unconditionally rather than on request: a
    // matrix built from a tractogram that was mostly discarded is a different
    // claim from one that was mostly kept, and the weights do not say which.
    println!(
        "wrote {}: {} regions, {} edges, density {:.4}",
        args.output.display(),
        matrix.region_count(),
        matrix.edge_count(),
        matrix.density()
    );
    println!(
        "  streamlines: {} considered, {} between regions ({:.1}%), \
         {} within one region, {} unassigned",
        accounting.total,
        accounting.assigned,
        100.0 * accounting.assigned_fraction(),
        accounting.intra_region,
        accounting.unassigned
    );

    if let Some(path) = &args.measures {
        let measures = matrix.measures();
        write_json(
            path,
            &serde_json::to_string(&measures).context("encoding the graph measures")?,
        )
        .with_context(|| format!("writing {}", path.display()))?;
        report(&measures);
        println!("wrote {}", path.display());
    }

    Ok(())
}

/// Print the whole-graph measures.
fn report(measures: &GraphMeasures) {
    println!(
        "  clustering {:.3}, global efficiency {:.3}, betweenness {:.4}",
        GraphMeasures::mean(measures.clustering()),
        measures.global_efficiency(),
        GraphMeasures::mean(measures.betweenness())
    );
    match measures.characteristic_path_length() {
        Some(length) => println!(
            "  characteristic path length {length:.3} over {:.1}% of node pairs",
            100.0 * measures.reachable_pair_fraction()
        ),
        None => println!("  no pair of regions is connected"),
    }
    println!(
        "  {} communities, modularity {:.3}",
        measures.communities().count(),
        measures.communities().modularity()
    );
    let components = measures.component_sizes();
    if components.len() > 1 {
        // Worth surfacing rather than burying: an isolated region makes every
        // path measure describe a graph the caller may not think they have.
        println!(
            "  {} components, largest {} of {} regions",
            components.len(),
            components.first().copied().unwrap_or(0),
            measures.node_count()
        );
    }
}

/// Read a label volume and place it on its own grid.
///
/// The volume is read as an ordinary image so that its affine comes with it: a
/// label reader returning only the raw array would leave the parcellation with
/// no way to answer where a voxel sits, which is the only question a streamline
/// endpoint asks of it.
fn read_parcellation(path: &Path) -> Result<Parcellation> {
    let image =
        ritk_io::read_image_native(path).with_context(|| format!("reading {}", path.display()))?;
    let voxels = image
        .data_slice()
        .context("the label volume is not contiguous in host memory")?;

    let labels: Box<[u32]> = voxels
        .iter()
        .map(|value| {
            if *value <= 0.0 {
                0
            } else {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the value is positive and a label volume holds integers"
                )]
                let label = value.round() as u32;
                label
            }
        })
        .collect();

    parcellation_from_labels(labels, &image, Vec::new())
        .with_context(|| format!("interpreting {} as a parcellation", path.display()))
}

/// Read streamlines from a `.tck` or `.trk` file.
fn read_streamlines(path: &Path) -> Result<Vec<Polyline<f64>>> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .with_context(|| {
            format!(
                "cannot infer a track format from {}: expected .tck or .trk",
                path.display()
            )
        })?;

    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);

    match extension.as_str() {
        "tck" => Ok(ritk_tck::TckTractogram::read(&mut reader)
            .with_context(|| format!("reading {}", path.display()))?
            .streamlines),
        "trk" => Ok(ritk_trk::TrkTractogram::read(&mut reader)
            .with_context(|| format!("reading {}", path.display()))?
            .streamlines),
        other => anyhow::bail!(
            "cannot read a connectome from a .{other} tractogram: expected .tck or .trk"
        ),
    }
}

fn write_json(path: &Path, contents: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(contents.as_bytes())?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests;
