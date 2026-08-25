//! Report NIfTI geometry and index-to-world coordinates for a CT/MR pair.
//! Pass the two NIfTI paths explicitly so the example is portable across datasets.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: example/test diagnostic output"
)]
use anyhow::{Context, Result};
use coeus_core::SequentialBackend;
use ritk_image::{grid, Image};
use ritk_io::{format::nifti::native::NiftiReader, ImageReader};
use std::path::{Path, PathBuf};

type B = SequentialBackend;

fn dump(name: &str, path: &Path, img: &Image<f32, B, 3>) {
    let backend = B::default();
    let shape = img.shape(); // [d0, d1, d2] = [z, y, x]
    println!("=== {name}: {}", path.display());
    println!(" shape    {:?}", shape);
    println!(" spacing  {:?}", img.spacing().to_array());
    println!(" origin   {:?}", img.origin());
    println!(" direction {:?}", img.direction());

    // The native registration path uses the same grid and index-to-world map.
    let grid = grid::generate_grid::<f32, B, 3>(shape, &backend);
    let world = img.index_to_world_native(&grid).as_slice().to_vec();
    let (ny, nx) = (shape[1], shape[2]);
    for (x, y, z) in [(0usize, 0usize, 0usize), (255, 255, 20), (100, 200, 15)] {
        if z < shape[0] && y < ny && x < nx {
            let flat = z * ny * nx + y * nx + x;
            let w = &world[flat * 3..flat * 3 + 3];
            println!(
                "  index (x={x}, y={y}, z={z}) -> world ({:.2}, {:.2}, {:.2})",
                w[0], w[1], w[2]
            );
        }
    }
}

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let ct_path = arguments.next().map(PathBuf::from).context(
        "usage: cargo run -p ritk-registration --example geometry_check -- <ct.nii.gz> <mr.nii.gz>",
    )?;
    let mri_path = arguments.next().map(PathBuf::from).context(
        "usage: cargo run -p ritk-registration --example geometry_check -- <ct.nii.gz> <mr.nii.gz>",
    )?;
    if arguments.next().is_some() {
        anyhow::bail!(
            "usage: cargo run -p ritk-registration --example geometry_check -- <ct.nii.gz> <mr.nii.gz>"
        );
    }

    let reader = NiftiReader::new(B::default());
    let ct: Image<f32, B, 3> = reader.read(&ct_path)?;
    let mri: Image<f32, B, 3> = reader.read(&mri_path)?;
    dump("CT", &ct_path, &ct);
    dump("MR", &mri_path, &mri);
    println!(
        "\nUse the reported spacing, origin, and direction as the registration geometry contract."
    );
    Ok(())
}
