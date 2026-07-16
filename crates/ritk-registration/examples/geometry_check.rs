//! Verify ritk's NIfTI import + index→world against SimpleITK ground truth.
//! Prints geometry and index→world for fixed voxel indices; compare to sitk.
use coeus_core::SequentialBackend;
use ritk_image::{grid, Image};
use ritk_io::read_nifti;

type B = SequentialBackend;

fn dump(name: &str, img: &Image<f32, B, 3>) {
    let device = img.data()B::default();
    let shape = img.shape(); // [d0, d1, d2] = [z, y, x]
    println!("=== {name}");
    println!(" shape    {:?}", shape);
    println!(" spacing  {:?}", img.spacing().to_array());
    println!(" origin   {:?}", img.origin());
    println!(" direction {:?}", img.direction());

    // EXACTLY what the registration does: generate_grid -> index_to_world_tensor.
    let g = grid::generate_grid(shape, &device);
    let world = img
        .index_to_world_tensor(g)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let (ny, nx) = (shape[1], shape[2]);
    // Probe the sitk voxel (x,y,z)=(255,255,20): row-major [z,y,x] flat index.
    for (x, y, z) in [(0usize, 0usize, 0usize), (255, 255, 20), (100, 200, 15)] {
        if z < shape[0] && y < ny && x < nx {
            let flat = z * ny * nx + y * nx + x;
            let w = &world[flat * 3..flat * 3 + 3];
            println!(
                "  sitk(x={x},y={y},z={z}) -> ritk world ({:.2}, {:.2}, {:.2})",
                w[0], w[1], w[2]
            );
        }
    }
}

fn main() -> anyhow::Result<()> {
    let device = Default::default();
    let ct = read_nifti::<B, _>("D:/kwavers/leoneuro/data/brain_ct.nii.gz", &device)?;
    let mri = read_nifti::<B, _>("D:/kwavers/leoneuro/data/brain_mri_t1.nii.gz", &device)?;
    dump("brain_ct.nii.gz", &ct);
    dump("brain_mri_t1.nii.gz", &mri);
    println!(
        "\nsitk reference: CT idx(255,255,20)->(-103.11,-103.11,60.0); direction diag(-1,-1,1)"
    );
    Ok(())
}
