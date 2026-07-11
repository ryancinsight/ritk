use coeus_autograd::{cat, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use ritk_model::transmorph::{TransMorphConfig, TransformIntegration};
use std::time::Instant;

fn main() {
    let config = TransMorphConfig::new(2, 12, 3)
        .with_window_size(4)
        .with_integration(TransformIntegration::Direct);
    let model = config.init::<MoiraiBackend>();
    let shape = [1, 1, 32, 32, 32];
    let fixed = volume(shape, 0);
    let moving = volume(shape, 1);
    let input = cat(&[&moving, &fixed], 1);

    let start = Instant::now();
    let output = model
        .forward(&input)
        .expect("example inputs satisfy the TransMorph contract");

    println!("Registration took: {:?}", start.elapsed());
    println!("Flow shape: {:?}", output.flow.tensor.shape());
    println!("Warped shape: {:?}", output.warped.tensor.shape());
}

fn volume(shape: [usize; 5], offset: usize) -> Var<f32, MoiraiBackend> {
    let [_batch, _channels, depth, height, width] = shape;
    let values = (0..depth)
        .flat_map(|z| {
            (0..height).flat_map(move |y| {
                (0..width).map(move |x| {
                    let dx = x.abs_diff(width / 2 + offset) as f32;
                    let dy = y.abs_diff(height / 2) as f32;
                    let dz = z.abs_diff(depth / 2) as f32;
                    (-(dx * dx + dy * dy + dz * dz) / 64.0).exp()
                })
            })
        })
        .collect::<Vec<_>>();
    Var::new(
        Tensor::from_slice_on(shape, &values, &MoiraiBackend::new()),
        false,
    )
}
