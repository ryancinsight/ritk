#![cfg(feature = "coeus")]

use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor as CoeusTensor;
use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;

type BurnB = NdArray<f32>;

fn get_coeus_data(t: &CoeusTensor<f32, MoiraiBackend>) -> Vec<f32> {
    use coeus_core::CpuAddressableStorage;
    t.storage().as_slice().to_vec()
}

fn get_burn_data(t: BurnTensor<BurnB, 2>) -> Vec<f32> {
    t.into_data().into_vec::<f32>().unwrap()
}

#[test]
fn differential_elementwise_add() {
    let shape = [2, 3];
    let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    // Coeus
    let backend = MoiraiBackend;
    let a_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &a_vals, &backend);
    let b_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &b_vals, &backend);
    let res_coeus = coeus_ops::add(&a_coeus, &b_coeus, &backend);
    let got_coeus = get_coeus_data(&res_coeus);

    // Burn
    let device = Default::default();
    let a_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(a_vals, burn::tensor::Shape::new(shape)), &device);
    let b_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(b_vals, burn::tensor::Shape::new(shape)), &device);
    let res_burn = a_burn.add(b_burn);
    let got_burn = get_burn_data(res_burn);

    assert_eq!(got_coeus, got_burn);
}

#[test]
fn differential_elementwise_sub() {
    let shape = [2, 3];
    let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    // Coeus
    let backend = MoiraiBackend;
    let a_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &a_vals, &backend);
    let b_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &b_vals, &backend);
    let res_coeus = coeus_ops::sub(&a_coeus, &b_coeus, &backend);
    let got_coeus = get_coeus_data(&res_coeus);

    // Burn
    let device = Default::default();
    let a_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(a_vals, burn::tensor::Shape::new(shape)), &device);
    let b_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(b_vals, burn::tensor::Shape::new(shape)), &device);
    let res_burn = a_burn.sub(b_burn);
    let got_burn = get_burn_data(res_burn);

    assert_eq!(got_coeus, got_burn);
}

#[test]
fn differential_elementwise_mul() {
    let shape = [2, 3];
    let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    // Coeus
    let backend = MoiraiBackend;
    let a_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &a_vals, &backend);
    let b_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &b_vals, &backend);
    let res_coeus = coeus_ops::mul(&a_coeus, &b_coeus, &backend);
    let got_coeus = get_coeus_data(&res_coeus);

    // Burn
    let device = Default::default();
    let a_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(a_vals, burn::tensor::Shape::new(shape)), &device);
    let b_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(b_vals, burn::tensor::Shape::new(shape)), &device);
    let res_burn = a_burn.mul(b_burn);
    let got_burn = get_burn_data(res_burn);

    assert_eq!(got_coeus, got_burn);
}

#[test]
fn differential_elementwise_div() {
    let shape = [2, 3];
    let a_vals = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let b_vals = vec![2.0, 4.0, 5.0, 8.0, 10.0, 12.0];

    // Coeus
    let backend = MoiraiBackend;
    let a_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &a_vals, &backend);
    let b_coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &b_vals, &backend);
    let res_coeus = coeus_ops::div(&a_coeus, &b_coeus, &backend);
    let got_coeus = get_coeus_data(&res_coeus);

    // Burn
    let device = Default::default();
    let a_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(a_vals, burn::tensor::Shape::new(shape)), &device);
    let b_burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(b_vals, burn::tensor::Shape::new(shape)), &device);
    let res_burn = a_burn.div(b_burn);
    let got_burn = get_burn_data(res_burn);

    assert_eq!(got_coeus, got_burn);
}

#[test]
fn differential_shape_ops() {
    let shape = [2, 3];
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Coeus reshape & transpose
    let backend = MoiraiBackend;
    let coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &vals, &backend);
    let reshaped_coeus = coeus.clone().reshape([3, 2]);
    assert_eq!(reshaped_coeus.shape(), &[3, 2]);
    let transposed_coeus = coeus.t();
    assert_eq!(transposed_coeus.shape(), &[3, 2]);

    // Burn reshape & transpose
    let device = Default::default();
    let burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(vals, burn::tensor::Shape::new(shape)), &device);
    let reshaped_burn = burn.clone().reshape([3, 2]);
    assert_eq!(reshaped_burn.shape().dims, [3, 2]);
    let transposed_burn = burn.transpose();
    assert_eq!(transposed_burn.shape().dims, [3, 2]);
}

#[test]
fn differential_reductions() {
    let shape = [2, 3];
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Coeus sum & mean
    let backend = MoiraiBackend;
    let coeus = CoeusTensor::<f32, _>::from_slice_on(shape, &vals, &backend);
    let got_sum_coeus = coeus_ops::sum(&coeus, &backend);
    let got_mean_coeus = coeus_ops::mean(&coeus, &backend);

    // Burn sum & mean
    let device = Default::default();
    let burn = BurnTensor::<BurnB, 2>::from_data(TensorData::new(vals, burn::tensor::Shape::new(shape)), &device);
    let sum_burn = burn.clone().sum();
    let got_sum_burn = sum_burn.into_data().into_vec::<f32>().unwrap()[0];
    let mean_burn = burn.mean();
    let got_mean_burn = mean_burn.into_data().into_vec::<f32>().unwrap()[0];

    assert!((got_sum_coeus - got_sum_burn).abs() < 1e-5);
    assert!((got_mean_coeus - got_mean_burn).abs() < 1e-5);
}
