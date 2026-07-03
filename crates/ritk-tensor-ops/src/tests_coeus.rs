#![cfg(feature = "coeus")]

use crate::coeus as coeus_tensor_ops;
use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor as CoeusTensor;
use ritk_image::native::Image as CoeusImage;
use ritk_spatial::{Direction, Point, Spacing};

type BurnB = NdArray<f32>;
type Shape2 = [usize; 2];

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

struct BinaryCase {
    op: BinaryOp,
    lhs: &'static [f32],
    rhs: &'static [f32],
    expected: &'static [f32],
}

const SHAPE: Shape2 = [2, 3];

const BINARY_CASES: &[BinaryCase] = &[
    BinaryCase {
        op: BinaryOp::Add,
        lhs: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rhs: &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        expected: &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
    },
    BinaryCase {
        op: BinaryOp::Sub,
        lhs: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rhs: &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        expected: &[-9.0, -18.0, -27.0, -36.0, -45.0, -54.0],
    },
    BinaryCase {
        op: BinaryOp::Mul,
        lhs: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rhs: &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        expected: &[10.0, 40.0, 90.0, 160.0, 250.0, 360.0],
    },
    BinaryCase {
        op: BinaryOp::Div,
        lhs: &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        rhs: &[2.0, 4.0, 5.0, 8.0, 10.0, 12.0],
        expected: &[5.0, 5.0, 6.0, 5.0, 5.0, 5.0],
    },
];

fn coeus_tensor(values: &[f32]) -> CoeusTensor<f32, MoiraiBackend> {
    CoeusTensor::<f32, _>::from_slice_on(SHAPE, values, &MoiraiBackend)
}

fn coeus_image(values: &[f32]) -> CoeusImage<f32, MoiraiBackend, 2> {
    CoeusImage::new(
        coeus_tensor(values),
        Point::new([10.0, 20.0]),
        Spacing::new([0.5, 1.5]),
        Direction::identity(),
    )
    .unwrap()
}

fn burn_tensor(values: &[f32]) -> BurnTensor<BurnB, 2> {
    let device = Default::default();
    BurnTensor::<BurnB, 2>::from_data(
        TensorData::new(values.to_vec(), burn::tensor::Shape::new(SHAPE)),
        &device,
    )
}

fn coeus_data(tensor: &CoeusTensor<f32, MoiraiBackend>) -> Vec<f32> {
    tensor.as_slice().to_vec()
}

fn burn_data<const D: usize>(tensor: BurnTensor<BurnB, D>) -> Vec<f32> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .expect("invariant: Burn differential tensor stores f32 values")
}

#[test]
fn differential_elementwise_binary_ops() {
    for case in BINARY_CASES {
        let backend = MoiraiBackend;
        let lhs_coeus = coeus_tensor(case.lhs);
        let rhs_coeus = coeus_tensor(case.rhs);
        let got_coeus = match case.op {
            BinaryOp::Add => coeus_data(&coeus_ops::add(&lhs_coeus, &rhs_coeus, &backend)),
            BinaryOp::Sub => coeus_data(&coeus_ops::sub(&lhs_coeus, &rhs_coeus, &backend)),
            BinaryOp::Mul => coeus_data(&coeus_ops::mul(&lhs_coeus, &rhs_coeus, &backend)),
            BinaryOp::Div => coeus_data(&coeus_ops::div(&lhs_coeus, &rhs_coeus, &backend)),
        };

        let lhs_burn = burn_tensor(case.lhs);
        let rhs_burn = burn_tensor(case.rhs);
        let got_burn = match case.op {
            BinaryOp::Add => burn_data(lhs_burn.add(rhs_burn)),
            BinaryOp::Sub => burn_data(lhs_burn.sub(rhs_burn)),
            BinaryOp::Mul => burn_data(lhs_burn.mul(rhs_burn)),
            BinaryOp::Div => burn_data(lhs_burn.div(rhs_burn)),
        };

        assert_eq!(got_coeus, case.expected);
        assert_eq!(got_coeus, got_burn);
    }
}

#[test]
fn differential_shape_ops_preserve_values() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let coeus = coeus_tensor(&values);
    let reshaped_coeus = coeus.clone().reshape([3, 2]);
    assert_eq!(reshaped_coeus.shape(), &[3, 2]);
    assert_eq!(coeus_data(&reshaped_coeus), values);

    let transposed_coeus = coeus.t();
    assert_eq!(transposed_coeus.shape(), &[3, 2]);
    assert_eq!(transposed_coeus.get(&[0, 0]), 1.0);
    assert_eq!(transposed_coeus.get(&[0, 1]), 4.0);
    assert_eq!(transposed_coeus.get(&[1, 0]), 2.0);
    assert_eq!(transposed_coeus.get(&[1, 1]), 5.0);
    assert_eq!(transposed_coeus.get(&[2, 0]), 3.0);
    assert_eq!(transposed_coeus.get(&[2, 1]), 6.0);

    let burn = burn_tensor(&values);
    let reshaped_burn = burn.clone().reshape([3, 2]);
    assert_eq!(reshaped_burn.shape().dims, [3, 2]);
    assert_eq!(burn_data(reshaped_burn), values);

    let transposed_burn = burn.transpose();
    assert_eq!(transposed_burn.shape().dims, [3, 2]);
    assert_eq!(
        burn_data(transposed_burn),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn differential_reductions() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let backend = MoiraiBackend;
    let coeus = coeus_tensor(&values);
    let got_sum_coeus = coeus_ops::sum(&coeus, &backend);
    let got_mean_coeus = coeus_ops::mean(&coeus, &backend);

    let burn = burn_tensor(&values);
    let got_sum_burn = burn_data(burn.clone().sum())[0];
    let got_mean_burn = burn_data(burn.mean())[0];

    assert_eq!(got_sum_coeus, 21.0);
    assert_eq!(got_mean_coeus, 3.5);
    assert_eq!(got_sum_coeus, got_sum_burn);
    assert_eq!(got_mean_coeus, got_mean_burn);
}

#[test]
fn coeus_extract_slice_borrows_contiguous_tensor() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = coeus_tensor(&values);

    let (slice, dims) = coeus_tensor_ops::extract_slice::<f32, MoiraiBackend, 2>(&tensor).unwrap();

    assert_eq!(dims, SHAPE);
    assert_eq!(slice, values.as_slice());
}

#[test]
fn coeus_extract_vec_matches_slice_values() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = coeus_tensor(&values);

    let (owned, dims) = coeus_tensor_ops::extract_vec::<f32, MoiraiBackend, 2>(&tensor).unwrap();

    assert_eq!(dims, SHAPE);
    assert_eq!(owned, values);
}

#[test]
fn coeus_extract_slice_rejects_non_contiguous_view() {
    let tensor = coeus_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let transposed = tensor.t();

    let err = coeus_tensor_ops::extract_slice::<f32, MoiraiBackend, 2>(&transposed)
        .expect_err("transposed view must not be borrowed as contiguous");

    assert_eq!(
        err.to_string(),
        "coeus tensor ops: extract_slice requires contiguous layout, got shape=[3, 2] strides=[1, 3]"
    );
}

#[test]
fn coeus_extract_slice_rejects_rank_mismatch() {
    let tensor = coeus_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let err = coeus_tensor_ops::extract_slice::<f32, MoiraiBackend, 3>(&tensor)
        .expect_err("rank mismatch must be reported");

    assert_eq!(
        err.to_string(),
        "coeus tensor ops: expected rank 3, got rank 2 shape=[2, 3]"
    );
}

#[test]
fn coeus_rebuild_validates_shape_product() {
    let backend = MoiraiBackend;
    let err = match coeus_tensor_ops::rebuild::<f32, MoiraiBackend, 2>(
        vec![1.0, 2.0, 3.0],
        SHAPE,
        &backend,
    ) {
        Ok(_) => panic!("shape/data mismatch must be reported"),
        Err(err) => err,
    };

    assert_eq!(
        err.to_string(),
        "coeus tensor ops: data length 3 does not match shape [2, 3] product 6"
    );
}

#[test]
fn coeus_rebuild_preserves_values_and_shape() {
    let backend = MoiraiBackend;
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let tensor =
        coeus_tensor_ops::rebuild::<f32, MoiraiBackend, 2>(values.clone(), SHAPE, &backend)
            .unwrap();

    assert_eq!(tensor.shape(), SHAPE);
    assert_eq!(tensor.as_slice(), values.as_slice());
}

#[test]
fn coeus_image_extract_slice_borrows_contiguous_image() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let image = coeus_image(&values);

    let (slice, dims) =
        coeus_tensor_ops::extract_image_slice::<f32, MoiraiBackend, 2>(&image).unwrap();

    assert_eq!(dims, SHAPE);
    assert_eq!(slice, values.as_slice());
}

#[test]
fn coeus_image_extract_vec_matches_slice_values() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let image = coeus_image(&values);

    let (owned, dims) =
        coeus_tensor_ops::extract_image_vec::<f32, MoiraiBackend, 2>(&image).unwrap();

    assert_eq!(dims, SHAPE);
    assert_eq!(owned, values);
}

#[test]
fn coeus_rebuild_image_preserves_values_shape_and_metadata() {
    let backend = MoiraiBackend;
    let source = coeus_image(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let values = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let rebuilt = coeus_tensor_ops::rebuild_image::<f32, MoiraiBackend, 2>(
        values.clone(),
        SHAPE,
        &source,
        &backend,
    )
    .unwrap();

    assert_eq!(rebuilt.shape(), SHAPE);
    assert_eq!(rebuilt.data_slice().unwrap(), values.as_slice());
    assert_eq!(rebuilt.origin(), source.origin());
    assert_eq!(rebuilt.spacing(), source.spacing());
    assert_eq!(rebuilt.direction(), source.direction());
}

#[test]
fn coeus_rebuild_image_validates_shape_product() {
    let backend = MoiraiBackend;
    let source = coeus_image(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let err = coeus_tensor_ops::rebuild_image::<f32, MoiraiBackend, 2>(
        vec![1.0, 2.0, 3.0],
        SHAPE,
        &source,
        &backend,
    )
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "coeus tensor ops: data length 3 does not match shape [2, 3] product 6"
    );
}
