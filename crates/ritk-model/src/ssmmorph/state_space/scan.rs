//! Differentiable selective associative scan.

use coeus_autograd::{add, cat, exp, mul, reshape, slice, sum_axis, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

pub(crate) fn selective_scan<B>(
    state_dim: usize,
    a_log: &Var<f32, B>,
    x: &Var<f32, B>,
    dt: &Var<f32, B>,
    input_matrix: &Var<f32, B>,
    output_matrix: &Var<f32, B>,
) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let shape = x.tensor.shape();
    let (batch, sequence, inner) = (shape[0], shape[1], shape[2]);
    let state = reshape(&coeus_autograd::neg(&exp(a_log)), [inner, state_dim]);
    let dt_expanded = reshape(dt, [batch, sequence, inner, 1]);
    let state_expanded = reshape(&state, [1, 1, inner, state_dim]);
    let transition = exp(&mul(&dt_expanded, &state_expanded));
    let input_expanded = reshape(input_matrix, [batch, sequence, 1, state_dim]);
    let discretized_input = mul(&dt_expanded, &input_expanded);
    let signal = reshape(x, [batch, sequence, inner, 1]);
    let forcing = mul(&discretized_input, &signal);
    let hidden = parallel_scan(transition, forcing);
    let output_expanded = reshape(output_matrix, [batch, sequence, 1, state_dim]);
    reshape(
        &sum_axis(&mul(&hidden, &output_expanded), 3),
        [batch, sequence, inner],
    )
}

/// Hillis-Steele scan for `h[t] = transition[t] * h[t-1] + forcing[t]`.
fn parallel_scan<B>(mut transition: Var<f32, B>, mut forcing: Var<f32, B>) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let shape = transition.tensor.shape();
    let (batch, sequence, inner, state) = (shape[0], shape[1], shape[2], shape[3]);
    let mut offset = 1;
    while offset < sequence {
        let current_transition = slice(
            &transition,
            &[(0, batch), (offset, sequence), (0, inner), (0, state)],
        );
        let current_forcing = slice(
            &forcing,
            &[(0, batch), (offset, sequence), (0, inner), (0, state)],
        );
        let previous_transition = slice(
            &transition,
            &[(0, batch), (0, sequence - offset), (0, inner), (0, state)],
        );
        let previous_forcing = slice(
            &forcing,
            &[(0, batch), (0, sequence - offset), (0, inner), (0, state)],
        );
        let next_transition = mul(&current_transition, &previous_transition);
        let next_forcing = add(
            &mul(&current_transition, &previous_forcing),
            &current_forcing,
        );
        let transition_prefix = slice(
            &transition,
            &[(0, batch), (0, offset), (0, inner), (0, state)],
        );
        let forcing_prefix = slice(&forcing, &[(0, batch), (0, offset), (0, inner), (0, state)]);
        transition = cat(&[&transition_prefix, &next_transition], 1);
        forcing = cat(&[&forcing_prefix, &next_forcing], 1);
        offset *= 2;
    }
    forcing
}
