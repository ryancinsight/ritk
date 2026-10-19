//! Differentiable selective associative scan.

use coeus_autograd::{add, cat, exp, mul, neg, reshape, slice, sum_axis, Var};

use crate::ModelError;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

pub(crate) fn selective_scan<B>(
    state_dim: usize,
    a_log: &Var<f32, B>,
    x: &Var<f32, B>,
    dt: &Var<f32, B>,
    input_matrix: &Var<f32, B>,
    output_matrix: &Var<f32, B>,
) -> Result<Var<f32, B>, ModelError>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let shape = x.tensor.shape();
    let (batch, sequence, inner) = (shape[0], shape[1], shape[2]);
    let decay = exp(a_log).map_err(ModelError::backend)?;
    let neg_decay = neg(&decay).map_err(ModelError::backend)?;
    let state = reshape(&neg_decay, [inner, state_dim]).map_err(ModelError::backend)?;
    let dt_expanded = reshape(dt, [batch, sequence, inner, 1]).map_err(ModelError::backend)?;
    let state_expanded = reshape(&state, [1, 1, inner, state_dim]).map_err(ModelError::backend)?;
    let dt_state = mul(&dt_expanded, &state_expanded).map_err(ModelError::backend)?;
    let transition = exp(&dt_state).map_err(ModelError::backend)?;
    let input_expanded =
        reshape(input_matrix, [batch, sequence, 1, state_dim]).map_err(ModelError::backend)?;
    let discretized_input = mul(&dt_expanded, &input_expanded).map_err(ModelError::backend)?;
    let signal = reshape(x, [batch, sequence, inner, 1]).map_err(ModelError::backend)?;
    let forcing = mul(&discretized_input, &signal).map_err(ModelError::backend)?;
    let hidden = parallel_scan(transition, forcing)?;
    let output_expanded =
        reshape(output_matrix, [batch, sequence, 1, state_dim]).map_err(ModelError::backend)?;
    let projected = mul(&hidden, &output_expanded).map_err(ModelError::backend)?;
    let reduced = sum_axis(&projected, 3).map_err(ModelError::backend)?;
    reshape(&reduced, [batch, sequence, inner]).map_err(ModelError::backend)
}

/// Hillis-Steele scan for `h[t] = transition[t] * h[t-1] + forcing[t]`.
fn parallel_scan<B>(
    mut transition: Var<f32, B>,
    mut forcing: Var<f32, B>,
) -> Result<Var<f32, B>, ModelError>
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
        )
        .map_err(ModelError::backend)?;
        let current_forcing = slice(
            &forcing,
            &[(0, batch), (offset, sequence), (0, inner), (0, state)],
        )
        .map_err(ModelError::backend)?;
        let previous_transition = slice(
            &transition,
            &[(0, batch), (0, sequence - offset), (0, inner), (0, state)],
        )
        .map_err(ModelError::backend)?;
        let previous_forcing = slice(
            &forcing,
            &[(0, batch), (0, sequence - offset), (0, inner), (0, state)],
        )
        .map_err(ModelError::backend)?;
        let next_transition =
            mul(&current_transition, &previous_transition).map_err(ModelError::backend)?;
        let carried = mul(&current_transition, &previous_forcing).map_err(ModelError::backend)?;
        let next_forcing = add(&carried, &current_forcing).map_err(ModelError::backend)?;
        let transition_prefix = slice(
            &transition,
            &[(0, batch), (0, offset), (0, inner), (0, state)],
        )
        .map_err(ModelError::backend)?;
        let forcing_prefix = slice(&forcing, &[(0, batch), (0, offset), (0, inner), (0, state)])
            .map_err(ModelError::backend)?;
        transition =
            cat(&[&transition_prefix, &next_transition], 1).map_err(ModelError::backend)?;
        forcing = cat(&[&forcing_prefix, &next_forcing], 1).map_err(ModelError::backend)?;
        offset *= 2;
    }
    Ok(forcing)
}
