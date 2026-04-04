use burn::tensor::{backend::Backend, Tensor};

/// Selective scan operation using Parallel Associative Scan
pub(crate) fn selective_scan<B: Backend>(
    state_dim: usize,
    a_log_val: Tensor<B, 1>,
    x: Tensor<B, 3>,  // [batch, seq, inner]
    dt: Tensor<B, 3>, // [batch, seq, inner]
    b: Tensor<B, 3>,  // [batch, seq, state]
    c: Tensor<B, 3>,  // [batch, seq, state]
) -> Tensor<B, 3> {
    let [_batch, _seq_len, inner_dim] = x.dims();

    // Get A from log parameter (A = -exp(parameter))
    let a = a_log_val.exp().neg().reshape([inner_dim, state_dim]);

    // Discretize: Ā = exp(ΔA)
    // dt: [batch, seq, inner] -> [batch, seq, inner, 1]
    let dt_expanded: Tensor<B, 4> = dt.unsqueeze_dim::<4>(3);
    // a: [inner, state] -> [1, 1, inner, state]
    let a_expanded: Tensor<B, 4> = a.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

    // Ā = exp(Δ * A) -> [batch, seq, inner, state]
    let a_bar = (dt_expanded.clone() * a_expanded).exp();

    // B̄ = Δ * B
    // b: [batch, seq, state] -> [batch, seq, 1, state]
    let b_expanded: Tensor<B, 4> = b.unsqueeze_dim::<4>(2);
    let b_bar = dt_expanded * b_expanded; // [batch, seq, inner, state]

    // U = B̄ * x
    // x: [batch, seq, inner] -> [batch, seq, inner, 1]
    let x_expanded: Tensor<B, 4> = x.unsqueeze_dim::<4>(3);
    // u = b_bar * x_expanded -> [batch, seq, inner, state]
    // Broadcasting: [batch, seq, inner, state] * [batch, seq, inner, 1] works
    let u = b_bar * x_expanded;

    // Perform Parallel Associative Scan
    // h_t = Ā_t * h_{t-1} + U_t
    let h = parallel_scan(a_bar, u);

    // Output: y = (C * h).sum(-1)
    // c: [batch, seq, state] -> [batch, seq, 1, state]
    let c_expanded: Tensor<B, 4> = c.unsqueeze_dim(2);
    // h: [batch, seq, inner, state]
    // y: [batch, seq, inner]
    let y = (h * c_expanded).sum_dim(3);
    let [batch_size, seq_length, inner, _] = y.dims();
    y.reshape([batch_size, seq_length, inner])
}

/// Parallel Associative Scan (Hillis-Steele)
/// Computes prefix scan for: h_t = a_t * h_{t-1} + u_t
pub(crate) fn parallel_scan<B: Backend>(mut a: Tensor<B, 4>, mut u: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, seq_len, inner, state] = a.dims();

    // Iterate log2(N) times
    let steps = (seq_len as f64).log2().ceil() as usize;

    for i in 0..steps {
        let k = 1 << i;
        if k >= seq_len {
            break;
        }

        // Slice views
        let a_curr = a.clone().slice([0..batch, k..seq_len, 0..inner, 0..state]);
        let u_curr = u.clone().slice([0..batch, k..seq_len, 0..inner, 0..state]);

        let a_prev = a
            .clone()
            .slice([0..batch, 0..seq_len - k, 0..inner, 0..state]);
        let u_prev = u
            .clone()
            .slice([0..batch, 0..seq_len - k, 0..inner, 0..state]);

        // Compute updates
        let a_new = a_curr.clone() * a_prev.clone();
        let u_new = a_curr * u_prev + u_curr;

        // Concatenate with unchanged prefix
        let a_prefix = a.clone().slice([0..batch, 0..k, 0..inner, 0..state]);
        let u_prefix = u.clone().slice([0..batch, 0..k, 0..inner, 0..state]);

        a = Tensor::cat(vec![a_prefix, a_new], 1);
        u = Tensor::cat(vec![u_prefix, u_new], 1);
    }

    u
}
