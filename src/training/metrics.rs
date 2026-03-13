use burn::prelude::*;

/// Mean squared error between `output` and `target` tensors of shape [batch, dim].
pub fn mse_loss<B: Backend>(output: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    (output - target).powi_scalar(2).mean()
}

/// Per-sample MSE reconstruction error: returns a Vec<f32> of length `batch`.
pub fn reconstruction_errors<B: Backend>(
    output: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Vec<f32> {
    // [batch, dim] → [batch] mean over dim=1
    let sq = (output - target).powi_scalar(2);
    // mean_dim(1) returns [batch, 1]; squeeze to [batch]
    let per_sample: Tensor<B, 1> = sq.mean_dim(1).squeeze::<1>();
    per_sample
        .into_data()
        .to_vec::<f32>()
        .expect("tensor to vec")
}
