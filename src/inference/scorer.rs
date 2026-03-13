//! Run saved models over a dataset and return per-sample reconstruction errors.

use burn::prelude::*;

use crate::datasets::Sample;
use crate::models::{DeepAE, ShallowAE};
use crate::training::metrics::reconstruction_errors;
use crate::training::trainer::samples_to_tensor;

pub fn score_shallow<B: Backend>(
    model: ShallowAE<B>,
    samples: &[Sample],
    batch_size: usize,
    device: &B::Device,
) -> Vec<f32> {
    score_with(samples, batch_size, device, |batch| model.forward(batch))
}

pub fn score_deep<B: Backend>(
    model: DeepAE<B>,
    samples: &[Sample],
    batch_size: usize,
    device: &B::Device,
) -> Vec<f32> {
    score_with(samples, batch_size, device, |batch| model.forward(batch))
}

/// Generic batched scoring with any forward function.
fn score_with<B: Backend, F>(
    samples: &[Sample],
    batch_size: usize,
    device: &B::Device,
    forward: F,
) -> Vec<f32>
where
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let mut all_errors: Vec<f32> = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(batch_size) {
        let batch = samples_to_tensor::<B>(chunk, device);
        let output = forward(batch.clone());
        let errors = reconstruction_errors(output, batch);
        all_errors.extend(errors);
    }

    all_errors
}

/// Collect latent vectors for an entire split (used by EDA).
pub fn encode_shallow<B: Backend>(
    model: &ShallowAE<B>,
    samples: &[Sample],
    batch_size: usize,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    encode_with(samples, batch_size, device, |batch| model.encode(batch))
}

pub fn encode_deep<B: Backend>(
    model: &DeepAE<B>,
    samples: &[Sample],
    batch_size: usize,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    encode_with(samples, batch_size, device, |batch| model.encode(batch))
}

fn encode_with<B: Backend, F>(
    samples: &[Sample],
    batch_size: usize,
    device: &B::Device,
    forward: F,
) -> Vec<Vec<f32>>
where
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let latent_dim = {
        // Probe with one sample to learn latent dim
        let probe = samples_to_tensor::<B>(&samples[..1.min(samples.len())], device);
        let out = forward(probe);
        out.dims()[1]
    };

    let mut result: Vec<Vec<f32>> = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(batch_size) {
        let batch = samples_to_tensor::<B>(chunk, device);
        let z = forward(batch);
        let flat: Vec<f32> = z
            .into_data()
            .to_vec::<f32>()
            .expect("latent tensor to vec");
        for row in flat.chunks(latent_dim) {
            result.push(row.to_vec());
        }
    }

    result
}
