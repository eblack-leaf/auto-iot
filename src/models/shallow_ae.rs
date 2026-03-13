//! Shallow autoencoder: one hidden layer in both encoder and decoder.
//!
//! Architecture (for input_dim D, hidden_dim H, latent_dim L):
//!   Encoder: D → H (ReLU) → L
//!   Decoder: L → H (ReLU) → D (Sigmoid)
//!
//! Sigmoid on the output keeps reconstructions in [0,1], matching MinMax-normalised inputs.

use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct ShallowAE<B: Backend> {
    enc1: Linear<B>,
    enc2: Linear<B>,
    dec1: Linear<B>,
    dec2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ShallowAEConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
}

impl ShallowAEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ShallowAE<B> {
        ShallowAE {
            enc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            enc2: LinearConfig::new(self.hidden_dim, self.latent_dim).init(device),
            dec1: LinearConfig::new(self.latent_dim, self.hidden_dim).init(device),
            dec2: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> ShallowAE<B> {
    /// Encode a batch to the latent space.
    pub fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.activation.forward(self.enc1.forward(x));
        self.enc2.forward(h)
    }

    /// Decode from latent space back to input space.
    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.activation.forward(self.dec1.forward(z));
        // Sigmoid to constrain output to [0, 1]
        burn::tensor::activation::sigmoid(self.dec2.forward(h))
    }

    /// Full forward pass: encode then decode.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decode(self.encode(x))
    }
}
