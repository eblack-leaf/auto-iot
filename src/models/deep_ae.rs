//! Deep autoencoder: two hidden layers in encoder and decoder.
//!
//! Architecture (input_dim D, hidden_dim H, latent_dim L):
//!   Encoder: D → H (ReLU) → H/2 (ReLU) → L
//!   Decoder: L → H/2 (ReLU) → H (ReLU) → D (Sigmoid)
//!
//! Roughly 3× more parameters than ShallowAE at the same hidden_dim.

use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct DeepAE<B: Backend> {
    enc1: Linear<B>,
    enc2: Linear<B>,
    enc3: Linear<B>,
    dec1: Linear<B>,
    dec2: Linear<B>,
    dec3: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct DeepAEConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,  // first hidden layer; second is hidden_dim / 2
    pub latent_dim: usize,
}

impl DeepAEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeepAE<B> {
        let h2 = (self.hidden_dim / 2).max(self.latent_dim + 1);
        DeepAE {
            enc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            enc2: LinearConfig::new(self.hidden_dim, h2).init(device),
            enc3: LinearConfig::new(h2, self.latent_dim).init(device),
            dec1: LinearConfig::new(self.latent_dim, h2).init(device),
            dec2: LinearConfig::new(h2, self.hidden_dim).init(device),
            dec3: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
            activation: Relu::new(),
        }
    }

    pub fn h2(&self) -> usize {
        (self.hidden_dim / 2).max(self.latent_dim + 1)
    }
}

impl<B: Backend> DeepAE<B> {
    pub fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h1 = self.activation.forward(self.enc1.forward(x));
        let h2 = self.activation.forward(self.enc2.forward(h1));
        self.enc3.forward(h2)
    }

    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let h1 = self.activation.forward(self.dec1.forward(z));
        let h2 = self.activation.forward(self.dec2.forward(h1));
        burn::tensor::activation::sigmoid(self.dec3.forward(h2))
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decode(self.encode(x))
    }
}
