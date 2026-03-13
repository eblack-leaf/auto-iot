//! Synthetic Gaussian dataset — no download required.
//!
//! Normal samples: each feature ~ N(0, 1).
//! Anomalies: features shifted by +4σ on a random subset of dimensions,
//! injected at the requested anomaly rate.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use super::normalizer::MinMaxNormalizer;
use super::{AnomalyDataset, Sample};

pub struct Synthetic {
    train: Vec<Sample>,
    val: Vec<Sample>,
    test: Vec<Sample>,
    dim: usize,
}

impl Synthetic {
    /// Generate a synthetic dataset in memory.
    ///
    /// - `dim`: feature dimension (same role as sequence length for other datasets)
    /// - `n_total`: total number of samples
    /// - `anomaly_rate`: fraction of test samples that are anomalies (0.0–1.0)
    pub fn generate(dim: usize, n_total: usize, anomaly_rate: f64, val_split: f32) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();

        let split_val = (n_total as f32 * 0.85) as usize;
        let split_train = (split_val as f32 * (1.0 - val_split)) as usize;

        // Generate raw features
        let mut raw: Vec<Vec<f32>> = (0..n_total)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // Inject anomalies into test portion
        let n_test = n_total - split_val;
        let n_anom = (n_test as f64 * anomaly_rate).round() as usize;
        let anom_indices: Vec<usize> = rand::seq::index::sample(&mut rng, n_test, n_anom).into_vec();

        for &idx in &anom_indices {
            let global_idx = split_val + idx;
            let affected_dims = dim / 2;
            for d in 0..affected_dims {
                raw[global_idx][d] += 4.0;
            }
        }

        // Normalise with training statistics
        let norm = MinMaxNormalizer::fit(&raw[..split_train]);

        let to_samples = |slice: &[Vec<f32>], label: Option<u8>| -> Vec<Sample> {
            slice
                .iter()
                .map(|f| Sample {
                    features: norm.transform(f),
                    label,
                })
                .collect()
        };

        // Only the anomalously-injected test samples get label 1
        let test: Vec<Sample> = raw[split_val..]
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let is_anom = anom_indices.contains(&i);
                Sample {
                    features: norm.transform(f),
                    label: Some(if is_anom { 1 } else { 0 }),
                }
            })
            .collect();

        Synthetic {
            train: to_samples(&raw[..split_train], Some(0)),
            val: to_samples(&raw[split_train..split_val], Some(0)),
            test,
            dim,
        }
    }
}

impl AnomalyDataset for Synthetic {
    fn name(&self) -> &str {
        "synthetic"
    }
    fn seq_len(&self) -> usize {
        self.dim
    }
    fn train_samples(&self) -> &[Sample] {
        &self.train
    }
    fn val_samples(&self) -> &[Sample] {
        &self.val
    }
    fn test_samples(&self) -> &[Sample] {
        &self.test
    }
}
