use serde::{Deserialize, Serialize};

/// A single point in the hyperparameter grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperPoint {
    pub arch: String,
    pub lr: f64,
    pub latent_dim: usize,
    pub hidden_dim: usize,
}

/// Configuration for one training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub dataset: String,
    pub arch: String,
    pub epochs: usize,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub patience: usize,
    pub artifact_dir: String,
    pub data_dir: String,
    pub window: usize,
    pub clean_train: bool,
}

/// Configuration for the hyperparameter grid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub learning_rates: Vec<f64>,
    pub latent_dims: Vec<usize>,
    pub hidden_dims: Vec<usize>,
    pub architectures: Vec<String>,
    pub epochs: usize,
    pub batch_size: usize,
    pub patience: usize,
}

impl Default for GridConfig {
    fn default() -> Self {
        GridConfig {
            learning_rates: vec![1e-3, 5e-4, 1e-4],
            latent_dims: vec![4, 8, 16],
            hidden_dims: vec![32, 64, 128],
            architectures: vec!["shallow".into(), "deep".into()],
            epochs: 100,
            batch_size: 64,
            patience: 3,
        }
    }
}

impl GridConfig {
    /// Expand to the Cartesian product of all hyperparameter axes.
    pub fn grid_points(&self) -> Vec<HyperPoint> {
        let mut points = Vec::new();
        for arch in &self.architectures {
            for &lr in &self.learning_rates {
                for &latent_dim in &self.latent_dims {
                    for &hidden_dim in &self.hidden_dims {
                        points.push(HyperPoint {
                            arch: arch.clone(),
                            lr,
                            latent_dim,
                            hidden_dim,
                        });
                    }
                }
            }
        }
        points
    }
}

/// Result recorded for each training run (single or grid).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainResult {
    pub hyper: HyperPoint,
    pub dataset: String,
    pub best_val_loss: f64,
    pub best_epoch: usize,
    pub total_epochs: usize,
    pub param_count: usize,
    pub train_secs: f64,
    pub artifact_path: String,
    /// AUROC on the test split — None if the dataset has no labels.
    pub auroc: Option<f64>,
}
