pub mod ecg5000;
pub mod nab_taxi;
pub mod normalizer;
pub mod synthetic;

pub use ecg5000::Ecg5000;
pub use nab_taxi::NabTaxi;
pub use synthetic::Synthetic;

use anyhow::Result;

/// A single sample ready for the model.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Flattened feature vector (already normalised).
    pub features: Vec<f32>,
    /// 1 = anomaly, 0 = normal, None = unlabelled.
    pub label: Option<u8>,
}

/// Common interface implemented by every dataset.
pub trait AnomalyDataset: Send + Sync {
    fn name(&self) -> &str;
    fn seq_len(&self) -> usize;

    fn train_samples(&self) -> &[Sample];
    fn val_samples(&self) -> &[Sample];
    fn test_samples(&self) -> &[Sample];

    /// True if the test split carries ground-truth labels.
    fn has_labels(&self) -> bool {
        self.test_samples().iter().any(|s| s.label.is_some())
    }
}

/// Download a dataset by name into `output_dir`.
pub fn fetch(dataset: &str, output_dir: &str) -> Result<()> {
    match dataset {
        "ecg5000" => ecg5000::fetch(output_dir),
        "nab-taxi" => nab_taxi::fetch(output_dir),
        "all" => {
            ecg5000::fetch(output_dir)?;
            nab_taxi::fetch(output_dir)?;
            Ok(())
        }
        other => anyhow::bail!(
            "Unknown dataset '{}'. Valid fetch targets: ecg5000, nab-taxi, all",
            other
        ),
    }
}

/// Load a dataset (already downloaded) and return a boxed trait object.
pub fn load(name: &str, data_dir: &str, window: usize) -> Result<Box<dyn AnomalyDataset>> {
    match name {
        "ecg5000" => Ok(Box::new(Ecg5000::load(data_dir)?)),
        "nab-taxi" => Ok(Box::new(NabTaxi::load(data_dir, window)?)),
        "synthetic" => Ok(Box::new(Synthetic::generate(window, 3000, 0.05))),
        other => anyhow::bail!(
            "Unknown dataset '{}'. Valid values: ecg5000, nab-taxi, synthetic",
            other
        ),
    }
}
