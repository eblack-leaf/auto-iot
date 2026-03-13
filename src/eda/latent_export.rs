//! Encode a dataset split through a saved model and write the latent vectors to CSV.
//!
//! Output CSV columns: z0, z1, …, z{L-1}, label
//! This file is also the input consumed by tsne.rs.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use burn::prelude::*;
use burn::record::CompactRecorder;

use crate::cli::EdaArgs;
use crate::datasets::AnomalyDataset;
use crate::inference::scorer::{encode_deep, encode_shallow};
use crate::models::{DeepAEConfig, ShallowAEConfig};

// We need a concrete backend for inference-only use.
// The EDA command uses the non-autodiff (inference) backend.
use burn::backend::NdArray;
type B = NdArray;

pub fn export(args: &EdaArgs, dataset: &dyn AnomalyDataset, model_path: &str) -> Result<()> {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let test_samples = dataset.test_samples();
    let batch_size = 256;

    let latent_vecs: Vec<Vec<f32>> = match args.arch.as_str() {
        "shallow" => {
            let cfg = ShallowAEConfig {
                input_dim: dataset.seq_len(),
                hidden_dim: args.hidden_dim,
                latent_dim: args.latent_dim,
            };
            let model = cfg
                .init::<B>(&device)
                .load_file(model_path, &CompactRecorder::new(), &device)
                .map_err(|e| anyhow::anyhow!("Load failed: {:?}", e))?;
            encode_shallow::<B>(&model, test_samples, batch_size, &device)
        }
        "deep" => {
            let cfg = DeepAEConfig {
                input_dim: dataset.seq_len(),
                hidden_dim: args.hidden_dim,
                latent_dim: args.latent_dim,
            };
            let model = cfg
                .init::<B>(&device)
                .load_file(model_path, &CompactRecorder::new(), &device)
                .map_err(|e| anyhow::anyhow!("Load failed: {:?}", e))?;
            encode_deep::<B>(&model, test_samples, batch_size, &device)
        }
        other => anyhow::bail!("Unknown arch '{}'. Use: shallow, deep", other),
    };

    // Write CSV
    let out_path = PathBuf::from(&args.output_dir).join("latent_vectors.csv");
    let mut wtr = csv::Writer::from_path(&out_path)?;

    // Header
    let latent_dim = latent_vecs.first().map(|v| v.len()).unwrap_or(args.latent_dim);
    let mut header: Vec<String> = (0..latent_dim).map(|i| format!("z{}", i)).collect();
    header.push("label".into());
    wtr.write_record(&header)?;

    for (vec, sample) in latent_vecs.iter().zip(test_samples.iter()) {
        let mut row: Vec<String> = vec.iter().map(|v| format!("{:.6}", v)).collect();
        row.push(sample.label.map(|l| l.to_string()).unwrap_or_else(|| "-1".into()));
        wtr.write_record(&row)?;
    }
    wtr.flush()?;

    println!("  Latent vectors ({} × {}) → {}", latent_vecs.len(), latent_dim, out_path.display());
    Ok(())
}
