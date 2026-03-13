//! Core training loop used by both single-run and grid-search modes.

use std::fs;
use std::time::Instant;

use anyhow::Result;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::TrainArgs;
use crate::config::{HyperPoint, TrainConfig, TrainResult};
use crate::datasets::{self, AnomalyDataset, Sample};
use crate::models::{
    count_parameters, DeepAE, DeepAEConfig, ShallowAE, ShallowAEConfig,
};
use crate::training::early_stopping::{EarlyStopping, StopResult};
use crate::training::metrics::mse_loss;

// ── Public dispatch ───────────────────────────────────────────────────────────

/// Single-run training invoked from the CLI `train` subcommand.
pub fn run_train<B: AutodiffBackend>(args: &TrainArgs, device: &B::Device) -> Result<TrainResult> {
    let dataset = datasets::load(&args.dataset, &args.data_dir, args.window, args.val_split)?;

    let cfg = TrainConfig {
        dataset: args.dataset.clone(),
        arch: args.arch.clone(),
        epochs: args.epochs,
        latent_dim: args.latent_dim,
        hidden_dim: args.hidden_dim,
        lr: args.lr,
        batch_size: args.batch_size,
        patience: args.patience,
        artifact_dir: args.artifact_dir.clone(),
        data_dir: args.data_dir.clone(),
        window: args.window,
        clean_train: args.clean_train,
        val_split: args.val_split,
    };

    let hyper = HyperPoint {
        arch: args.arch.clone(),
        lr: args.lr,
        latent_dim: args.latent_dim,
        hidden_dim: args.hidden_dim,
    };

    match args.arch.as_str() {
        "shallow" => train_shallow::<B>(cfg, hyper, dataset.as_ref(), device),
        "deep" => train_deep::<B>(cfg, hyper, dataset.as_ref(), device),
        other => anyhow::bail!("Unknown architecture '{}'. Use: shallow, deep", other),
    }
}

// ── Architecture-specific entry points (called by grid search too) ────────────

pub fn train_shallow<B: AutodiffBackend>(
    cfg: TrainConfig,
    hyper: HyperPoint,
    dataset: &dyn AnomalyDataset,
    device: &B::Device,
) -> Result<TrainResult> {
    let model_cfg = ShallowAEConfig {
        input_dim: dataset.seq_len(),
        hidden_dim: hyper.hidden_dim,
        latent_dim: hyper.latent_dim,
    };
    let model = model_cfg.init::<B>(device);
    let params = count_parameters(&model);
    train_loop(cfg, hyper, model, params, dataset, device)
}

pub fn train_deep<B: AutodiffBackend>(
    cfg: TrainConfig,
    hyper: HyperPoint,
    dataset: &dyn AnomalyDataset,
    device: &B::Device,
) -> Result<TrainResult> {
    let model_cfg = DeepAEConfig {
        input_dim: dataset.seq_len(),
        hidden_dim: hyper.hidden_dim,
        latent_dim: hyper.latent_dim,
    };
    let model = model_cfg.init::<B>(device);
    let params = count_parameters(&model);
    train_loop(cfg, hyper, model, params, dataset, device)
}

// ── Generic training loop ─────────────────────────────────────────────────────

fn train_loop<B, M>(
    cfg: TrainConfig,
    hyper: HyperPoint,
    mut model: M,
    param_count: usize,
    dataset: &dyn AnomalyDataset,
    device: &B::Device,
) -> Result<TrainResult>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + Clone + std::fmt::Debug + Send,
    M::InnerModule: AeForward<B::InnerBackend>,
    M: AeForward<B>,
{
    fs::create_dir_all(&cfg.artifact_dir)?;

    let mut optimizer = AdamConfig::new().init::<B, M>();
    let mut stopper = EarlyStopping::new(cfg.patience);
    let artifact_path = artifact_path(&cfg, &hyper);

    use crate::models::metrics::{param_bytes, print_param_summary};
    println!(
        "\n┌─ {} │ arch={} │ lr={:.0e} │ latent={} │ hidden={}",
        dataset.name(),
        hyper.arch,
        hyper.lr,
        hyper.latent_dim,
        hyper.hidden_dim,
    );
    print_param_summary(&hyper.arch, param_count);
    println!(
        "  RAM footprint (f32): {:.1} KB",
        param_bytes(param_count) as f64 / 1024.0
    );

    let train_samples_owned: Vec<_>;
    let train_data: &[_] = if cfg.clean_train {
        train_samples_owned = dataset
            .train_samples()
            .iter()
            .filter(|s| s.label == Some(0))
            .cloned()
            .collect();
        println!(
            "  clean_train: {}/{} samples kept (label=0 only)",
            train_samples_owned.len(),
            dataset.train_samples().len()
        );
        &train_samples_owned
    } else {
        dataset.train_samples()
    };
    // Early stopping measures reconstruction on normal samples only.
    // If val contains anomalies their rising reconstruction error would
    // trigger early stopping immediately as the model specialises on normals.
    // Early stopping measures reconstruction on normal samples only.
    // If val contains anomalies their rising reconstruction error would
    // trigger early stopping immediately as the model specialises on normals.
    let all_val = dataset.val_samples();
    let val_normal_owned: Vec<_> = if all_val.iter().any(|s| s.label.is_some()) {
        all_val.iter().filter(|s| s.label == Some(0)).cloned().collect()
    } else {
        all_val.to_vec()
    };
    let val_data: &[_] = &val_normal_owned;

    let epoch_pb = ProgressBar::new(cfg.epochs as u64);
    epoch_pb.set_style(
        ProgressStyle::default_bar()
            .template("  epoch {pos:>3}/{len}  [{bar:40.cyan/blue}]  train={msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let t_start = Instant::now();
    let mut total_epochs = 0;

    for epoch in 0..cfg.epochs {
        total_epochs = epoch + 1;

        // ── train ──────────────────────────────────────────────────────────
        let train_loss = run_epoch::<B, M>(
            &mut model,
            Some(&mut optimizer),
            train_data,
            cfg.batch_size,
            hyper.lr,
            device,
        );

        // ── validate ───────────────────────────────────────────────────────
        let val_loss = {
            let inner = model.valid();
            run_epoch_inference::<B::InnerBackend, M::InnerModule>(
                &inner,
                val_data,
                cfg.batch_size,
                device.clone().into(),
            )
        };

        epoch_pb.set_message(format!("{:.6}  val={:.6}", train_loss, val_loss));
        epoch_pb.inc(1);

        match stopper.update(epoch, val_loss) {
            StopResult::Improved => {
                model
                    .clone()
                    .save_file(&artifact_path, &CompactRecorder::new())
                    .map_err(|e| anyhow::anyhow!("Save failed: {:?}", e))?;
            }
            StopResult::Continue => {}
            StopResult::Stop => {
                epoch_pb.finish_with_message(format!(
                    "early stop at epoch {} (patience={})",
                    epoch, cfg.patience
                ));
                break;
            }
        }
    }

    if total_epochs == cfg.epochs {
        epoch_pb.finish();
    }

    let train_secs = t_start.elapsed().as_secs_f64();

    // ── Post-training AUROC on test set ────────────────────────────────────
    let auroc = if dataset.has_labels() {
        let test_data = dataset.test_samples();
        let inner = model.valid();
        let errors = {
            let mut all: Vec<f32> = Vec::with_capacity(test_data.len());
            for chunk in test_data.chunks(cfg.batch_size) {
                let batch = samples_to_tensor::<B::InnerBackend>(chunk, &device.clone().into());
                let out = inner.ae_forward(batch.clone());
                let errs = crate::training::metrics::reconstruction_errors(out, batch);
                all.extend(errs);
            }
            all
        };
        let labels: Vec<u8> = test_data.iter().map(|s| s.label.unwrap_or(0)).collect();
        crate::inference::evaluator::compute_auroc(&errors, &labels)
    } else {
        None
    };

    let result = TrainResult {
        hyper,
        dataset: cfg.dataset.clone(),
        best_val_loss: stopper.best_loss(),
        best_epoch: stopper.best_epoch,
        total_epochs,
        param_count,
        train_secs,
        artifact_path: artifact_path.clone(),
        auroc,
    };

    println!(
        "└─ done  best_val={:.6}  auroc={}  epoch={}  time={:.1}s  → {}",
        result.best_val_loss,
        result.auroc.map(|a| format!("{:.4}", a)).unwrap_or("n/a".into()),
        result.best_epoch,
        result.train_secs,
        artifact_path
    );

    Ok(result)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Trait that both ShallowAE and DeepAE implement, allowing generic dispatch.
pub trait AeForward<B: Backend> {
    fn ae_forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}

impl<B: Backend> AeForward<B> for ShallowAE<B> {
    fn ae_forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: Backend> AeForward<B> for DeepAE<B> {
    fn ae_forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

fn run_epoch<B, M>(
    model: &mut M,
    optimizer: Option<&mut impl Optimizer<M, B>>,
    samples: &[Sample],
    batch_size: usize,
    lr: f64,
    device: &B::Device,
) -> f64
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + Clone + AeForward<B>,
{
    let optimizer = optimizer.expect("optimizer required for training");
    let mut total_loss = 0.0_f64;
    let mut n_batches = 0;

    for chunk in samples.chunks(batch_size) {
        let batch = samples_to_tensor::<B>(chunk, device);
        let output = model.ae_forward(batch.clone());
        let loss = mse_loss(output, batch);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optimizer.step(lr, model.clone(), grads);

        let loss_val: f32 = loss
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0);
        total_loss += loss_val as f64;
        n_batches += 1;
    }

    if n_batches == 0 {
        0.0
    } else {
        total_loss / n_batches as f64
    }
}

fn run_epoch_inference<B, M>(
    model: &M,
    samples: &[Sample],
    batch_size: usize,
    device: B::Device,
) -> f64
where
    B: Backend,
    M: AeForward<B>,
{
    let mut total_loss = 0.0_f64;
    let mut n_batches = 0;

    for chunk in samples.chunks(batch_size) {
        let batch = samples_to_tensor::<B>(chunk, &device);
        let output = model.ae_forward(batch.clone());
        let loss = mse_loss(output, batch);

        let loss_val: f32 = loss
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0);
        total_loss += loss_val as f64;
        n_batches += 1;
    }

    if n_batches == 0 {
        0.0
    } else {
        total_loss / n_batches as f64
    }
}

pub fn samples_to_tensor<B: Backend>(samples: &[Sample], device: &B::Device) -> Tensor<B, 2> {
    let dim = samples[0].features.len();
    let flat: Vec<f32> = samples.iter().flat_map(|s| s.features.iter().copied()).collect();
    Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(flat, [samples.len(), dim]),
        device,
    )
}

fn artifact_path(cfg: &TrainConfig, hyper: &HyperPoint) -> String {
    let lr_str = format!("{:.0e}", hyper.lr).replace("e-0", "e-").replace("e+0", "e");
    format!(
        "{}/{}_{}_ld{}_hd{}_lr{}",
        cfg.artifact_dir,
        cfg.dataset.replace('-', "_"),
        hyper.arch,
        hyper.latent_dim,
        hyper.hidden_dim,
        lr_str,
    )
}
