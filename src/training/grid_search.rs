//! Hyperparameter grid search over arch × lr × latent_dim × hidden_dim.
//!
//! Results are ranked by best validation loss and written to a CSV file.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::TrainArgs;
use crate::config::{GridConfig, TrainConfig, TrainResult};
use crate::datasets;
use crate::training::trainer::{train_deep, train_shallow};

pub fn run_grid_search<B: AutodiffBackend>(
    args: &TrainArgs,
    device: &B::Device,
) -> Result<()> {
    let grid = GridConfig::default();
    let points = grid.grid_points();
    let n = points.len();

    println!(
        "\n══ Grid search: {} points ({} archs × {} lrs × {} latent × {} hidden) ══",
        n,
        grid.architectures.len(),
        grid.learning_rates.len(),
        grid.latent_dims.len(),
        grid.hidden_dims.len(),
    );

    // Lazy-load the dataset once — shared across all grid points.
    let dataset = datasets::load(&args.dataset, &args.data_dir, args.window)?;

    let overall_pb = ProgressBar::new(n as u64);
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template("\n[{elapsed_precise}] grid {pos}/{len} {bar:30.green/white} {msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let mut results: Vec<TrainResult> = Vec::with_capacity(n);

    for (idx, hyper) in points.into_iter().enumerate() {
        overall_pb.set_message(format!(
            "{} arch={} lr={:.0e} ld={} hd={}",
            args.dataset, hyper.arch, hyper.lr, hyper.latent_dim, hyper.hidden_dim
        ));

        let cfg = TrainConfig {
            dataset: args.dataset.clone(),
            arch: hyper.arch.clone(),
            epochs: grid.epochs,
            latent_dim: hyper.latent_dim,
            hidden_dim: hyper.hidden_dim,
            lr: hyper.lr,
            batch_size: grid.batch_size,
            patience: grid.patience,
            artifact_dir: args.artifact_dir.clone(),
            data_dir: args.data_dir.clone(),
            window: args.window,
        };

        let result = match hyper.arch.as_str() {
            "shallow" => train_shallow::<B>(cfg, hyper, dataset.as_ref(), device),
            "deep" => train_deep::<B>(cfg, hyper, dataset.as_ref(), device),
            other => {
                eprintln!("  [skip] unknown arch '{}'", other);
                overall_pb.inc(1);
                continue;
            }
        };

        match result {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("  [error] grid point {}: {}", idx, e),
        }

        overall_pb.inc(1);
    }

    overall_pb.finish_with_message("grid search complete");

    print_results_table(&results);
    save_results_csv(&results, &args.artifact_dir)?;

    Ok(())
}

fn print_results_table(results: &[TrainResult]) {
    if results.is_empty() {
        println!("No results to display.");
        return;
    }

    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| a.best_val_loss.partial_cmp(&b.best_val_loss).unwrap());

    println!("\n══ Grid Search Results (ranked by validation loss) ══");
    println!(
        "{:<8} {:<4} {:>8} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "arch", "lr", "latent", "hidden", "params", "val_loss", "epoch", "secs"
    );
    println!("{}", "─".repeat(68));

    for r in &sorted {
        println!(
            "{:<8} {:<4.0e} {:>8} {:>7} {:>7} {:>8.6} {:>7} {:>7.1}",
            r.hyper.arch,
            r.hyper.lr,
            r.hyper.latent_dim,
            r.hyper.hidden_dim,
            r.param_count,
            r.best_val_loss,
            r.best_epoch,
            r.train_secs,
        );
    }

    if let Some(best) = sorted.first() {
        println!(
            "\n★ Best: {} arch={} lr={:.0e} latent={} hidden={} → val_loss={:.6}",
            best.dataset,
            best.hyper.arch,
            best.hyper.lr,
            best.hyper.latent_dim,
            best.hyper.hidden_dim,
            best.best_val_loss,
        );
        println!("  Saved to: {}", best.artifact_path);
    }
}

fn save_results_csv(results: &[TrainResult], artifact_dir: &str) -> Result<()> {
    if results.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(artifact_dir)?;
    let path = PathBuf::from(artifact_dir).join("grid_results.csv");
    let mut wtr = csv::Writer::from_path(&path)?;
    wtr.write_record([
        "dataset",
        "arch",
        "lr",
        "latent_dim",
        "hidden_dim",
        "params",
        "best_val_loss",
        "best_epoch",
        "total_epochs",
        "train_secs",
        "artifact_path",
    ])?;
    for r in results {
        wtr.write_record(&[
            &r.dataset,
            &r.hyper.arch,
            &r.hyper.lr.to_string(),
            &r.hyper.latent_dim.to_string(),
            &r.hyper.hidden_dim.to_string(),
            &r.param_count.to_string(),
            &r.best_val_loss.to_string(),
            &r.best_epoch.to_string(),
            &r.total_epochs.to_string(),
            &r.train_secs.to_string(),
            &r.artifact_path,
        ])?;
    }
    wtr.flush()?;
    println!("\nResults CSV → {}", path.display());
    Ok(())
}
