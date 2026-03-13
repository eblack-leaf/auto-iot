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
    let dataset = datasets::load(&args.dataset, &args.data_dir, args.window, args.val_split)?;

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
            batch_size: args.batch_size,
            patience: grid.patience,
            artifact_dir: args.artifact_dir.clone(),
            data_dir: args.data_dir.clone(),
            window: args.window,
            clean_train: args.clean_train,
            val_split: args.val_split,
            scheduler: args.scheduler.clone(),
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
            Ok(r) => {
                overall_pb.set_message(format!(
                    "{} arch={} lr={:.0e} ld={} hd={}  auroc={}  val={:.6}",
                    args.dataset, r.hyper.arch, r.hyper.lr, r.hyper.latent_dim, r.hyper.hidden_dim,
                    r.auroc.map(|a| format!("{:.4}", a)).unwrap_or("n/a".into()),
                    r.best_val_loss,
                ));
                results.push(r);
            }
            Err(e) => eprintln!("  [error] grid point {}: {}", idx, e),
        }

        overall_pb.inc(1);
    }

    overall_pb.finish_with_message("grid search complete");

    print_results_table(&results);
    save_results_csv(&results, &args.artifact_dir)?;

    Ok(())
}

fn composite_score(r: &TrainResult, has_auroc: bool,
    auroc_min: f64, auroc_max: f64,
    val_min: f64,  val_max: f64,
    param_min: f64, param_max: f64,
) -> f64 {
    let norm = |v: f64, lo: f64, hi: f64| {
        if (hi - lo).abs() < f64::EPSILON { 0.5 } else { (v - lo) / (hi - lo) }
    };
    // All components scaled to [0,1] where 1 = best.
    let auroc_score = if has_auroc {
        norm(r.auroc.unwrap_or(0.0), auroc_min, auroc_max)
    } else {
        0.0
    };
    let val_score   = 1.0 - norm(r.best_val_loss, val_min, val_max);
    let param_score = 1.0 - norm(r.param_count as f64, param_min, param_max);

    if has_auroc {
        0.60 * auroc_score + 0.25 * val_score + 0.15 * param_score
    } else {
        0.70 * val_score   + 0.30 * param_score
    }
}

fn print_results_table(results: &[TrainResult]) {
    if results.is_empty() {
        println!("No results to display.");
        return;
    }

    let has_auroc = results.iter().any(|r| r.auroc.is_some());

    // Compute ranges for normalisation.
    let auroc_vals: Vec<f64> = results.iter().filter_map(|r| r.auroc).collect();
    let auroc_min = auroc_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let auroc_max = auroc_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let val_min  = results.iter().map(|r| r.best_val_loss).fold(f64::INFINITY, f64::min);
    let val_max  = results.iter().map(|r| r.best_val_loss).fold(f64::NEG_INFINITY, f64::max);
    let param_min = results.iter().map(|r| r.param_count as f64).fold(f64::INFINITY, f64::min);
    let param_max = results.iter().map(|r| r.param_count as f64).fold(f64::NEG_INFINITY, f64::max);

    let score = |r: &TrainResult| {
        composite_score(r, has_auroc, auroc_min, auroc_max, val_min, val_max, param_min, param_max)
    };

    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| score(b).partial_cmp(&score(a)).unwrap());

    println!("\n══ Grid Search Results (ranked by composite: auroc×0.6 + val×0.25 + params×0.15) ══");
    println!(
        "{:<8} {:<6} {:>6} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7} {:>6}",
        "arch", "lr", "latent", "hidden", "params", "val_loss", "auroc", "epoch", "secs", "score"
    );
    println!("{}", "─".repeat(85));

    for r in &sorted {
        println!(
            "{:<8} {:<6.0e} {:>6} {:>7} {:>7} {:>8.6} {:>7} {:>7} {:>7.1} {:>6.3}",
            r.hyper.arch,
            r.hyper.lr,
            r.hyper.latent_dim,
            r.hyper.hidden_dim,
            r.param_count,
            r.best_val_loss,
            r.auroc.map(|a| format!("{:.4}", a)).unwrap_or("n/a".into()),
            r.best_epoch,
            r.train_secs,
            score(r),
        );
    }

    if let Some(best) = sorted.first() {
        let metric = format!(
            "auroc={}  val={:.6}  params={}  score={:.3}",
            best.auroc.map(|a| format!("{:.4}", a)).unwrap_or("n/a".into()),
            best.best_val_loss,
            best.param_count,
            score(best),
        );
        println!(
            "\n★ Best: {} arch={} lr={:.0e} latent={} hidden={} → {}",
            best.dataset,
            best.hyper.arch,
            best.hyper.lr,
            best.hyper.latent_dim,
            best.hyper.hidden_dim,
            metric,
        );
        println!("  Saved to: {}", best.artifact_path);
    }
}

fn save_results_csv(results: &[TrainResult], artifact_dir: &str) -> Result<()> {
    if results.is_empty() {
        return Ok(());
    }

    let has_auroc = results.iter().any(|r| r.auroc.is_some());
    let auroc_vals: Vec<f64> = results.iter().filter_map(|r| r.auroc).collect();
    let auroc_min = auroc_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let auroc_max = auroc_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let val_min  = results.iter().map(|r| r.best_val_loss).fold(f64::INFINITY, f64::min);
    let val_max  = results.iter().map(|r| r.best_val_loss).fold(f64::NEG_INFINITY, f64::max);
    let param_min = results.iter().map(|r| r.param_count as f64).fold(f64::INFINITY, f64::min);
    let param_max = results.iter().map(|r| r.param_count as f64).fold(f64::NEG_INFINITY, f64::max);

    fs::create_dir_all(artifact_dir)?;
    let path = PathBuf::from(artifact_dir).join("grid_results.csv");
    let mut wtr = csv::Writer::from_path(&path)?;
    wtr.write_record([
        "dataset", "arch", "lr", "latent_dim", "hidden_dim", "params",
        "best_val_loss", "auroc", "best_epoch", "total_epochs", "train_secs", "score", "artifact_path",
    ])?;
    for r in results {
        let s = composite_score(r, has_auroc, auroc_min, auroc_max, val_min, val_max, param_min, param_max);
        wtr.write_record(&[
            &r.dataset,
            &r.hyper.arch,
            &r.hyper.lr.to_string(),
            &r.hyper.latent_dim.to_string(),
            &r.hyper.hidden_dim.to_string(),
            &r.param_count.to_string(),
            &r.best_val_loss.to_string(),
            &r.auroc.map(|a| format!("{:.6}", a)).unwrap_or("".into()),
            &r.best_epoch.to_string(),
            &r.total_epochs.to_string(),
            &r.train_secs.to_string(),
            &format!("{:.4}", s),
            &r.artifact_path,
        ])?;
    }
    wtr.flush()?;
    println!("\nResults CSV → {}", path.display());
    Ok(())
}
