pub mod histogram;
pub mod latent_export;
pub mod stats;
pub mod tsne;

use anyhow::Result;

use crate::cli::EdaArgs;
use crate::datasets;

pub fn run_eda(args: &EdaArgs) -> Result<()> {
    use std::fs;
    fs::create_dir_all(&args.output_dir)?;

    println!("\n══ EDA: {} ══", args.dataset);

    // Synthetic doesn't need a download.
    let dataset = datasets::load(&args.dataset, &args.data_dir, args.window)?;

    let train = dataset.train_samples();
    let test = dataset.test_samples();

    // ── 1. Descriptive statistics ──────────────────────────────────────────
    println!("\n── Training set statistics ──");
    stats::print_stats(train, dataset.seq_len());

    println!("\n── Test set statistics ──");
    stats::print_stats(test, dataset.seq_len());

    // ── 2. ASCII histograms ────────────────────────────────────────────────
    println!("\n── Amplitude distribution (training, first feature) ──");
    let first_feature: Vec<f64> = train.iter().map(|s| s.features[0] as f64).collect();
    histogram::print_histogram(&first_feature, 20);

    // ── 3. Latent-space analysis (optional — requires a model path) ────────
    if let Some(model_path) = &args.model {
        println!("\n── Latent-space export ──");
        latent_export::export(args, dataset.as_ref(), model_path)?;

        if args.tsne {
            println!("\n── t-SNE on latent vectors ──");
            let latent_csv = format!("{}/latent_vectors.csv", args.output_dir);
            let tsne_csv = format!("{}/tsne_embedding.csv", args.output_dir);
            tsne::run_tsne(&latent_csv, &tsne_csv, 30.0, 1000)?;
            println!("  t-SNE embedding saved to {}", tsne_csv);
        }
    } else {
        println!("\n  Tip: pass --model <path> to enable latent-space and t-SNE analysis.");
    }

    Ok(())
}
