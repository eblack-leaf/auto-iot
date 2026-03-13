pub mod evaluator;
pub mod scorer;

use anyhow::Result;
use burn::prelude::*;

use crate::cli::InferArgs;

pub use evaluator::evaluate;

pub fn run_infer<B: Backend>(args: &InferArgs, device: &B::Device) -> Result<()> {
    use crate::datasets;
    use crate::models::{DeepAEConfig, ShallowAEConfig};

    let dataset = datasets::load(&args.dataset, &args.data_dir, args.window, 0.1)?;

    println!(
        "\n── Inference: {} │ arch={} │ model={}",
        args.dataset, args.arch, args.model
    );

    let scores = match args.arch.as_str() {
        "shallow" => {
            let cfg = ShallowAEConfig {
                input_dim: dataset.seq_len(),
                hidden_dim: args.hidden_dim,
                latent_dim: args.latent_dim,
            };
            let model = cfg
                .init::<B>(device)
                .load_file(&args.model, &burn::record::CompactRecorder::new(), device)
                .map_err(|e| anyhow::anyhow!("Load failed: {:?}", e))?;
            scorer::score_shallow::<B>(model, dataset.test_samples(), args.batch_size(), device)
        }
        "deep" => {
            let cfg = DeepAEConfig {
                input_dim: dataset.seq_len(),
                hidden_dim: args.hidden_dim,
                latent_dim: args.latent_dim,
            };
            let model = cfg
                .init::<B>(device)
                .load_file(&args.model, &burn::record::CompactRecorder::new(), device)
                .map_err(|e| anyhow::anyhow!("Load failed: {:?}", e))?;
            scorer::score_deep::<B>(model, dataset.test_samples(), args.batch_size(), device)
        }
        other => anyhow::bail!("Unknown arch '{}'. Use: shallow, deep", other),
    };

    // Threshold from training reconstruction errors (approximate with test set here).
    let threshold = evaluator::percentile_threshold(&scores, args.threshold_pct);

    println!("\n── Reconstruction error distribution ──");
    crate::eda::histogram::print_error_histogram(&scores, threshold, 20);
    println!("  Threshold ({:.0}th pct): {:.6}", args.threshold_pct, threshold);

    if dataset.has_labels() {
        let labels: Vec<u8> = dataset
            .test_samples()
            .iter()
            .map(|s| s.label.unwrap_or(0))
            .collect();
        evaluate(&scores, &labels, threshold);
    } else {
        println!("  No labels available — printing top anomalies by reconstruction error:");
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
            println!("    #{:<3} sample {:>5}  error={:.6}", rank + 1, idx, score);
        }
    }

    Ok(())
}

// Add batch_size to InferArgs via extension trait to avoid changing cli.rs
trait InferArgsExt {
    fn batch_size(&self) -> usize;
}
impl InferArgsExt for InferArgs {
    fn batch_size(&self) -> usize {
        128
    }
}
