mod cli;
mod config;
mod datasets;
mod eda;
mod error;
mod inference;
mod models;
mod training;

use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;

use cli::{Cli, Commands};

/// WGPU (GPU-accelerated) backend for training.
type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch(args) => {
            datasets::fetch(&args.dataset, &args.output_dir)?;
        }

        Commands::Train(args) => {
            let device = WgpuDevice::default();
            if args.grid_search {
                training::run_grid_search::<AutodiffBackend>(&args, &device)?;
            } else {
                let result = training::run_train::<AutodiffBackend>(&args, &device)?;
                println!(
                    "\nDone. Best val loss: {:.6}  (epoch {})  → {}",
                    result.best_val_loss, result.best_epoch, result.artifact_path
                );
            }
        }

        Commands::Infer(args) => {
            // Inference uses the non-autodiff backend for efficiency.
            let device = WgpuDevice::default();
            inference::run_infer::<Backend>(&args, &device)?;
        }

        Commands::Eda(args) => {
            eda::run_eda(&args)?;
        }
    }

    Ok(())
}
