//! Model size tracking and parameter counting.

use burn::prelude::*;

/// Count total trainable scalar parameters in a Burn module.
///
/// Works by visiting every parameter tensor and summing element counts.
/// The module is consumed (use `.clone()` to preserve it).
pub fn count_parameters<B: Backend, M: Module<B>>(model: &M) -> usize {
    // num_params() counts all trainable scalar elements across all parameter tensors.
    model.num_params()
}

/// Print a formatted parameter count summary.
pub fn print_param_summary(name: &str, param_count: usize) {
    let (display, unit) = if param_count >= 1_000_000 {
        (param_count as f64 / 1_000_000.0, "M")
    } else if param_count >= 1_000 {
        (param_count as f64 / 1_000.0, "K")
    } else {
        (param_count as f64, "")
    };

    println!(
        "  {:>12} │ {:.2}{} parameters",
        name, display, unit
    );
}

/// Estimate RAM footprint in bytes, assuming f32 (4 bytes per param).
pub fn param_bytes(count: usize) -> usize {
    count * 4
}
