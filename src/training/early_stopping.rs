/// Tracks validation loss and signals when training should stop.
pub struct EarlyStopping {
    patience: usize,
    best_loss: f64,
    no_improve_count: usize,
    pub best_epoch: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize) -> Self {
        EarlyStopping {
            patience,
            best_loss: f64::INFINITY,
            no_improve_count: 0,
            best_epoch: 0,
        }
    }

    /// Feed the current epoch's validation loss.
    /// Returns `true` if this was the best seen so far (→ save the model).
    pub fn update(&mut self, epoch: usize, val_loss: f64) -> StopResult {
        if val_loss < self.best_loss {
            self.best_loss = val_loss;
            self.no_improve_count = 0;
            self.best_epoch = epoch;
            StopResult::Improved
        } else {
            self.no_improve_count += 1;
            if self.no_improve_count >= self.patience {
                StopResult::Stop
            } else {
                StopResult::Continue
            }
        }
    }

    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

#[derive(Debug, PartialEq)]
pub enum StopResult {
    /// Validation loss improved — caller should save this model.
    Improved,
    /// No improvement, but patience not yet exhausted.
    Continue,
    /// Patience exhausted — caller should halt training.
    Stop,
}
