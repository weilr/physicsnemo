import numpy as np

class EarlyStopping:
    def __init__(
        self,
        patience: int = 40,
        delta: float = 0.001,
        verbose: bool = True,
        save_state_fn=None,
        logger=None,
    ):
        self.patience  = patience
        self.delta     = delta
        self.verbose   = verbose
        self.save_fn   = save_state_fn or (lambda *a, **k: None)
        self.logger    = logger
        self.best_score   = None
        self.val_min      = np.inf
        self.counter      = 0
        self.early_stop   = False


    def __call__(self, val_loss: float, *save_args, **save_kwargs):
        score = -val_loss
        if self.best_score is None:
            self._save_checkpoint(val_loss, *save_args, **save_kwargs)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: 
                self.logger.info(f"[EarlyStopping] counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._save_checkpoint(val_loss, *save_args, **save_kwargs)
            self.counter = 0


    def _save_checkpoint(self, val_loss, *args, **kwargs):
        if self.verbose:
            self.logger.info(f"[EarlyStopping] Validation loss decreased: {self.val_min:.6f} → {val_loss:.6f}")
        self.val_min    = val_loss
        self.best_score = -val_loss
        
        self.save_fn(*args, **kwargs)
