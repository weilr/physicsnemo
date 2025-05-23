import numpy as np
import torch
from typing import Dict, Optional, Union

class EarlyStopping:
    def __init__(
        self,
        patience: int = 40,
        delta: float = 0.001,
        verbose: bool = True,
        save_state_fn=None,
        logger=None,
        metric_weights: Optional[Dict[str, float]] = None,
        mode: str = "min",
    ):
        """
        Early stopping handler that can monitor multiple metrics.
        
        Args:
            patience: Number of epochs to wait before stopping
            delta: Minimum change in score to qualify as improvement
            verbose: Whether to print messages
            save_state_fn: Function to save checkpoint
            logger: Logger object
            metric_weights: Dictionary of metrics and their weights for combined score
                            If None, only a single val_loss is used (backward compatible)
            mode: 'min' if lower values are better, 'max' if higher values are better
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_fn = save_state_fn or (lambda *a, **k: None)
        self.logger = logger
        self.metric_weights = metric_weights or {"val_loss": 1.0}
        self.mode = mode
        
        self.best_score = None
        self.best_metrics = {}
        self.val_min = np.inf
        self.counter = 0
        self.early_stop = False
        
        # Sign for score comparison (1 for max, -1 for min)
        self.sign = 1 if mode == "max" else -1
    
    @DeprecationWarning
    def evaluate_metrics(self, val_loss: float, *save_args, **save_kwargs):
        """
        Backward compatible call with only validation loss
        """
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

    def __call__(self, metrics: Dict[str, Union[float, torch.Tensor]], *save_args, **save_kwargs):
        """
        Evaluate multiple metrics and decide whether to save checkpoint
        
        Args:
            metrics: Dictionary of metric names and values
        """
        # Convert any tensor values to float
        metrics = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        # Compute weighted score
        score = 0.0
        score_str = ""
        for name, weight in self.metric_weights.items():
            if name in metrics:
                score += self.sign * weight * metrics[name]
                score_str += f"{name}: {metrics[name]:.6f}, w: {weight:.2f}|"

        self.logger.info(f"[EarlyStopping] score: {score:.6f}, {score_str}")
        
        if self.best_score is None:
            self._save_checkpoint_multi(metrics, score, *save_args, **save_kwargs)
        elif score < self.best_score + self.sign * self.delta:
            # No improvement
            self.counter += 1
            if self.verbose: 
                self.logger.info(f"[EarlyStopping] counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found
            self._save_checkpoint_multi(metrics, score, *save_args, **save_kwargs)
            self.counter = 0

    def _save_checkpoint(self, val_loss, *args, **kwargs):
        if self.verbose:
            self.logger.info(f"[EarlyStopping] Validation loss decreased: {self.val_min:.6f} → {val_loss:.6f}")
        self.val_min = val_loss
        self.best_score = -val_loss
        
        self.save_fn(*args, **kwargs)

    def _save_checkpoint_multi(self, metrics, score, *args, **kwargs):
        if self.verbose:
            if self.best_metrics:
                message = "[EarlyStopping] Metrics improved: "
                for k in self.metric_weights.keys():
                    if k in metrics and k in self.best_metrics:
                        message += f"{k}: {self.best_metrics[k]:.6f} → {metrics[k]:.6f}, "
                self.logger.info(message)
            else:
                self.logger.info(f"[EarlyStopping] Initial checkpoint saved")
                
        self.best_score = score
        self.best_metrics = metrics.copy()
        if "val_loss" in metrics:
            self.val_min = metrics["val_loss"]
        
        self.save_fn(*args, **kwargs)
