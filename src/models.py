"""
models.py

TFT and NHiTS model factories.
"""
import logging

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models import NHiTS

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# PART 2: TFT MODEL
# ══════════════════════════════════════════════════════════════════

def build_tft(training_dataset, cfg):
    """
    Build Temporal Fusion Transformer.

    Key hyperparameters:
      hidden_size:        main network width (larger = more capacity)
      attention_head_size: multi-head attention (interpretability)
      dropout:            regularization
      hidden_continuous_size: continuous variable embedding size
      loss:               QuantileLoss → gives prediction intervals [10%, 50%, 90%]
    """
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate           = cfg["learning_rate"],
        hidden_size             = cfg["tft_hidden_size"],
        attention_head_size     = cfg["tft_attention_head_size"],
        dropout                 = cfg["tft_dropout"],
        hidden_continuous_size  = cfg["tft_hidden_continuous_size"],
        loss                    = QuantileLoss(quantiles=cfg["tft_quantiles"]),
        log_interval            = 50,
        reduce_on_plateau_patience = 4,
        optimizer               = "adamw",
    )
    print(f"\nTFT parameters: {tft.size() / 1e3:.1f}k")
    return tft


def build_nhits(training_dataset, cfg):
    """
    NHiTS (Neural Hierarchical Interpolation for Time Series) baseline for comparison.
    Uses same dataset interface as TFT for fair comparison.
    """
    model = NHiTS.from_dataset(
        training_dataset,
        learning_rate  = cfg["learning_rate"],
        weight_decay   = cfg["nhits_weight_decay"],
        loss           = MAE(),
        hidden_size    = cfg["nhits_hidden_size"],
        optimizer      = "adamw",
    )
    print(f"NHiTS parameters: {model.size() / 1e3:.1f}k")
    return model
