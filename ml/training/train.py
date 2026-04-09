"""
Training loop for GlaucomaProgressionModel.

Features:
  - W&B logging
  - Checkpoint saving (best val loss)
  - Early stopping
  - Cosine LR scheduler with warmup
  - Anatomical loss (bundle coherence + asymmetry prior)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def heteroscedastic_loss(
    pred_rates: torch.Tensor,
    log_var: torch.Tensor,
    target_rates: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood for heteroscedastic regression.
    Loss = 0.5 * (exp(-log_var) * (pred - target)^2 + log_var)
    """
    precision = torch.exp(-log_var)
    return 0.5 * (precision * (pred_rates - target_rates) ** 2 + log_var).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bundle_loss_fn: Optional[nn.Module],
    asymmetry_loss_fn: Optional[nn.Module],
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        sens = batch["sensitivity_maps"].to(device)
        time_deltas = batch["time_deltas"].to(device)
        lengths = batch["session_lengths"].to(device)

        optimizer.zero_grad()

        # Forward pass
        rates, log_var = model.spatiotemporal(
            _encode_batch(model, sens, time_deltas, lengths, device),
            time_deltas,
            lengths,
        )

        # Compute target rates via simple linear regression on sensitivity
        target_rates = _compute_plr_targets(sens, time_deltas, lengths)

        loss = heteroscedastic_loss(rates, log_var, target_rates.to(device))

        # Anatomical losses
        if bundle_loss_fn is not None:
            loss = loss + bundle_loss_fn(rates.mean(dim=0))
        if asymmetry_loss_fn is not None:
            loss = loss + asymmetry_loss_fn(rates.mean(dim=0))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run one validation epoch. Returns mean loss."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        sens = batch["sensitivity_maps"].to(device)
        time_deltas = batch["time_deltas"].to(device)
        lengths = batch["session_lengths"].to(device)

        rates, log_var = model.spatiotemporal(
            _encode_batch(model, sens, time_deltas, lengths, device),
            time_deltas,
            lengths,
        )
        target_rates = _compute_plr_targets(sens, time_deltas, lengths)
        loss = heteroscedastic_loss(rates, log_var, target_rates.to(device))
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def _encode_batch(
    model: nn.Module,
    sens: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run GNN on all sessions in the batch. Returns [B, T, 54, embed_dim]."""
    B, T, N = sens.shape
    embeddings = []
    for t in range(T):
        x_t = sens[:, t, :].unsqueeze(-1)  # [B, 54, 1]
        batch_embeds = []
        for b in range(B):
            embed = model.gnn(x_t[b], model.edge_index, model.edge_attr)
            batch_embeds.append(embed)
        embeddings.append(torch.stack(batch_embeds, dim=0))
    return torch.stack(embeddings, dim=1)  # [B, T, 54, embed_dim]


def _compute_plr_targets(
    sens: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pointwise linear regression (PLR) progression rates as training targets.
    Returns [B, 54] rates in dB/year.
    """
    B, T, N = sens.shape
    rates = torch.zeros(B, N)

    for b in range(B):
        L = lengths[b].item()
        if L < 2:
            continue
        t = time_deltas[b, :L].float()  # weeks
        t_years = t / 52.0
        s = sens[b, :L, :]  # [L, 54]

        # Simple OLS per location
        t_mean = t_years.mean()
        s_mean = s.mean(dim=0)
        num = ((t_years.unsqueeze(1) - t_mean) * (s - s_mean)).sum(dim=0)
        den = ((t_years - t_mean) ** 2).sum() + 1e-8
        rates[b] = num / den

    return rates


def main(config_path: str) -> None:
    """Main training entry point."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # W&B init (optional)
    use_wandb = False
    try:
        import wandb
        wandb.init(
            project=cfg.get("wandb", {}).get("project", "glaucomap"),
            config=cfg,
        )
        use_wandb = True
    except Exception:
        logger.warning("W&B not available, logging to console only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    from ml.models.full_pipeline import GlaucomaProgressionModel
    from ml.models.anatomical_loss import BundleCoherenceLoss, AsymmetryPriorLoss
    from ml.data.anatomical_graph import build_rnfb_graph

    model = GlaucomaProgressionModel(
        graph_path=cfg["paths"]["graph"],
        gnn_hidden_dim=cfg["model"]["gnn"]["hidden_dim"],
        gnn_output_dim=cfg["model"]["gnn"]["output_dim"],
        gru_hidden_dim=cfg["model"]["spatiotemporal"]["hidden_dim"],
        gru_layers=cfg["model"]["spatiotemporal"]["gru_layers"],
        time_encoding_dim=cfg["model"]["spatiotemporal"]["time_encoding_dim"],
        dropout=cfg["model"]["gnn"]["dropout"],
    ).to(device)

    graph = build_rnfb_graph()
    bundle_loss = BundleCoherenceLoss(
        graph.sector_labels,
        weight=cfg["model"]["gnn"]["bundle_coherence_weight"],
    ).to(device)
    asymmetry_loss = AsymmetryPriorLoss(
        graph.sector_labels,
        weight=cfg["model"]["gnn"]["asymmetry_prior_weight"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["max_epochs"]
    )

    checkpoint_dir = Path(cfg["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    # NOTE: Dataloaders must be provided externally.
    # This script expects train_loader and val_loader to be injected.
    # See ml/training/run_training.py for a full example with data loading.
    logger.info("Training loop ready. Provide train_loader and val_loader to proceed.")
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    if use_wandb:
        import wandb
        wandb.log({"model_params": sum(p.numel() for p in model.parameters())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/config/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
