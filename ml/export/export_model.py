"""
Export trained GlaucomaProgressionModel to TorchScript.

Usage:
    python ml/export/export_model.py \
        --checkpoint ml/checkpoints/best_model.pt \
        --output ml/export/glaucoma_model.torchscript.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    config_path: str = "ml/config/base_config.yaml",
) -> None:
    """
    Load a trained model checkpoint and export to TorchScript.

    Args:
        checkpoint_path : Path to .pt checkpoint file.
        output_path     : Destination path for TorchScript model.
        config_path     : Path to base_config.yaml.
    """
    import yaml
    from ml.models.full_pipeline import GlaucomaProgressionModel

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = GlaucomaProgressionModel(
        graph_path=cfg["paths"]["graph"],
        gnn_hidden_dim=cfg["model"]["gnn"]["hidden_dim"],
        gnn_output_dim=cfg["model"]["gnn"]["output_dim"],
        gru_hidden_dim=cfg["model"]["spatiotemporal"]["hidden_dim"],
        gru_layers=cfg["model"]["spatiotemporal"]["gru_layers"],
        time_encoding_dim=cfg["model"]["spatiotemporal"]["time_encoding_dim"],
        dropout=0.0,  # disable dropout for inference
    )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Benchmark inference time before export
    dummy_sens = torch.rand(1, 5, 54)
    dummy_time = torch.tensor([[0.0, 13.0, 26.0, 39.0, 52.0]])
    dummy_lengths = torch.tensor([5])

    times = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            model(dummy_sens, dummy_time, dummy_lengths)
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    logger.info("Avg inference time (CPU): %.1f ms (target < 200ms)", avg_ms)
    if avg_ms > 200:
        logger.warning("Inference time %.1f ms exceeds 200ms target", avg_ms)

    # TorchScript export via tracing
    with torch.no_grad():
        scripted = torch.jit.trace(
            model,
            (dummy_sens, dummy_time, dummy_lengths),
            strict=False,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(output_path)
    logger.info("Exported TorchScript model → %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="ml/export/glaucoma_model.torchscript.pt")
    parser.add_argument("--config", default="ml/config/base_config.yaml")
    args = parser.parse_args()
    export_torchscript(args.checkpoint, args.output, args.config)
