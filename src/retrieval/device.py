"""Torch device selection for the BGE-M3 query embedder."""

import os


def default_device() -> str:
    """
    Device for the embedder: env EMBEDDER_DEVICE (e.g. "cuda:1"), otherwise
    "cuda" if available, otherwise "cpu".
    """
    device = os.getenv("EMBEDDER_DEVICE")
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
