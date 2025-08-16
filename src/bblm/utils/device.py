"""Helper method for devices."""

import torch


def auto_choose_device() -> str:
    """Prioritzes using cuda or mps enabled gpus over cpu devices.

    Note that there are many more choices for devices to train with
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    return device
