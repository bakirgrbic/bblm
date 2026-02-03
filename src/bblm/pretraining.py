"""Implements BabyLM strict small data track pretraining task."""

import logging
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM

from bblm.dataset import Dataset
from bblm.utils import auto_choose_device

logger = logging.getLogger("main." + __name__)


def get_file_names() -> list[str]:
    """Gathers all pretraining data file names from data/train_10M dir."""

    return [
        str(data_file)
        for data_file in Path("data/train_10M").glob("[!._]*.train")
    ]


def create_dataset(
    data_files: list[str],
    tokenizer: transformers.AutoTokenizer,
) -> torch.utils.data.Dataset:
    """Create a datalset for pretraining.

    Parameters
    ----------
    data_files
        List of file names to get data from.
    tokenizer
        Transformer tokenizer.

    Returns
    -------
    dataset
        Overridden torch dataset object.
    """

    return Dataset(data_files, tokenizer=tokenizer)


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create a dataloader for pretraining.

    Parameters
    ----------
    dataset
        Overridden torch Dataset object.
    batch_size
        Batch size to use before updating model weights.

    Returns
    -------
    loader
        Dataloader containing pretraining data.

    """

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def pre_train(
    model: AutoModelForMaskedLM,
    loader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: torch.optim.Adam,
    device: str,
) -> None:
    """Run main training loop.

    Parameters
    ----------
    model
        Transformer model to pretrain.
    loader
        dataloader containing pretraining data.
    epochs
        Number of epochs to pretrain for.
    optimizer
        Torch optimizer.
    device
        Which hardware device to use.
    """

    for epoch in range(epochs):
        logger.info(f"Begining pretrain epoch {epoch}")
        loop = tqdm(loader, leave=True)
        model.train()
        losses = []

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            losses.append(loss.item())

        logger.info(f"Epoch {epoch} Mean Training Loss: {np.mean(losses)}")


def pre_train_task(
    model_name: str,
    revision: str,
    loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    save_dir: Path,
    device: str,
) -> None:
    """Run BabyLM pretraining task and logs artifacts.

    Parameters
    ----------
    model_name
        Name of huggingface model or relative file path of a local model.
    revision
        the specific commit of a model to use from huggingface.
    loader
        Torch data loader with pretraining data.
    epochs
        Number of epochs to pretrain for.
    learning_rate
        Learning rate for the optimizer.
    save_dir
        Directory to save model artifacts to.
    device
        desired hardware to train on. If not specified, gpus are chosen if
        available.

    Returns
    -------
    None
        Saves model to save_dir/babylm_pretraining.
    """
    task_name = "babylm_pretraining"

    config = AutoConfig.from_pretrained(model_name, revision=revision)
    model = AutoModelForMaskedLM.from_config(config)

    if not device:
        device = auto_choose_device()

    try:
        model.to(device)
    except (AssertionError, RuntimeError) as err:
        old_device = device
        device = auto_choose_device()
        logger.error(err)
        logger.error(
            f"Could not complete task with {old_device}, but will proceed with {device}"
        )
        model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    logger.info(f"{task_name} start with {device}")
    pre_train(model, loader, epochs, optimizer, device)
    logger.info("{task_name} done!")

    save_dir = save_dir / Path(task_name)

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    logger.info(f"Saving pretrained model {model_name} to {save_dir}")
    model.save_pretrained(save_dir)
    logger.info(f"Saved {model_name}!")
