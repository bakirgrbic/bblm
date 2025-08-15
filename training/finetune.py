#!/usr/bin/env python3
"""Script that finetunes local or huggingface models on the web of science task."""

import argparse
import logging

from transformers import AutoTokenizer

from bblm.tasks.finetuning.web_of_science.wos import (create_dataloaders,
                                                      load_data, wos_task)
from bblm.utils.log import setup_logger


def get_parser() -> argparse.ArgumentParser:
    """Parser to read cli arguments."""
    parser = argparse.ArgumentParser(
        prog="python3 -m bblm.finetune.py",
        description="""Script that finetunes local or huggingface models on
                       the web of science task.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="bakirgrbic/electra-tiny",
        help="""Name of huggingface model or relative file path
                of a local model.""",
    )
    parser.add_argument(
        "-ml",
        "--max_len",
        type=int,
        default=128,
        help="""Maximum length of words tokenizer will read for a given text.""",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="""Batch size to use before updating model weights.""",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=3,
        help="""Number of epochs to finetune for.""",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=2e-05,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="",
        help="Device to train with. If left empty, gpu devices will be prioritized to be used.",
    )

    return parser


def get_args() -> argparse.Namespace:
    """Read cli arguments."""
    parser = get_parser()

    return parser.parse_known_args()[0]


args = get_args()

logger = logging.getLogger("main")
save_dir = setup_logger(logger)

logger.info("Setting up finetune task")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_data, train_labels, test_data, test_labels = load_data()
training_loader, testing_loader = create_dataloaders(
    train_data,
    train_labels,
    test_data,
    test_labels,
    tokenizer,
    args.max_len,
    args.batch_size,
)

logger.info(
    f"Hyperparameters:  model_name={args.model_name}, max_length={args.max_len}, batch_size={args.batch_size}, epochs={args.epochs}, learning_rate={args.learning_rate}"
)
wos_task(
    args.model_name,
    training_loader,
    testing_loader,
    args.epochs,
    args.learning_rate,
    args.device,
)
logger.info("End of finetune task")
