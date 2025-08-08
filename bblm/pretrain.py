#/usr/bin/env python3
"""Script that pretrains local or huggingface models."""

import argparse
import logging

from transformers import AutoTokenizer

from bblm.tasks.pretraining.pretraining import (create_dataloader,
                                                create_dataset, get_file_names,
                                                pre_train_task)
from utils.log import setup_logger


def get_parser() -> argparse.ArgumentParser:
    """Parser to read cli arguments."""
    parser = argparse.ArgumentParser(
        prog="python3 -m bblm.pretrain.py",
        description="""Script that pretrains local or huggingface models.""",
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
        "-bs",
        "--batch_size",
        type=int,
        default=8,
        help="""Batch size to use before updating model weights.""",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="""Number of epochs to pretrain for.""",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-04,
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

logger.info("Setting up pretrain task")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

file_names = get_file_names()
dataset = create_dataset(file_names, tokenizer)
loader = create_dataloader(dataset, args.batch_size)

logger.info(
    f"Hyperparameters: model_name={args.model_name}, batch_size={args.batch_size}, epochs={args.epochs}, learning_rate={args.learning_rate}"
)
pre_train_task(
    args.model_name,
    loader,
    args.epochs,
    args.learning_rate,
    save_dir=save_dir,
    device=args.device
)

logger.info(f"Saving tokenizer for {args.model_name} to {save_dir}")
tokenizer.save_pretrained(save_dir)
logger.info(f"Saved tokenizer {args.model_name}!")

logger.info("End of pretrain task")
