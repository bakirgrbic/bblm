"""Helper method for logging."""

import datetime
import logging
import sys
from pathlib import Path


def setup_logger(logger: logging.Logger) -> Path:
    """Preconfigures a given logger and creates directory to save task artifacts.

    Parameters
    ----------
    logger
        Logger to preconfigure.

    Returns
    -------
    save_dir
        Directory to save model artifacts to. Logs outputs to
        log/version_datetime.
    """
    logger.setLevel(logging.INFO)

    curr_dir = Path.cwd()
    experiment_dir = Path("version_" + str(datetime.datetime.now()))
    save_dir = curr_dir / "log" / experiment_dir

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    log_file = save_dir / Path("run.log")

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return save_dir
