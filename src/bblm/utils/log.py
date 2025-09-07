"""Helper method for logging."""

import datetime
import logging
import sys
from pathlib import Path


def setup_logger(current_dir: str, logger: logging.Logger) -> Path:
    """Preconfigures a given logger and creates directory to save task artifacts.

    Parameters
    ----------
    current_dir:
        Name of the current directory.
    logger
        Logger to preconfigure.

    Returns
    -------
    save_dir
        Directory where model artifacts and log files are saved to.
    """
    logger.setLevel(logging.INFO)

    project_root_dir = find_parent_path(current_dir)
    experiment_dir = Path("version_" + str(datetime.datetime.now()))
    save_dir = project_root_dir / "log" / experiment_dir

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


def find_parent_path(current_dir: str, target: str = ".git") -> Path:
    """Finds the absolute path of parent target dir.

    Parameters
    ----------
    current_dir:
        Name of the current directory.
    target
        Name of parent directory to search for.

    Returns
    -------
    absolute_path
        The aboslute path of parent.
    """
    absolute_path = Path(current_dir).resolve()

    for parent in [absolute_path] + list(absolute_path.parents):
        if (parent / target).exists():
            return parent

    raise FileNotFoundError(f"Could not find {target}")
