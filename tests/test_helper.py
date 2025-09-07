import logging

import pytest

from bblm.utils.log import find_parent_path, setup_logger


@pytest.mark.parametrize(
    "search_start_dir",
    [
        pytest.param(
            ".",
            id="project root",
        ),
        pytest.param(
            "script",
            id="scripts",
        ),
    ],
)
def test_find_parent_path_target_found(tmp_path, search_start_dir):
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)
    current_dir = tmp_path / search_start_dir
    current_dir.mkdir(parents=True, exist_ok=True)

    result = find_parent_path(current_dir)

    assert result == tmp_path


@pytest.mark.parametrize(
    "search_start_dir",
    [
        pytest.param(
            ".",
            id="project root",
        ),
        pytest.param(
            "script",
            id="scripts",
        ),
    ],
)
def test_find_parent_path_target_not_found(tmp_path, search_start_dir):
    current_dir = tmp_path / search_start_dir
    current_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        find_parent_path(current_dir)


@pytest.mark.parametrize(
    "run_dir",
    [
        pytest.param(
            ".",
            id="project root",
        ),
        pytest.param(
            "script",
            id="scripts",
        ),
    ],
)
def test_setup_logger_correct_log_path(tmp_path, run_dir):
    logger = logging.getLogger("main")

    save_dir = setup_logger(run_dir, logger)

    assert save_dir.exists()
