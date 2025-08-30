#!/usr/bin/env python3
"""Script that downloads and sets up data directory."""

import osfclient
from huggingface_hub import snapshot_download


def download_wos() -> None:
    """Gets web of science data from huggingface."""
    snapshot_download(repo_id="bakirgrbic/web-of-science", local_dir="")
    # TODO Does this snapshot include the folder all the files are in or do I gotta make the folder?


def download_bblm() -> None:
    """Gets BabyLM challenge data for the 10M track."""
    # TODO read docs and build it
    osfclient()
    return


if __name__ == "__main__":
    print()
