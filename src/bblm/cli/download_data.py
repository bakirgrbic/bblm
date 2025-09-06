#!/usr/bin/env python3
"""Script that downloads and sets up data directory."""

import zipfile

import osfclient
from huggingface_hub import snapshot_download

from bblm.utils.log import find_parent_path


def download_wos(current_dir: str) -> None:
    """Gets web of science data from huggingface."""
    print("Creating data/wos/ dir")

    project_root_dir = find_parent_path(current_dir)
    save_dir = project_root_dir / "data" / "wos"

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    print("Downloading wos data")

    snapshot_download(
        repo_id="bakirgrbic/web-of-science",
        local_dir=str(save_dir),
        repo_type="dataset",
        allow_patterns="*.txt",
    )

    print("Wos download complete!")


def download_bblm(current_dir: str) -> None:
    """Gets BabyLM challenge data for the 10M track."""
    BABYLM_PROJECT_ID = "ad7qg"
    TARGET_ZIP = "train_10M.zip"

    project_root_dir = find_parent_path(current_dir)
    save_dir = project_root_dir / "data"
    zip_file = save_dir / TARGET_ZIP

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    # Connect to osf project that has babylm 2024 data
    osf = osfclient.OSF()
    project = osf.project(BABYLM_PROJECT_ID)

    print(f"Downloading {TARGET_ZIP}!")

    for storage in project.storages:
        for file in storage.files:
            if file.name == TARGET_ZIP:
                try:
                    with open(zip_file, "wb") as f:
                        file.write_to(f)
                except Exception as e:
                    print(f"Failed to download {file.name}. Error {e}")
                    return

                break  # currently only need one zip

    print(f"Unzipping {TARGET_ZIP}!")

    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"Successfully unzipped {TARGET_ZIP}")

    except zipfile.BadZipFile:
        print(f"{TARGET_ZIP} is not a valid zip")
    except FileNotFoundError:
        print(f"Could not find {TARGET_ZIP}")

    print(f"Deleting {TARGET_ZIP}!")

    if zip_file.exists():
        zip_file.unlink()
        print(f"Successfully deleted {TARGET_ZIP}")
    else:
        print(f"{TARGET_ZIP} was already deleted")


def main() -> None:
    """Main entrypoint for downloading data."""
    download_wos(__file__)
    download_bblm(__file__)


if __name__ == "__main__":
    main()
