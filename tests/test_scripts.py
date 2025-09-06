import pytest

from bblm.cli.download_data import download_bblm, download_wos


@pytest.mark.integration
def test_wos_save_path(tmp_path):
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)

    download_wos(tmp_path)

    assert (tmp_path / "data" / "wos").exists()
    assert (tmp_path / "data" / "wos" / "X.txt").exists()
    assert (tmp_path / "data" / "wos" / "Y.txt").exists()
    assert (tmp_path / "data" / "wos" / "YL1.txt").exists()
    assert (tmp_path / "data" / "wos" / "YL2.txt").exists()


@pytest.mark.integration
def test_bblm_save_path(tmp_path):
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)

    download_bblm(tmp_path)

    assert (tmp_path / "data" / "train_10M").exists()
    assert (tmp_path / "data" / "train_10M" / "bnc_spoken.train").exists()
    assert (tmp_path / "data" / "train_10M" / "childes.train").exists()
    assert (tmp_path / "data" / "train_10M" / "gutenberg.train").exists()
    assert (tmp_path / "data" / "train_10M" / "open_subtitles.train").exists()
    assert (tmp_path / "data" / "train_10M" / "simple_wiki.train").exists()
    assert (tmp_path / "data" / "train_10M" / "switchboard.train").exists()
    assert not (tmp_path / "data" / "train_10M.zip").exists()
