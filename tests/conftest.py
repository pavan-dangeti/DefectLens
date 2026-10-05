from __future__ import annotations

import pytest


@pytest.fixture
def dataset_dir(tmp_path):
    import shutil

    dest = tmp_path / "datasets"
    shutil.copytree("configs/datasets", dest)
    return dest
