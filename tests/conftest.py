import os
from pathlib import Path

import pytest

# Ray recommends at least 50% of available memory for the object store.
# The default is ~40% which triggers a warning. Set this before ray.init().
os.environ.setdefault("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", "0.5")

# Enable Ray Data rich progress bars and suppress the "new progress UI" info message.
# These must be set before ray.data.context is imported so the defaults are picked up.
# See https://docs.ray.io/en/2.55.1/data/api/doc/ray.data.DataContext.html
os.environ.setdefault("RAY_DATA_ENABLE_RICH_PROGRESS_BARS", "1")
os.environ.setdefault("RAY_TQDM", "0")

file_root = Path(__file__).parent.absolute()
root = Path(".").resolve()


@pytest.fixture
def rel_path():
    path = f"{root}/tests/test_data/archive.zip"
    return path


@pytest.fixture
def relative_root():
    return root


if __name__ == "__main__":
    print("root is:", file_root, root)
