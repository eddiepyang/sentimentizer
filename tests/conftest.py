from pathlib import Path
import pytest

file_root = Path(__file__).parent.absolute()
root = Path(".").resolve()


@pytest.fixture
def rel_path():
    global root
    path = f"{root}/tests/test_data/archive.zip"
    return path


@pytest.fixture
def relative_root():
    global root
    return root


if __name__ == "__main__":
    print("root is:", file_root, root)
