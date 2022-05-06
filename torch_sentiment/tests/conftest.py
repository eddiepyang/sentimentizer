from pathlib import Path
import pytest

file_root = Path(__file__).parent.absolute()
relative_root = Path(".").resolve()

@pytest.fixture
def rel_path():
    global relative_root
    path = f"{relative_root}/torch_sentiment/tests/test_data/archive.zip"
    return path

if __name__=="__main__":

    print("root is:", file_root, relative_root)
