from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


def push_model_to_hub(
    local_path: str | Path,
    repo_id: str,
    model_type: str,
) -> None:
    """Upload model weights to the Hugging Face Hub.

    Args:
        local_path: Path to the local .pth weight file.
        repo_id: Hugging Face repository ID (e.g., 'username/repo').
        model_type: Model type ('rnn', 'encoder', or 'decoder') used for the filename.
    """
    path = Path(local_path)
    if not path.exists():
        logger.error(f"Local weight file not found: {path}")
        return

    filename = f"{model_type}_weights.pth"
    api = HfApi()

    logger.info(f"Pushing {filename} to Hugging Face Hub: {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=f"Update {model_type} weights",
        )
        logger.info(f"Successfully pushed weights to {repo_id}/{filename}")
    except Exception as e:
        logger.error(f"Failed to push weights to Hugging Face Hub: {e}")


def pull_model_from_hub(
    repo_id: str,
    model_type: str,
    local_path: str | Path,
) -> bool:
    """Download model weights from the Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/repo').
        model_type: Model type ('rnn', 'encoder', or 'decoder') used for the filename.
        local_path: Local path where the weights should be saved.

    Returns:
        True if the download was successful, False otherwise.
    """
    filename = f"{model_type}_weights.pth"
    local_path = Path(local_path)

    logger.info(f"Pulling {filename} from Hugging Face Hub: {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        # Ensure target directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the downloaded file to the expected local path
        import shutil
        shutil.copy(downloaded_path, local_path)

        logger.info(f"Successfully pulled weights to {local_path}")
        return True
    except EntryNotFoundError:
        logger.warning(f"Weights file {filename} not found in repository {repo_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to pull weights from Hugging Face Hub: {e}")
        return False
