from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, HF_WEIGHTS_REPOS

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
        # Check if error is because repo doesn't exist
        if "Repository Not Found" in str(e) or "404 Client Error" in str(e):
            logger.info(f"Repository {repo_id} not found. Attempting to create it...")
            try:
                api.create_repo(repo_id=repo_id, exist_ok=True)
                # Retry upload after creation
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    commit_message=f"Initial {model_type} weights upload",
                )
                logger.info(f"Successfully created repo and pushed weights to {repo_id}/{filename}")
                return
            except Exception as create_error:
                logger.error(f"Failed to create repository or retry upload: {create_error}")
                return

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


def download_weights(model_type: str, local_path: str | Path) -> Path | None:
    """Download model weights from the Hugging Face Hub.

    Looks up the per-model repository from ``HF_WEIGHTS_REPOS`` and downloads
    the ``{model_type}_weights.pth`` file to *local_path*.

    Args:
        model_type: One of 'rnn', 'encoder', or 'decoder'.
        local_path: Destination path on the local filesystem.

    Returns:
        The path to the downloaded weights file, or ``None`` if the download
        failed (e.g. the repo or file doesn't exist, or there is no network).
    """
    repo_id = HF_WEIGHTS_REPOS.get(model_type)
    if repo_id is None:
        logger.error(f"No Hugging Face repo configured for model type {model_type!r}")
        return None

    local_path = Path(local_path)
    filename = f"{model_type}_weights.pth"

    logger.info(f"Downloading {filename} from {repo_id} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        # Ensure target directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the downloaded file to the expected local path
        import shutil

        shutil.copy2(downloaded_path, local_path)
        logger.info(f"Weights downloaded to {local_path}")
        return local_path

    except EntryNotFoundError:
        logger.warning(f"Weights file {filename} not found in {repo_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to download weights from Hugging Face Hub: {e}")
        return None
