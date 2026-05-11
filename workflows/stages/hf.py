"""HuggingFace Hub push/pull stage.

Heavy imports are DEFERRED to function bodies to avoid importing the ML stack
at module level. Do NOT add module-level imports of torch, ray, gensim, or
sentimentizer.config here.
"""

from __future__ import annotations

from workflows.lifecycle import State, logger


def run_hf_push(state: State, *, repo_id: str | None) -> None:
    """Push model weights to Hugging Face Hub."""
    from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig, weights_path_for
    from sentimentizer.hf import push_model_to_hub

    weights_path = weights_path_for(state.model)
    resolved_repo = repo_id if repo_id is not None else HF_WEIGHTS_REPOS.get(state.model)

    push_model_to_hub(
        local_path=weights_path,
        model_type=state.model,
        repo_id=resolved_repo,
        dict_path=DriverConfig.files.dictionary_file_path,
    )


def run_hf_pull(state: State, *, repo_id: str | None) -> None:
    """Pull model weights from Hugging Face Hub."""
    from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig, weights_path_for
    from sentimentizer.hf import download_weights

    weights_path = weights_path_for(state.model)
    resolved_repo = repo_id if repo_id is not None else HF_WEIGHTS_REPOS.get(state.model)

    result_path = download_weights(
        model_type=state.model,
        local_path=weights_path,
        repo_id=resolved_repo,
        dict_path=DriverConfig.files.dictionary_file_path,
    )
    if result_path:
        logger.info(f"Pulled {state.model} weights from HF Hub to {result_path}")
    else:
        logger.error(f"Failed to pull {state.model} weights from HF Hub")
