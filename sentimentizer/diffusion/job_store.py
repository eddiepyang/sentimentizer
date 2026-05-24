"""Ray actor that tracks image generation job metadata.

A cluster-wide singleton (detached named actor) that stores job
records keyed by ``job_id``. Status is managed explicitly:
the dispatcher pushes terminal status (succeeded/failed)
via a background asyncio task after awaiting the ObjectRef.

ObjectRefs cannot be passed across Ray actor boundaries — Ray
auto-resolves them to their values. The dispatcher therefore holds
the refs locally and pushes terminal results here.

Lifecycle:
    - Bootstrapped once in ``serve/app.py:main()`` before ``serve.run()``.
    - Accessible from any dispatcher replica via ``ray.get_actor("diffusion_job_store")``.
    - Destroyed on ``ray.shutdown()`` (detached actors are cleaned up when
      the cluster is destroyed). Tests must call ``ray.kill(actor)`` in
      teardown to avoid leaking actors between cases.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

import ray

from sentimentizer.serve.diffusion_models import JobResponse


@dataclass
class JobRecord:
    job_id: str
    api_key_prefix: str
    created: int
    model: str
    user: str | None = None
    status: str = "processing"
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    terminal_at: int | None = None


class JobStoreLogic:
    """Plain-Python job metadata store (no Ray dependency).

    Unit-testable directly. Wrapped by ``JobStore`` (a Ray actor) for
    cluster-wide access.
    """

    def __init__(self, ttl_s: int = 3600) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._by_key: dict[str, list[str]] = {}
        self._ttl_s = ttl_s

    def submit(self, model: str, user: str | None, api_key: str) -> str:
        job_id = f"job_{secrets.token_urlsafe(12)}"
        now = int(time.time())
        rec = JobRecord(
            job_id=job_id,
            api_key_prefix=api_key[:8],
            created=now,
            model=model,
            user=user,
            status="processing",
        )
        self._jobs[job_id] = rec
        self._by_key.setdefault(rec.api_key_prefix, []).append(job_id)
        return job_id

    def get(self, job_id: str, api_key: str) -> dict[str, Any] | None:
        rec = self._jobs.get(job_id)
        if rec is None or rec.api_key_prefix != api_key[:8]:
            return None
        return _rec_to_response(rec).model_dump()

    def set_succeeded(self, job_id: str, result: dict[str, Any]) -> bool:
        rec = self._jobs.get(job_id)
        if rec is None:
            return False
        now = int(time.time())
        rec.status = "succeeded"
        rec.result = result
        rec.terminal_at = now
        return True

    def set_failed(self, job_id: str, error_code: str, error_message: str) -> bool:
        rec = self._jobs.get(job_id)
        if rec is None:
            return False
        now = int(time.time())
        rec.status = "failed"
        rec.error = {"code": error_code, "message": error_message}
        rec.terminal_at = now
        return True

    def cancel(self, job_id: str, api_key: str) -> dict[str, Any] | None:
        rec = self._jobs.get(job_id)
        if rec is None or rec.api_key_prefix != api_key[:8]:
            return None
        if rec.status not in ("succeeded", "failed", "canceled"):
            rec.status = "canceled"
            rec.terminal_at = int(time.time())
        return _rec_to_response(rec).model_dump()

    def list_jobs(
        self,
        api_key: str,
        page_size: int = 20,
        page_token: str | None = None,
        status_filter: str | None = None,
        model_filter: str | None = None,
    ) -> dict[str, Any]:
        prefix = api_key[:8]
        job_ids = self._by_key.get(prefix, [])

        items: list[dict[str, Any]] = []
        for jid in job_ids:
            rec = self._jobs.get(jid)
            if rec is None:
                continue
            if model_filter is not None and rec.model != model_filter:
                continue
            if status_filter is not None and rec.status != status_filter:
                continue
            items.append(_rec_to_response(rec).model_dump())

        start = 0
        if page_token is not None:
            try:
                start = int(page_token)
            except ValueError:
                start = 0

        page = items[start : start + page_size]
        next_token = None
        if start + page_size < len(items):
            next_token = str(start + page_size)

        return {"jobs": page, "next_page_token": next_token}

    def reap_expired(self) -> int:
        now = int(time.time())
        expired: list[str] = []
        for jid, rec in self._jobs.items():
            if rec.terminal_at is not None and now - rec.terminal_at > self._ttl_s:
                expired.append(jid)
        for jid in expired:
            rec = self._jobs.pop(jid)
            key_jobs = self._by_key.get(rec.api_key_prefix, [])
            if jid in key_jobs:
                key_jobs.remove(jid)
        return len(expired)


def _rec_to_response(rec: JobRecord) -> JobResponse:
    updated = rec.terminal_at or int(time.time())
    return JobResponse(
        job_id=rec.job_id,
        status=rec.status,
        created=rec.created,
        updated=updated,
        model=rec.model,
        user=rec.user,
        result=rec.result,
        error=rec.error,
    )


@ray.remote(num_cpus=0.1)
class JobStore:
    """Thin Ray actor wrapper around ``JobStoreLogic``."""

    def __init__(self, ttl_s: int = 3600) -> None:
        self._logic = JobStoreLogic(ttl_s=ttl_s)

    def submit(self, model: str, user: str | None, api_key: str) -> str:
        return self._logic.submit(model, user, api_key)

    def get(self, job_id: str, api_key: str) -> dict[str, Any] | None:
        return self._logic.get(job_id, api_key)

    def set_succeeded(self, job_id: str, result: dict[str, Any]) -> bool:
        return self._logic.set_succeeded(job_id, result)

    def set_failed(self, job_id: str, error_code: str, error_message: str) -> bool:
        return self._logic.set_failed(job_id, error_code, error_message)

    def cancel(self, job_id: str, api_key: str) -> dict[str, Any] | None:
        return self._logic.cancel(job_id, api_key)

    def list_jobs(
        self,
        api_key: str,
        page_size: int = 20,
        page_token: str | None = None,
        status_filter: str | None = None,
        model_filter: str | None = None,
    ) -> dict[str, Any]:
        return self._logic.list_jobs(api_key, page_size, page_token, status_filter, model_filter)

    def reap_expired(self) -> int:
        return self._logic.reap_expired()
