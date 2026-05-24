"""Tests for the JobStore module.

Unit tests exercise JobStoreLogic as a plain Python class (no Ray).
The integration test at the end uses a Ray cluster to verify
cross-replica visibility via a named detached actor.
"""

from __future__ import annotations

import time

import pytest

from sentimentizer.diffusion.job_store import JobRecord, JobStoreLogic, _rec_to_response

# ---------------------------------------------------------------------------
# Unit tests (no Ray)
# ---------------------------------------------------------------------------


class TestJobStoreSubmitAndGet:
    def test_submit_returns_job_id(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        assert job_id.startswith("job_")
        assert len(job_id) == 20

    def test_get_processing_job(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("flux", "user1", "alpha-key-12345678")
        result = store.get(job_id, "alpha-key-12345678")
        assert result is not None
        assert result["status"] == "processing"
        assert result["job_id"] == job_id
        assert result["model"] == "flux"
        assert result["user"] == "user1"

    def test_get_wrong_key_returns_none(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        result = store.get(job_id, "beta-key-12345678")
        assert result is None

    def test_get_missing_job_returns_none(self) -> None:
        store = JobStoreLogic()
        result = store.get("job_nonexistent", "alpha-key-12345678")
        assert result is None


class TestJobStoreSetSucceeded:
    def test_succeeded(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        ok = store.set_succeeded(job_id, {"id": "img_abc", "model": "sd"})
        assert ok is True
        result = store.get(job_id, "alpha-key-12345678")
        assert result["status"] == "succeeded"
        assert result["result"]["id"] == "img_abc"

    def test_succeeded_missing_returns_false(self) -> None:
        store = JobStoreLogic()
        ok = store.set_succeeded("job_nonexistent", {})
        assert ok is False


class TestJobStoreSetFailed:
    def test_failed(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("flux", None, "alpha-key-12345678")
        ok = store.set_failed(job_id, "oom", "CUDA out of memory")
        assert ok is True
        result = store.get(job_id, "alpha-key-12345678")
        assert result["status"] == "failed"
        assert result["error"]["code"] == "oom"
        assert "CUDA" in result["error"]["message"]


class TestJobStoreCancel:
    def test_cancel_sets_canceled(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("flux", None, "alpha-key-12345678")
        result = store.cancel(job_id, "alpha-key-12345678")
        assert result["status"] == "canceled"

    def test_cancel_wrong_key_returns_none(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        result = store.cancel(job_id, "beta-key-12345678")
        assert result is None

    def test_cancel_missing_returns_none(self) -> None:
        store = JobStoreLogic()
        result = store.cancel("job_nonexistent", "alpha-key-12345678")
        assert result is None

    def test_cancel_terminal_succeeded_is_noop(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        store.set_succeeded(job_id, {"id": "img_1"})
        result = store.cancel(job_id, "alpha-key-12345678")
        assert result is not None
        assert result["status"] == "succeeded"

    def test_cancel_terminal_failed_is_noop(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        store.set_failed(job_id, "error", "boom")
        result = store.cancel(job_id, "alpha-key-12345678")
        assert result is not None
        assert result["status"] == "failed"


class TestJobStoreList:
    def test_list_scoped_to_key(self) -> None:
        store = JobStoreLogic()
        store.submit("sd", None, "alpha-key-12345678")
        store.submit("flux", None, "beta-key-12345678")
        result_a = store.list_jobs(api_key="alpha-key-12345678")
        assert len(result_a["jobs"]) == 1
        assert result_a["jobs"][0]["model"] == "sd"
        result_b = store.list_jobs(api_key="beta-key-12345678")
        assert len(result_b["jobs"]) == 1
        assert result_b["jobs"][0]["model"] == "flux"

    def test_list_pagination(self) -> None:
        store = JobStoreLogic()
        for _ in range(5):
            store.submit("sd", None, "alpha-key-12345678")
        page1 = store.list_jobs(api_key="alpha-key-12345678", page_size=2)
        assert len(page1["jobs"]) == 2
        assert page1["next_page_token"] is not None
        page2 = store.list_jobs(
            api_key="alpha-key-12345678",
            page_size=2,
            page_token=page1["next_page_token"],
        )
        assert len(page2["jobs"]) == 2
        assert page2["next_page_token"] is not None
        page3 = store.list_jobs(
            api_key="alpha-key-12345678",
            page_size=2,
            page_token=page2["next_page_token"],
        )
        assert len(page3["jobs"]) == 1
        assert page3["next_page_token"] is None

    def test_list_model_filter(self) -> None:
        store = JobStoreLogic()
        store.submit("sd", None, "alpha-key-12345678")
        store.submit("flux", None, "alpha-key-12345678")
        result = store.list_jobs(api_key="alpha-key-12345678", model_filter="flux")
        assert len(result["jobs"]) == 1
        assert result["jobs"][0]["model"] == "flux"

    def test_list_status_filter(self) -> None:
        store = JobStoreLogic()
        job_id = store.submit("sd", None, "alpha-key-12345678")
        store.set_succeeded(job_id, {"id": "img_1"})
        result = store.list_jobs(api_key="alpha-key-12345678", status_filter="succeeded")
        assert len(result["jobs"]) == 1
        assert result["jobs"][0]["status"] == "succeeded"


class TestJobStoreReap:
    def test_reap_expired(self) -> None:
        store = JobStoreLogic(ttl_s=1)
        job_id = store.submit("sd", None, "alpha-key-12345678")
        store.set_succeeded(job_id, {"id": "img_1"})
        time.sleep(2)
        count = store.reap_expired()
        assert count == 1
        result = store.get(job_id, "alpha-key-12345678")
        assert result is None

    def test_reap_keeps_active(self) -> None:
        store = JobStoreLogic(ttl_s=3600)
        job_id = store.submit("sd", None, "alpha-key-12345678")
        store.set_succeeded(job_id, {"id": "img_1"})
        count = store.reap_expired()
        assert count == 0
        result = store.get(job_id, "alpha-key-12345678")
        assert result is not None


class TestRecToResponse:
    def test_processing_returns_correct_fields(self) -> None:
        rec = JobRecord(
            job_id="job_abc123",
            api_key_prefix="alpha-ke",
            created=1700000000,
            model="flux",
            user="test_user",
        )
        resp = _rec_to_response(rec)
        assert resp.job_id == "job_abc123"
        assert resp.status == "processing"
        assert resp.model == "flux"
        assert resp.user == "test_user"
        assert resp.result is None
        assert resp.error is None

    def test_succeeded_with_result(self) -> None:
        rec = JobRecord(
            job_id="job_abc123",
            api_key_prefix="alpha-ke",
            created=1700000000,
            model="sd",
            status="succeeded",
            result={"id": "img_abc", "model": "sd"},
            terminal_at=1700000010,
        )
        resp = _rec_to_response(rec)
        assert resp.status == "succeeded"
        assert resp.result["id"] == "img_abc"
        assert resp.updated == 1700000010


# ---------------------------------------------------------------------------
# Ray integration test (single cluster session)
# ---------------------------------------------------------------------------

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module")
def _ray_cluster():
    import os

    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(ignore_reinit_error=True, namespace="sentimentizer")
    yield
    try:
        actor = ray.get_actor("diffusion_job_store")
        ray.kill(actor)
    except ValueError:
        pass
    ray.shutdown()


class TestJobStoreCrossReplica:
    """Verify cross-replica visibility via Ray named detached actor."""

    def test_visibility_across_actor_handles(self, _ray_cluster) -> None:
        from sentimentizer.diffusion.job_store import JobStore

        try:
            old = ray.get_actor("diffusion_job_store")
            ray.kill(old)
        except ValueError:
            pass

        store = JobStore.options(
            name="diffusion_job_store",
            lifetime="detached",
            get_if_exists=True,
        ).remote(ttl_s=3600)

        job_id = ray.get(store.submit.remote("sd", None, "alpha-key-12345678"))
        ray.get(store.set_succeeded.remote(job_id, {"id": "img_1"}))

        store2 = ray.get_actor("diffusion_job_store")
        result = ray.get(store2.get.remote(job_id, "alpha-key-12345678"))
        assert result is not None
        assert result["status"] == "succeeded"
