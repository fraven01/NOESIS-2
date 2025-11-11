import json
from datetime import datetime, timedelta, timezone

from django.test import Client, TestCase, override_settings
from django.urls import reverse

from theme.dev_hitl_store import store


class DevHitlViewTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_page_returns_404_when_feature_disabled(self) -> None:
        response = self.client.get(reverse("dev-hitl"))
        self.assertEqual(response.status_code, 404)

    @override_settings(DEV_FEATURE_HITL_UI=True)
    def test_page_renders_initial_payload(self) -> None:
        response = self.client.get(reverse("dev-hitl"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("hitl-initial-data", response.content.decode())

    @override_settings(DEV_FEATURE_HITL_UI=True)
    def test_get_run_requires_dev_header(self) -> None:
        url = reverse("dev-hitl-run-api", kwargs={"run_id": store.default_run_id()})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)

        authed = self.client.get(url, HTTP__DEV_ONLY_="true")
        self.assertEqual(authed.status_code, 200)
        payload = authed.json()
        self.assertIn("top_k", payload)
        self.assertIn("coverage_delta", payload)

    @override_settings(DEV_FEATURE_HITL_UI=True)
    def test_post_submission_validates_payload(self) -> None:
        url = reverse("dev-hitl-approve")
        body = {
            "run_id": store.default_run_id(),
            "approved_ids": [],
            "rejected_ids": [],
            "custom_urls": ["ftp://invalid.local"],
        }
        response = self.client.post(
            url,
            data=json.dumps(body),
            content_type="application/json",
            HTTP__DEV_ONLY_="true",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_custom_urls")

    @override_settings(DEV_FEATURE_HITL_UI=True)
    def test_post_submission_is_idempotent(self) -> None:
        url = reverse("dev-hitl-approve")
        run_id = store.default_run_id()
        body = {
            "run_id": run_id,
            "approved_ids": [f"{run_id}-cand-1"],
            "rejected_ids": [],
            "custom_urls": [],
        }
        first = self.client.post(
            url,
            data=json.dumps(body),
            content_type="application/json",
            HTTP__DEV_ONLY_="true",
        )
        second = self.client.post(
            url,
            data=json.dumps(body),
            content_type="application/json",
            HTTP__DEV_ONLY_="true",
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())

    @override_settings(DEV_FEATURE_HITL_UI=True)
    def test_progress_stream_emits_events(self) -> None:
        run_id = store.default_run_id()
        url = reverse("dev-hitl-approve")
        body = {
            "run_id": run_id,
            "approved_ids": [f"{run_id}-cand-2"],
            "rejected_ids": [],
            "custom_urls": [],
        }
        self.client.post(
            url,
            data=json.dumps(body),
            content_type="application/json",
            HTTP__DEV_ONLY_="true",
        )

        stream_url = (
            reverse("dev-hitl-progress", kwargs={"run_id": run_id}) + "?dev_token=true"
        )
        response = self.client.get(stream_url, HTTP__DEV_ONLY_="true")
        self.assertEqual(response.status_code, 200)

        events: list[str] = []
        for chunk in response.streaming_content:
            decoded = chunk.decode() if isinstance(chunk, bytes) else chunk
            events.append(decoded)
            if len(events) >= 3:
                break
        joined = "".join(events)
        self.assertIn("event: ingestion_update", joined)

    @override_settings(DEV_FEATURE_HITL_UI=True)
    def test_deadline_auto_approval_triggers(self) -> None:
        run_id = "deadline-test"
        run = store.get(run_id)
        with run._lock:  # type: ignore[attr-defined]
            run.payload["meta"]["deadline_utc"] = (
                (datetime.now(timezone.utc) - timedelta(minutes=1))
                .isoformat()
                .replace("+00:00", "Z")
            )
        stream_url = (
            reverse("dev-hitl-progress", kwargs={"run_id": run_id}) + "?dev_token=true"
        )
        response = self.client.get(stream_url, HTTP__DEV_ONLY_="true")
        events = []
        for chunk in response.streaming_content:
            decoded = chunk.decode() if isinstance(chunk, bytes) else chunk
            events.append(decoded)
            if '"auto_approved": true' in decoded or len(events) >= 6:
                break
        self.assertTrue(any('"auto_approved": true' in event for event in events))

        payload = run.serialize()
        self.assertTrue(payload["meta"]["auto_approved"])
