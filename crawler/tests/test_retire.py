from datetime import datetime, timedelta, timezone

from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from crawler.retire import (
    LifecycleState,
    RetirePolicy,
    RetireSignals,
    evaluate_lifecycle,
)


def _make_request() -> FetchRequest:
    politeness = PolitenessContext(host="example.com")
    return FetchRequest("https://example.com/doc", politeness)


def _make_fetch_result(
    status: FetchStatus,
    *,
    status_code: int,
    detail: str,
) -> FetchResult:
    request = _make_request()
    metadata = FetchMetadata(
        status_code=status_code,
        content_type="text/html",
        etag=None,
        last_modified=None,
        content_length=None,
    )
    telemetry = FetchTelemetry(latency=0.1, bytes_downloaded=0)
    return FetchResult(
        status=status,
        request=request,
        payload=None,
        metadata=metadata,
        telemetry=telemetry,
        detail=detail,
    )


def test_manual_retire_signal_wins() -> None:
    signals = RetireSignals(
        manual_state=LifecycleState.RETIRED,
        manual_reason="policy_retired",
    )

    decision = evaluate_lifecycle(fetch=None, signals=signals)

    assert decision.state is LifecycleState.RETIRED
    assert decision.reason == "policy_retired"
    assert decision.policy_events == ("manual_retire",)


def test_http_410_immediately_retires() -> None:
    fetch = _make_fetch_result(
        FetchStatus.GONE,
        status_code=410,
        detail="status_410",
    )

    decision = evaluate_lifecycle(fetch=fetch)

    assert decision.state is LifecycleState.RETIRED
    assert decision.reason == "gone_410"
    assert decision.policy_events == ("gone_410",)


def test_repeated_404_over_interval_retires() -> None:
    now = datetime.now(timezone.utc)
    signals = RetireSignals(
        consecutive_not_found=2,
        first_not_found_at=now - timedelta(days=8),
    )
    policy = RetirePolicy(
        consecutive_not_found_threshold=3,
        not_found_interval=timedelta(days=7),
    )
    fetch = _make_fetch_result(
        FetchStatus.GONE,
        status_code=404,
        detail="status_404",
    )

    decision = evaluate_lifecycle(fetch=fetch, signals=signals, policy=policy, now=now)

    assert decision.state is LifecycleState.RETIRED
    assert decision.reason == "not_found_streak:3"
    assert decision.policy_events == ("not_found_streak",)


def test_single_404_does_not_retire() -> None:
    now = datetime.now(timezone.utc)
    signals = RetireSignals(
        consecutive_not_found=0,
        first_not_found_at=None,
    )
    fetch = _make_fetch_result(
        FetchStatus.GONE,
        status_code=404,
        detail="status_404",
    )

    decision = evaluate_lifecycle(fetch=fetch, signals=signals, now=now)

    assert decision.state is LifecycleState.ACTIVE
    assert decision.reason == "not_found"
    assert decision.policy_events == ()


def test_permanent_redirect_from_fetch_retires() -> None:
    fetch = _make_fetch_result(
        FetchStatus.TEMPORARY_ERROR,
        status_code=301,
        detail="status_301",
    )

    decision = evaluate_lifecycle(fetch=fetch)

    assert decision.state is LifecycleState.RETIRED
    assert decision.reason == "permanent_redirect:301"
    assert decision.policy_events == ("permanent_redirect",)


def test_redirect_signal_without_fetch_retires() -> None:
    signals = RetireSignals(permanent_redirect_target="https://example.com/new")

    decision = evaluate_lifecycle(fetch=None, signals=signals)

    assert decision.state is LifecycleState.RETIRED
    assert decision.reason == "permanent_redirect:https://example.com/new"
    assert decision.policy_events == ("permanent_redirect",)


def test_manual_delete_signal_sets_deleted_state() -> None:
    signals = RetireSignals(
        manual_state=LifecycleState.DELETED,
        manual_reason="purged",
    )

    decision = evaluate_lifecycle(fetch=None, signals=signals)

    assert decision.state is LifecycleState.DELETED
    assert decision.reason == "purged"
    assert decision.policy_events == ("manual_delete",)
