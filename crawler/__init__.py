"""Crawler app contracts for source canonicalization, frontier, and fetcher policies."""

from .contracts import (
    DEFAULT_TRACKING_PARAMETERS,
    DEFAULT_TRACKING_PREFIXES,
    Decision,
    DEFAULT_NATIVE_ID_GETTER,
    HTTP_URL_CANONICALIZER,
    NormalizedSource,
    ProviderRules,
    deregister_provider,
    get_provider_rules,
    normalize_source,
    register_provider,
)
from .frontier import (
    CrawlSignals,
    FrontierAction,
    FrontierDecision,
    HostPolitenessPolicy,
    HostVisitState,
    RecrawlFrequency,
    RobotsPolicy,
    SourceDescriptor,
    decide_frontier_action,
)
from .fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
    evaluate_fetch_response,
)
from common.guardrails import FetcherLimits
from .errors import CrawlerError, ErrorClass
from .http_fetcher import FetchRetryPolicy, HttpFetcher, HttpFetcherConfig
from .policies import (
    HostPolicy,
    PolicyRegistry,
    ProviderPolicy,
    build_policy_registry,
)

__all__ = [
    "DEFAULT_TRACKING_PARAMETERS",
    "DEFAULT_TRACKING_PREFIXES",
    "DEFAULT_NATIVE_ID_GETTER",
    "Decision",
    "HTTP_URL_CANONICALIZER",
    "NormalizedSource",
    "ProviderRules",
    "deregister_provider",
    "get_provider_rules",
    "normalize_source",
    "register_provider",
    "CrawlSignals",
    "FrontierAction",
    "FrontierDecision",
    "HostPolitenessPolicy",
    "HostVisitState",
    "RecrawlFrequency",
    "RobotsPolicy",
    "SourceDescriptor",
    "decide_frontier_action",
    "FetchMetadata",
    "FetchRequest",
    "FetchResult",
    "FetchStatus",
    "FetchTelemetry",
    "FetcherLimits",
    "PolitenessContext",
    "evaluate_fetch_response",
    "CrawlerError",
    "ErrorClass",
    "FetchRetryPolicy",
    "HttpFetcher",
    "HttpFetcherConfig",
    "HostPolicy",
    "PolicyRegistry",
    "ProviderPolicy",
    "build_policy_registry",
]
