from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from ai_core.settings import RAG


TOPK_DEFAULT = RAG.TOPK_DEFAULT
TOPK_MAX = RAG.TOPK_MAX
_ALLOWED_KEYS = {
    "alpha",
    "min_sim",
    "top_k",
    "vec_limit",
    "lex_limit",
    "trgm_limit",
    "max_candidates",
}


@dataclass(frozen=True)
class HybridParameters:
    alpha: float
    min_sim: float
    top_k: int
    vec_limit: int
    lex_limit: int
    trgm_limit: float | None
    max_candidates: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "alpha": self.alpha,
            "min_sim": self.min_sim,
            "top_k": self.top_k,
            "vec_limit": self.vec_limit,
            "lex_limit": self.lex_limit,
            "trgm_limit": self.trgm_limit,
            "max_candidates": self.max_candidates,
        }


def _ensure_mapping(value: object, *, message: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(message)


def _coerce_float(value: object, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hybrid.{field} must be a number") from exc


def _coerce_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"hybrid.{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hybrid.{field} must be an integer") from exc
    return number


def _normalise_fraction(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalise_positive_int(value: int, *, minimum: int = RAG.TOPK_MIN) -> int:
    return max(minimum, value)


def _parse_hybrid_mapping(mapping: Mapping[str, Any]) -> HybridParameters:
    unknown = set(mapping) - _ALLOWED_KEYS
    if unknown:
        detail = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown hybrid parameter(s): {detail}")

    alpha_raw = mapping.get("alpha", RAG.HYBRID_ALPHA_DEFAULT)
    min_sim_raw = mapping.get("min_sim", RAG.MIN_SIM_DEFAULT)
    top_k_raw = mapping.get("top_k", TOPK_DEFAULT)
    vec_limit_raw = mapping.get("vec_limit", 50)
    lex_limit_raw = mapping.get("lex_limit", 50)
    trgm_limit_raw = mapping.get("trgm_limit")
    max_candidates_raw = mapping.get("max_candidates")

    alpha = _normalise_fraction(_coerce_float(alpha_raw, field="alpha"))
    min_sim = _normalise_fraction(_coerce_float(min_sim_raw, field="min_sim"))
    top_k_value = _normalise_positive_int(
        _coerce_int(top_k_raw, field="top_k"), minimum=RAG.TOPK_MIN
    )
    top_k = min(TOPK_MAX, top_k_value)

    vec_limit_value = _normalise_positive_int(
        _coerce_int(vec_limit_raw, field="vec_limit")
    )
    lex_limit_value = _normalise_positive_int(
        _coerce_int(lex_limit_raw, field="lex_limit")
    )

    trgm_limit: float | None
    if trgm_limit_raw is None:
        trgm_limit = None
    else:
        trgm_limit = _normalise_fraction(
            _coerce_float(trgm_limit_raw, field="trgm_limit")
        )

    if max_candidates_raw is None:
        max_candidates_value = max(vec_limit_value, lex_limit_value)
    else:
        max_candidates_value = _normalise_positive_int(
            _coerce_int(max_candidates_raw, field="max_candidates")
        )
        max_candidates_value = max(max_candidates_value, vec_limit_value, lex_limit_value)

    max_candidates = max(max_candidates_value, top_k)

    return HybridParameters(
        alpha=alpha,
        min_sim=min_sim,
        top_k=top_k,
        vec_limit=vec_limit_value,
        lex_limit=lex_limit_value,
        trgm_limit=trgm_limit,
        max_candidates=max_candidates,
    )


def parse_hybrid_parameters(
    state: MutableMapping[str, Any],
    *,
    override_top_k: int | None = None,
) -> HybridParameters:
    if "hybrid" not in state:
        raise ValueError("state must include a 'hybrid' configuration")

    hybrid_raw = _ensure_mapping(state["hybrid"], message="state.hybrid must be a mapping")
    params = _parse_hybrid_mapping(hybrid_raw)

    if override_top_k is not None:
        top_k_override = _normalise_positive_int(
            int(override_top_k), minimum=RAG.TOPK_MIN
        )
        top_k = min(TOPK_MAX, top_k_override)
        max_candidates = max(params.max_candidates, top_k)
        params = HybridParameters(
            alpha=params.alpha,
            min_sim=params.min_sim,
            top_k=top_k,
            vec_limit=params.vec_limit,
            lex_limit=params.lex_limit,
            trgm_limit=params.trgm_limit,
            max_candidates=max_candidates,
        )

    state["hybrid"] = params.as_dict()
    return params


__all__ = [
    "HybridParameters",
    "TOPK_DEFAULT",
    "TOPK_MAX",
    "parse_hybrid_parameters",
]

