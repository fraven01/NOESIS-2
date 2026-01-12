from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RagasSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
        }
        if self.ground_truth is not None:
            payload["ground_truth"] = self.ground_truth
        return payload


def load_jsonl_dataset(path: str | Path) -> list[RagasSample]:
    dataset_path = Path(path)
    samples: list[RagasSample] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(
            RagasSample(
                question=str(payload.get("question", "")).strip(),
                answer=str(payload.get("answer", "")).strip(),
                contexts=list(payload.get("contexts") or []),
                ground_truth=payload.get("ground_truth"),
            )
        )
    return samples


def run_ragas_evaluation(
    samples: Sequence[RagasSample] | Iterable[Mapping[str, Any]],
) -> Any:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ragas and datasets are required for evaluation. "
            "Install with `pip install ragas datasets`."
        ) from exc

    sample_list = list(samples)
    if sample_list and not isinstance(sample_list[0], RagasSample):
        sample_payloads = [dict(sample) for sample in sample_list]
    else:
        sample_payloads = [sample.to_dict() for sample in sample_list]

    dataset = Dataset.from_list(sample_payloads)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


__all__ = ["RagasSample", "load_jsonl_dataset", "run_ragas_evaluation"]
