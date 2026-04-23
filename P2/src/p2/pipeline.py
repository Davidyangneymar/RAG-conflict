"""
P2 end-to-end pipelines.

Two entry points:
  - run_p2_pipeline_from_path: stance only, returns P2Output.
  - run_full_p2_pipeline_from_records / _from_path: stance + conflict
    typing, returns ConflictTypedOutput ready for downstream modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .p1_adapter import InputRecord, load_p1_payload, parse_p1_payload
from .contracts import P2Output
from .stance import PairStanceRunner
from .conflict_typing import type_sample, ConflictTypedOutput
from .prompt_strategy import AnswerPlan, build_answer_plans


def run_p2_pipeline_from_path(
    payload_path: Union[str, Path],
    model_output_dir: Optional[str | Path] = None,
    preferred_device: Optional[str] = None,
) -> P2Output:
    records = load_p1_payload(payload_path)
    return run_p2_pipeline_from_records(
        records,
        model_output_dir=model_output_dir,
        preferred_device=preferred_device,
    )


def run_p2_pipeline_from_records(
    records: List[InputRecord],
    model_output_dir: Optional[str | Path] = None,
    preferred_device: Optional[str] = None,
) -> P2Output:
    runner = PairStanceRunner(
        model_output_dir=model_output_dir,
        preferred_device=preferred_device,
    )
    return P2Output(samples=runner.run_records(records))


def run_full_p2_pipeline_from_records(
    records: List[InputRecord],
    model_output_dir: Optional[str | Path] = None,
    preferred_device: Optional[str] = None,
    gold_verdicts: Optional[List[Optional[str]]] = None,
) -> ConflictTypedOutput:
    """
    Stance analysis + conflict typing in one call.

    `gold_verdicts`, if passed, must be aligned to `records` and is copied
    into TypedSample.gold_verdict for downstream scoring. Use None for
    samples without a gold label.
    """
    stanced = run_p2_pipeline_from_records(
        records,
        model_output_dir=model_output_dir,
        preferred_device=preferred_device,
    )
    typed_samples = []
    for i, (stanced_sample, record) in enumerate(zip(stanced.samples, records)):
        gold = None
        if gold_verdicts is not None and i < len(gold_verdicts):
            gold = gold_verdicts[i]
        typed_samples.append(
            type_sample(stanced_sample, input_record=record, gold_verdict=gold)
        )
    return ConflictTypedOutput(samples=typed_samples)


def run_full_p2_pipeline_from_path(
    payload_path: Union[str, Path],
    model_output_dir: Optional[str | Path] = None,
    preferred_device: Optional[str] = None,
) -> ConflictTypedOutput:
    records = load_p1_payload(payload_path)
    return run_full_p2_pipeline_from_records(
        records,
        model_output_dir=model_output_dir,
        preferred_device=preferred_device,
    )


def run_full_p2_with_answer_plans_from_records(
    records: List[InputRecord],
    model_output_dir: Optional[str | Path] = None,
    preferred_device: Optional[str] = None,
    gold_verdicts: Optional[List[Optional[str]]] = None,
) -> Tuple[ConflictTypedOutput, List[AnswerPlan]]:
    """
    Stance + conflict typing + P6-facing answer-plan prompts in one call.

    The answer-plan layer only consumes the typed output and original input
    records for claim lookup / traceability.
    """
    typed = run_full_p2_pipeline_from_records(
        records,
        model_output_dir=model_output_dir,
        preferred_device=preferred_device,
        gold_verdicts=gold_verdicts,
    )
    plans = build_answer_plans(typed.samples, records)
    return typed, plans


def run_full_p2_with_answer_plans_from_path(
    payload_path: Union[str, Path],
    model_output_dir: Optional[str | Path] = None,
    preferred_device: Optional[str] = None,
) -> Tuple[ConflictTypedOutput, List[AnswerPlan]]:
    records = load_p1_payload(payload_path)
    return run_full_p2_with_answer_plans_from_records(
        records,
        model_output_dir=model_output_dir,
        preferred_device=preferred_device,
    )
