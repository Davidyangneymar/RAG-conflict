from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time

from p1.blocking import MultiStageBlocker
from p1.claim_extraction import ClaimExtractor, SentenceClaimExtractor, build_claim_extractor
from p1.nli import HeuristicNLIModel, NLIModel
from p1.observability import elapsed_ms, log_event
from p1.schemas import ChunkInput, PipelineOutput, RetrievalInput


logger = logging.getLogger(__name__)


@dataclass
class P1Pipeline:
    extractor: ClaimExtractor = field(default_factory=SentenceClaimExtractor)
    blocker: MultiStageBlocker = field(default_factory=MultiStageBlocker)
    nli_model: NLIModel = field(default_factory=HeuristicNLIModel)

    def run(self, chunks: list[ChunkInput]) -> PipelineOutput:
        """Run claim extraction, blocking, and NLI over normalized input chunks."""
        pipeline_started_at = time.perf_counter()
        stage_started_at = pipeline_started_at
        log_event(logger, "p1.pipeline.start", chunks=len(chunks))
        claims = self.extractor.extract_many(chunks)
        now = time.perf_counter()
        log_event(
            logger,
            "p1.pipeline.claims_extracted",
            claims=len(claims),
            duration_ms=elapsed_ms(stage_started_at, now),
        )

        stage_started_at = now
        candidate_pairs = self.blocker.generate_pairs(claims)
        now = time.perf_counter()
        log_event(
            logger,
            "p1.pipeline.candidate_pairs",
            pairs=len(candidate_pairs),
            duration_ms=elapsed_ms(stage_started_at, now),
        )
        stage_started_at = now
        nli_results = self.nli_model.predict_many(candidate_pairs)
        now = time.perf_counter()
        log_event(
            logger,
            "p1.pipeline.complete",
            nli_results=len(nli_results),
            nli_duration_ms=elapsed_ms(stage_started_at, now),
            total_duration_ms=elapsed_ms(pipeline_started_at, now),
        )

        return PipelineOutput(
            claims=claims,
            candidate_pairs=candidate_pairs,
            nli_results=nli_results,
        )

    def run_retrieval_input(self, retrieval_input: RetrievalInput) -> PipelineOutput:
        """Adapt a retrieval contract payload and run the standard P1 pipeline."""
        from p1.data.retrieval import retrieval_input_to_chunk_inputs

        log_event(
            logger,
            "p1.pipeline.retrieval_input",
            sample_id=retrieval_input.sample_id,
            retrieved_chunks=len(retrieval_input.retrieved_chunks),
        )
        return self.run(retrieval_input_to_chunk_inputs(retrieval_input))


def build_pipeline(
    extractor_kind: str = "sentence",
    entity_backend: str = "auto",
    extractor_options: dict[str, object] | None = None,
    blocker: MultiStageBlocker | None = None,
    nli_model: NLIModel | None = None,
    nli_kind: str = "hf",
) -> P1Pipeline:
    """Build a P1 pipeline.

    By default we now use the HuggingFace cross-encoder NLI backend
    (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, bidirectional) — this is
    the configuration that hits macro-F1 = 0.4449 on FNC-1 N=360 vs the
    heuristic baseline's 0.3453. If the HF model can't be loaded (no torch /
    transformers / weights / network), we silently fall back to the heuristic
    model so callers still get a working pipeline. Output schema is unchanged
    either way — this is a contract-compatible default upgrade.

    Pass ``nli_kind="heuristic"`` (or your own ``nli_model``) to override.
    """
    if nli_model is None:
        if nli_kind == "heuristic":
            nli_model = HeuristicNLIModel()
        else:
            try:
                from p1.nli import build_nli_model

                from pathlib import Path as _Path

                _local = _Path("manual_models/DeBERTa-v3-base-mnli-fever-anli")
                _model = (
                    str(_local)
                    if _local.exists()
                    else "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
                )
                nli_model = build_nli_model(
                    kind=nli_kind,
                    model_name=_model,
                    bidirectional=True,
                )
            except Exception as exc:  # pragma: no cover - depends on local env
                log_event(
                    logger,
                    "p1.pipeline.nli_fallback",
                    requested_kind=nli_kind,
                    error=str(exc),
                )
                nli_model = HeuristicNLIModel()
    return P1Pipeline(
        extractor=build_claim_extractor(
            kind=extractor_kind,
            entity_backend=entity_backend,
            **(extractor_options or {}),
        ),
        blocker=blocker or MultiStageBlocker(),
        nli_model=nli_model,
    )
