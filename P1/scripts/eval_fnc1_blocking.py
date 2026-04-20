from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.blocking import BlockingConfig, MultiStageBlocker, SentenceTransformersEmbedder, TransformersMeanPoolEmbedder
from p1.data.fnc1 import read_jsonl, sample_to_claim_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate blocking retention on normalized FNC-1 data.")
    parser.add_argument("--input", default="data/processed/fnc1_train.jsonl", help="Path to normalized FNC-1 JSONL")
    parser.add_argument("--limit", type=int, default=1000, help="Number of samples to inspect")
    parser.add_argument(
        "--body-mode",
        choices=["full", "best_sentence", "top2_span", "top3_span"],
        default="top2_span",
        help="How to build the body-side claim for evaluation",
    )
    parser.add_argument("--min-lexical-similarity", type=float, default=0.1, help="Blocking lexical threshold")
    parser.add_argument(
        "--disable-lexical",
        action="store_true",
        help="Disable the lexical stage so Phase 2 can be evaluated closer to the original plan design",
    )
    parser.add_argument("--min-entity-overlap", type=int, default=1, help="Minimum shared entities for entity stage")
    parser.add_argument("--min-embedding-similarity", type=float, default=None, help="Embedding threshold for stage 2")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformers model name or local directory for embedding-based blocking",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "transformers-mean-pool"],
        default="sentence-transformers",
        help="Embedding implementation used for the Phase 2 blocking funnel",
    )
    parser.add_argument(
        "--use-embedding",
        action="store_true",
        help="Enable the embedding similarity funnel required by the plan document",
    )
    parser.add_argument(
        "--sweep",
        default="",
        help="Comma-separated lexical thresholds to compare, for example: 0.05,0.1,0.15,0.2",
    )
    parser.add_argument(
        "--embedding-sweep",
        default="",
        help="Comma-separated embedding thresholds to compare, for example: 0.5,0.6,0.7",
    )
    parser.add_argument("--allow-empty-entities", action="store_true", help="Allow pairs without shared entities")
    parser.add_argument(
        "--combine-mode",
        choices=["cascade", "union"],
        default="cascade",
        help="How to combine entity and embedding stages",
    )
    parser.add_argument("--preview", type=int, default=5, help="Number of dropped positive samples to preview")
    parser.add_argument(
        "--entity-backend",
        choices=["auto", "regex", "spacy"],
        default="auto",
        help="Entity extraction backend used to build claim pairs",
    )
    parser.add_argument(
        "--stance-filter",
        default="",
        help="Optional comma-separated FNC-1 stance labels to keep, for example: agree,disagree",
    )
    parser.add_argument(
        "--nli-label-filter",
        default="",
        help="Optional comma-separated NLI labels to keep, for example: entailment,contradiction",
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def parse_filter(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    stance_filter = parse_filter(args.stance_filter)
    nli_label_filter = parse_filter(args.nli_label_filter)
    if stance_filter:
        records = [record for record in records if record.get("stance_label") in stance_filter]
    if nli_label_filter:
        records = [record for record in records if record.get("nli_label") in nli_label_filter]
    records = records[: args.limit]
    thresholds = [args.min_lexical_similarity]
    if args.sweep.strip():
        thresholds = [float(item.strip()) for item in args.sweep.split(",") if item.strip()]
    embedding_thresholds = [args.min_embedding_similarity]
    if args.embedding_sweep.strip():
        embedding_thresholds = [float(item.strip()) for item in args.embedding_sweep.split(",") if item.strip()]

    embedder = None
    if args.use_embedding:
        if args.embedding_backend == "sentence-transformers":
            embedder = SentenceTransformersEmbedder(model_name=args.embedding_model)
        else:
            embedder = TransformersMeanPoolEmbedder(model_name=args.embedding_model)

    for threshold in thresholds:
        for embedding_threshold in embedding_thresholds:
            run_single_threshold(
                records=records,
                input_path=args.input,
                body_mode=args.body_mode,
                lexical_threshold=None if args.disable_lexical else threshold,
                min_entity_overlap=args.min_entity_overlap,
                embedding_threshold=embedding_threshold,
                allow_empty_entities=args.allow_empty_entities,
                combine_mode=args.combine_mode,
                preview=args.preview,
                embedder=embedder,
                entity_backend=args.entity_backend,
                stance_filter=stance_filter,
                nli_label_filter=nli_label_filter,
            )


def run_single_threshold(
    records: list[dict[str, str]],
    input_path: str,
    body_mode: str,
    lexical_threshold: float | None,
    min_entity_overlap: int,
    embedding_threshold: float | None,
    allow_empty_entities: bool,
    combine_mode: str,
    preview: int,
    embedder: SentenceTransformersEmbedder | None,
    entity_backend: str,
    stance_filter: set[str],
    nli_label_filter: set[str],
) -> None:
    blocker = MultiStageBlocker(
        config=BlockingConfig(
            min_entity_overlap=min_entity_overlap,
            min_lexical_similarity=lexical_threshold,
            min_embedding_similarity=embedding_threshold,
            allow_empty_entities=allow_empty_entities,
            combine_mode=combine_mode,
        ),
        embedder=embedder,
    )

    total = 0
    kept = 0
    gold_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    neutral_stance_counts: Counter[str] = Counter()
    kept_neutral_stance_counts: Counter[str] = Counter()
    dropped_positive_samples: list[dict[str, str | float]] = []

    for sample in records:
        total += 1
        gold = sample["nli_label"]
        gold_counts.update([gold])
        pair = sample_to_claim_pair(sample, body_mode=body_mode, entity_backend=entity_backend)
        built = blocker._build_pair(pair.claim_a, pair.claim_b)
        if gold == "neutral":
            neutral_stance_counts.update([sample["stance_label"]])

        if built is not None:
            kept += 1
            kept_counts.update([gold])
            if gold == "neutral":
                kept_neutral_stance_counts.update([sample["stance_label"]])
        elif gold != "neutral" and len(dropped_positive_samples) < preview:
            dropped_positive_samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "gold": gold,
                    "stance_label": sample["stance_label"],
                    "headline": sample["headline"][:180],
                    "body_preview": pair.claim_b.text[:180],
                }
            )

    print(f"Input: {input_path}")
    print(f"Evaluated samples: {total}")
    print(f"Body mode: {body_mode}")
    print(f"Blocking combine mode: {combine_mode}")
    print(f"Entity backend: {entity_backend}")
    print(f"Blocking min entity overlap: {min_entity_overlap}")
    print(f"Blocking lexical threshold: {lexical_threshold}")
    print(f"Blocking embedding threshold: {embedding_threshold}")
    print(f"Embedding enabled: {embedder is not None}")
    print(f"Allow empty entities: {allow_empty_entities}")
    print(f"Stance filter: {sorted(stance_filter) if stance_filter else 'none'}")
    print(f"NLI label filter: {sorted(nli_label_filter) if nli_label_filter else 'none'}")
    print(f"Overall retention: {safe_ratio(kept, total):.4f} ({kept}/{total})")

    print("\nPer-label retention:")
    for label in sorted(gold_counts):
        retained = kept_counts[label]
        total_label = gold_counts[label]
        print(f"  {label}: {safe_ratio(retained, total_label):.4f} ({retained}/{total_label})")

    if neutral_stance_counts:
        print("\nNeutral breakdown:")
        for stance in sorted(neutral_stance_counts):
            retained = kept_neutral_stance_counts[stance]
            total_stance = neutral_stance_counts[stance]
            print(f"  {stance}: {safe_ratio(retained, total_stance):.4f} ({retained}/{total_stance})")

    if dropped_positive_samples:
        print("\nDropped non-neutral samples:")
        for sample in dropped_positive_samples:
            print(f"\n- sample_id: {sample['sample_id']}")
            print(f"  gold={sample['gold']} stance={sample['stance_label']}")
            print(f"  headline={sample['headline']}")
            print(f"  body_preview={sample['body_preview']}")
    print()


if __name__ == "__main__":
    main()
