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
from p1.data.fnc1 import (
    NEGATION_HINTS,
    NUMBER_RE,
    TOKEN_RE,
    rank_body_sentences,
    read_jsonl,
    sample_to_claim_pair,
)
from p1.nli import build_nli_model

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "for", "of", "to", "that", "this", "with", "by",
    "is", "was", "are", "were", "as", "at", "from",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice FNC-1 failure cases into coarse pattern buckets.")
    parser.add_argument("--input", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--mode", choices=["nli", "blocking", "retained_neutral"], default="blocking")
    parser.add_argument("--body-mode", choices=["full", "best_sentence", "top2_span", "top3_span"], default="top2_span")
    parser.add_argument("--preview", type=int, default=8)
    parser.add_argument("--model", choices=["heuristic", "hf"], default="heuristic")
    parser.add_argument(
        "--hf-model-name",
        default="cross-encoder/nli-deberta-v3-large",
        help="HuggingFace model name or local model directory used when --model hf",
    )
    parser.add_argument("--min-entity-overlap", type=int, default=1)
    parser.add_argument("--min-lexical-similarity", type=float, default=0.1)
    parser.add_argument("--disable-lexical", action="store_true")
    parser.add_argument("--min-embedding-similarity", type=float, default=None)
    parser.add_argument("--allow-empty-entities", action="store_true")
    parser.add_argument("--combine-mode", choices=["cascade", "union"], default="cascade")
    parser.add_argument("--use-embedding", action="store_true")
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "transformers-mean-pool"],
        default="transformers-mean-pool",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--entity-backend",
        choices=["auto", "regex", "spacy"],
        default="auto",
    )
    return parser.parse_args()


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS}


def categorize(sample: dict[str, str], selected_sentence: str) -> list[str]:
    headline = sample["headline"]
    headline_tokens = tokenize(headline)
    sentence_tokens = tokenize(selected_sentence)
    overlap = headline_tokens & sentence_tokens
    categories: list[str] = []

    if len(headline_tokens) <= 3:
        categories.append("short_headline")
    if not overlap:
        categories.append("zero_token_overlap")
    elif len(overlap) <= 2:
        categories.append("low_token_overlap")

    headline_numbers = set(NUMBER_RE.findall(headline))
    sentence_numbers = set(NUMBER_RE.findall(selected_sentence))
    if headline_numbers and not (headline_numbers & sentence_numbers):
        categories.append("number_mismatch")

    headline_neg = any(token in headline_tokens for token in NEGATION_HINTS)
    sentence_neg = any(token in sentence_tokens for token in NEGATION_HINTS)
    if headline_neg != sentence_neg:
        categories.append("negation_mismatch")

    if "'" in headline or '"' in headline:
        categories.append("quoted_headline")

    if ":" in headline:
        categories.append("colon_headline")

    if not categories:
        categories.append("other")
    return categories


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)[: args.limit]
    embedder = None
    if args.use_embedding:
        if args.embedding_backend == "sentence-transformers":
            embedder = SentenceTransformersEmbedder(model_name=args.embedding_model)
        else:
            embedder = TransformersMeanPoolEmbedder(model_name=args.embedding_model)
    blocker = MultiStageBlocker(
        config=BlockingConfig(
            min_entity_overlap=args.min_entity_overlap,
            min_lexical_similarity=None if args.disable_lexical else args.min_lexical_similarity,
            min_embedding_similarity=args.min_embedding_similarity,
            allow_empty_entities=args.allow_empty_entities,
            combine_mode=args.combine_mode,
        ),
        embedder=embedder,
    )
    nli_model = build_nli_model(kind=args.model, model_name=args.hf_model_name)

    bucket_counts: Counter[str] = Counter()
    previews: list[dict[str, str]] = []

    for sample in records:
        if args.mode in {"blocking", "nli"} and sample["nli_label"] == "neutral":
            continue

        ranked = rank_body_sentences(sample["headline"], sample["body"], top_k=1)
        selected_sentence = ranked[0]["sentence"] if ranked else sample["body"]
        pair = sample_to_claim_pair(sample, body_mode=args.body_mode, entity_backend=args.entity_backend)
        built_pair = blocker._build_pair(pair.claim_a, pair.claim_b)
        pair.lexical_similarity = built_pair.lexical_similarity if built_pair else 0.0

        failed = False
        if args.mode == "blocking":
            failed = built_pair is None
        elif args.mode == "retained_neutral":
            failed = sample["nli_label"] == "neutral" and built_pair is not None
        else:
            result = nli_model.predict(pair)
            failed = result.label.value != sample["nli_label"]

        if not failed:
            continue

        categories = categorize(sample, selected_sentence)
        bucket_counts.update(categories)
        if len(previews) < args.preview:
            previews.append(
                {
                    "sample_id": sample["sample_id"],
                    "gold": sample["nli_label"],
                    "stance": sample["stance_label"],
                    "categories": ", ".join(categories),
                    "headline": sample["headline"][:180],
                    "selected_sentence": selected_sentence[:220],
                }
            )

    print(f"Input: {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Body mode: {args.body_mode}")
    print(f"Model: {args.model}")
    print(f"Combine mode: {args.combine_mode}")
    print(f"Entity backend: {args.entity_backend}")
    print(f"Lexical threshold: {None if args.disable_lexical else args.min_lexical_similarity}")
    print(f"Embedding threshold: {args.min_embedding_similarity}")
    print(f"Embedding enabled: {args.use_embedding}")
    print("\nFailure buckets:")
    for category, count in bucket_counts.most_common():
        print(f"  {category}: {count}")

    if previews:
        print("\nFailure previews:")
        for item in previews:
            print(f"\n- sample_id: {item['sample_id']}")
            print(f"  gold={item['gold']} stance={item['stance']}")
            print(f"  categories={item['categories']}")
            print(f"  headline={item['headline']}")
            print(f"  selected_sentence={item['selected_sentence']}")


if __name__ == "__main__":
    main()
