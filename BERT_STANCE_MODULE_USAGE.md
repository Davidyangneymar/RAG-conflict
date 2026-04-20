# BERT Stance Module Usage

This document describes how to use the frozen BERT stance inference module without retraining.

## 1. Files

- `fnc1_bert_stance_module.py`: formal reusable module for loading model artifacts and running inference.
- `smoke_test_bert_stance_module.py`: smoke test script for quick verification.

## 2. Required Artifacts

The module expects an already trained output directory that contains:

- `model/runtime_config.json`
- `model/hf_model/`
- `model/hf_tokenizer/`

Default path:

- `outputs/fnc1_bert_upgrade_full`

## 3. Single Inference

```python
from fnc1_bert_stance_module import predict_stance

result = predict_stance(
    claim="Scientists report rising sea levels",
    evidence_text="A new climate report says global sea levels increased over the last decade.",
    output_dir="outputs/fnc1_bert_upgrade_full",
)
print(result)
```

Output fields include:

- `pred_label_4way`: one of `agree/disagree/discuss/unrelated`
- `pred_label_3way_a`: one of `support/oppose/neutral` or `None`
- `pred_label_3way_a_with_filter`: mapped label or `filtered` for unrelated
- `is_filtered_3way_a`: boolean filter flag for scheme A
- `decision_score`: confidence score of predicted 4-way class
- `claim`, `body_first3sent`, `mapping_scheme`

## 4. Batch Inference

```python
from fnc1_bert_stance_module import predict_stance_batch

claims = [
    "Scientists report rising sea levels",
    "Tech company denies bankruptcy rumors",
]
evidence_texts = [
    "A new climate report says global sea levels increased over the last decade.",
    "Executives said there are no plans to file for bankruptcy.",
]

results = predict_stance_batch(
    claims=claims,
    evidence_texts=evidence_texts,
    output_dir="outputs/fnc1_bert_upgrade_full",
)
print(results)
```

## 5. Direct Class-Based Usage

```python
from fnc1_bert_stance_module import BertStancePredictor

predictor = BertStancePredictor.from_output_dir("outputs/fnc1_bert_upgrade_full")
row = predictor.predict_stance("claim text", "evidence text")
rows = predictor.predict_stance_batch(["c1", "c2"], ["e1", "e2"])
```

## 6. Smoke Test

Run:

```bash
python smoke_test_bert_stance_module.py --output_dir outputs/fnc1_bert_upgrade_full
```

Optional custom CSV output:

```bash
python smoke_test_bert_stance_module.py --output_dir outputs/fnc1_bert_upgrade_full --save_csv outputs/fnc1_bert_upgrade_full/smoke_test_predictions.csv
```

## 7. Integration Hint for Conflict Typing

Recommended hand-off logic:

1. Run stance inference and read `pred_label_3way_a_with_filter`.
2. If value is `filtered` (or `is_filtered_3way_a=True`), skip conflict typing.
3. Otherwise, pass `claim`, `body_first3sent`, and 3-way stance label (`support/oppose/neutral`) to the conflict typing module.
