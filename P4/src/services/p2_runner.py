# services/p2_runner.py
from pathlib import Path
from p2.p1_adapter import parse_p1_payload
from p2.pipeline import run_full_p2_pipeline_from_records

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
P2_MODEL_DIR = PROJECT_ROOT / "P2" / "outputs" / "fnc1_bert_upgrade_full"

def run_p2(p1_payload_dict: dict):
    records = parse_p1_payload(p1_payload_dict)
    input_record = records[0]
    typed_output = run_full_p2_pipeline_from_records(
        records,
        model_output_dir=str(P2_MODEL_DIR)
    )
    return typed_output, input_record

def run_p2_with_p1(p1_payload_dict: dict):
    """Returns (typed_output, input_record, p1_payload_dict) for benchmark usage."""
    records = parse_p1_payload(p1_payload_dict)
    input_record = records[0]
    typed_output = run_full_p2_pipeline_from_records(
        records,
        model_output_dir=str(P2_MODEL_DIR)
    )
    return typed_output, input_record, p1_payload_dict