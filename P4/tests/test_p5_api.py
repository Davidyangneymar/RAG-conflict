import requests
import json
from pathlib import Path

url = "http://localhost:8000/api/v1/benchmark/export"
test_data = {
    "claims": [
        {"id": "1", "label": "SUPPORTS", "claim": "Vitamin C prevents colds."},
        {"id": "2", "label": "REFUTES", "claim": "The capital of France is Berlin."}
    ]
}
response = requests.post(url, json=test_data)
response.raise_for_status()

# 输出文件放在脚本所在目录（即 P4/tests/）
output_file = Path(__file__).parent / "p5_output_from_python.jsonl"
with open(output_file, "w") as f:
    f.write(response.text)

print(f"Saved to {output_file}")
for line in response.text.strip().split("\n"):
    row = json.loads(line)
    print(f"{row['sample_id']}: gold={row['gold_label']} pred={row['predicted_label']}")