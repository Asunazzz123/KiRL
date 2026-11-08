"""GSM8K dataset implementation."""

import json
from ..datasets import BaseDataset

def load_gsm8k(path: str) -> BaseDataset:
    samples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            raw = json.loads(line)
            _, separator, answer = raw["answer"].rpartition("####")
            if not separator:
                raise ValueError("GSM8K answer 缺少 #### 最终答案标记")

            samples.append({
                "sample_id": str(len(samples)),
                "prompt": raw["question"],
                "solution": answer.strip(),
            })

    return BaseDataset(samples)