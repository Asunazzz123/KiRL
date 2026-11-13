"""GSM8K dataset implementation."""

import json
from pathlib import Path
from ..datasets import BaseDataset


def load_gsm8k(
    path: str,
    *,
    prompt_template_path: str | None = None,
) -> BaseDataset:
    """Load a GSM8K JSONL file into a BaseDataset.

    Each line must contain ``question`` and ``answer`` fields; the answer must
    include a ``####`` separator marking the final numeric answer.

    If *prompt_template_path* is given, its content is used as a
    ``str.format``-style template with a single ``{question}`` placeholder.
    """
    template: str | None = None
    if prompt_template_path is not None:
        template = Path(prompt_template_path).read_text(encoding="utf-8")

    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            raw = json.loads(line)
            _, separator, answer = raw["answer"].rpartition("####")
            if not separator:
                raise ValueError("GSM8K answer 缺少 #### 最终答案标记")

            prompt = template.format(question=raw["question"]) if template else raw["question"]
            samples.append({
                "sample_id": str(len(samples)),
                "prompt": prompt,
                "solution": answer.strip(),
            })

    return BaseDataset(samples)