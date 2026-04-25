from __future__ import annotations

import json
from dataclasses import asdict
from typing import List

from .judge_types import JudgeExample


class JudgeReplayBuffer:
    def __init__(self):
        self.examples: List[JudgeExample] = []

    def add(self, example: JudgeExample) -> None:
        self.examples.append(example)

    def size(self) -> int:
        return len(self.examples)

    def clear(self) -> None:
        self.examples.clear()

    def flush_jsonl(self, path: str) -> None:
        if not self.examples:
            return
        with open(path, "a", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=True) + "\n")
        self.clear()
