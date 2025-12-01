from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from strategic_ttc.interfaces.benchmark import Question


def resolve_device(device: Optional[str]) -> torch.device:
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _to_torch_dtype(dtype_str: Optional[str]):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: None,
    }
    return mapping.get(dtype_str, None)


@dataclass
class ArmoRewardModel:
    model_id: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    device: str = "auto"
    dtype: Optional[str] = "bfloat16"
    trust_remote_code: bool = True
    local_files_only: bool = False

    def __post_init__(self):
        torch_dtype = _to_torch_dtype(self.dtype)
        self._device = resolve_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self.model.to(self._device)

        self.name = f"armo_rm:{self.model_id}"

    @torch.no_grad()
    def score(self, question: Question, answer: str) -> float:
        messages = [
            {"role": "user", "content": question.prompt},
            {"role": "assistant", "content": answer},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).to(self._device)

        output = self.model(input_ids)
        pref_score = output.score.squeeze().float().item()
        return pref_score
