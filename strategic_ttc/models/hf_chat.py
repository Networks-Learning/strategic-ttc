from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def resolve_device(device: Optional[str]) -> torch.device:
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class HFChatModel:
    model_id: str
    device: str = "auto"
    dtype: Optional[str] = "bfloat16"
    max_tokens: int = 512
    temperature: float = 0.0

    def __post_init__(self):
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            None: None
        }.get(self.dtype)

        self.device = resolve_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        ).to(self.device)

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
