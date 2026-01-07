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


def _to_torch_dtype(dtype_str: Optional[str]):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: None,
    }
    return mapping.get(dtype_str, None)


@dataclass
class GenerationResult:
    text: str
    tokens: int


@dataclass
class HFChatModel:
    model_id: str
    device: str = "auto"
    dtype: Optional[str] = None

    max_tokens: int = 512
    temperature: float = 0.0

    trust_remote_code: bool = False
    local_files_only: bool = False

    reasoning: bool = False

    def __post_init__(self):
        torch_dtype = _to_torch_dtype(self.dtype)
        self._device = resolve_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        ).to(self._device)

        self.name = f"hf_chat:{self.model_id}"

    def _generate_from_messages(self, messages) -> GenerationResult:
        if self.reasoning:
            messages = [
                *messages,
                {"role": "assistant", "content": "<think>\n"},
            ]
            add_gen_prompt = False
        else:
            add_gen_prompt = True

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self._device)

        do_sample = self.temperature is not None and float(self.temperature) > 0.0

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            temperature=float(self.temperature) if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        num_tokens = gen_ids.shape[0]

        decoded = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if self.reasoning:
            decoded = "<think>\n" + decoded
            
        return GenerationResult(text=decoded, tokens=num_tokens)

    def generate(self, prompt: str) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        return self._generate_from_messages(messages)
