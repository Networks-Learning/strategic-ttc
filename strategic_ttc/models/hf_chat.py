from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

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
    texts: List[str]
    tokens: List[int]
    waits: int = 0


@dataclass
class HFChatModel:
    model_id: str
    device: str = "auto"
    dtype: Optional[str] = None

    max_tokens: int = 512
    temperature: float = 0.0

    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self):
        torch_dtype = _to_torch_dtype(self.dtype)
        self._device = resolve_device(self.device)

        if "Ministral-3" in self.model_id or "Ministral3" in self.model_id:
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


            print(f"[HFChatModel] Using Ministral-3 tokenizer + model for {self.model_id}")

            self.tokenizer = MistralCommonBackend.from_pretrained(self.model_id)
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
            ).to(self._device)

            self.name = f"hf_chat:{self.model_id}"
            self._is_ministral = True

            return

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

    def _generate_once(self, messages) -> tuple[str, int]:
        do_sample = self.temperature is not None and float(self.temperature) > 0.0

        if getattr(self, "_is_ministral", False):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,          
                return_tensors="pt",
            ).to(self._device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=do_sample,
                temperature=float(self.temperature) if do_sample else None,
            )

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[0][prompt_len:]
            num_tokens = gen_ids.shape[0]

            decoded = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
            )
            return decoded, num_tokens

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self._device)

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
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return decoded, num_tokens


    def generate(self, prompt: str, wait: int = 0) -> GenerationResult:
        all_texts: List[str] = []
        all_tokens: List[int] = []

        messages = [{"role": "user", "content": prompt}]
        answer_0, tok_0 = self._generate_once(messages)
        all_texts.append(answer_0)
        all_tokens.append(tok_0)
        current_answer = answer_0

        for _ in range(max(0, wait)):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": current_answer},
                {
                    "role": "user",
                    "content": (
                        "Wait. Reflect on your previous answer carefully. "
                        "If you find any mistake, correct it and give an improved final answer. "
                        "Otherwise, restate your final answer clearly."
                    ),
                },
            ]

            refined, tok_r = self._generate_once(messages)
            all_texts.append(refined)
            all_tokens.append(tok_r)
            current_answer = refined

        return GenerationResult(
            texts=all_texts,
            tokens=all_tokens,
            waits=max(0, wait),
        )
