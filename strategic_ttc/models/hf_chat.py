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


    def generate(
        self,
        prompt: str,
        wait: int = 0,
        feedback_model: Optional["HFChatModel"] = None,
    ) -> GenerationResult:
        all_texts: List[str] = []
        all_tokens: List[int] = []

        messages = [{"role": "user", "content": prompt}]
        answer_0, tok_0 = self._generate_once(messages)
        all_texts.append(answer_0)
        all_tokens.append(tok_0)
        current_answer = answer_0

        refine_messages = [
            {"role": "user", "content": prompt},
        ]
        
        for _ in range(max(0, wait)):
            refine_messages.append({"role": "assistant", "content": current_answer})
            if feedback_model is not None:
                fb_messages = [
                    {
                        "role": "user",
                        "content": (
                            "Here is a math problem and a proposed solution.\n\n"
                            f"Problem:\n{prompt}\n\n"
                            f"Proposed solution:\n{current_answer}\n\n"
                            f"History of previous attempts and refinements:\n"
                            f"{refine_messages}\n"
                            "Carefully check the reasoning and the final answer. "
                            "If there is any mistake, explain it briefly and provide a corrected solution. "
                            "If it is already correct, explain why it is correct and restate the final answer within \\boxed{}.\n\n"
                        ),
                    },
                ]

                feedback_text, _ = feedback_model._generate_once(fb_messages)
            else:
                raise NotImplementedError("Feedback model is required for refinement.")

            print(feedback_text)
            print("\n\n")
            refine_instruction = (
                "You previously answered the question above. "
                "Now you are given feedback on your solution:\n\n"
                f"{feedback_text}\n\n"
                "Using this feedback, produce an improved final answer to the original problem. "
                "If the feedback says your answer was correct, you can keep it but restate the reasoning steps clearly. "
                "Always end your response with your final answer within \\boxed{}\n"
            )

            refine_messages.append({"role": "user", "content": refine_instruction})
            refined, tok_r = self._generate_once(refine_messages)
            all_texts.append(refined)
            all_tokens.append(tok_r)
            current_answer = refined

        return GenerationResult(
            texts=all_texts,
            tokens=all_tokens,
            waits=max(0, wait),
        )
