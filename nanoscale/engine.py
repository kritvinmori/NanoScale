import os
import re
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .verifier import StepVerifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "verifier_head.pt")

class NanoScaleEngine:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", verifier_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NanoScale] Initializing engine on: {self.device}")

        if verifier_path is None:
            verifier_path = DEFAULT_WEIGHTS

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 1. Fix: Left-padding is mandatory for decoder-only batch generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

        self.verifier = StepVerifier(hidden_dim=1536).to(self.device)
        if os.path.exists(verifier_path):
            self.verifier.load(verifier_path, map_location=self.device)

    def _extract_boxed_answer(self, text: str) -> str:
        """Extracts answer from \boxed{...} or the last stated number."""
        # 1. Look for LaTeX \boxed{...}
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            clean = boxed[-1].replace("$", "").strip()
            # If boxed contains an equation like x = 12, take the right side
            if "=" in clean:
                clean = clean.split("=")[-1].strip()
            return clean

        # 2. Look for explicit answer statements
        match = re.findall(r"(?:answer is|takes|equals|=)\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            return match[-1].strip()

        # 3. Fallback: last number in the text
        nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
        return nums[-1].strip() if nums else "PARSE_ERROR"

    def run_baseline(self, prompt: str, max_tokens: int = 350) -> str:
        """Standard greedy baseline using Qwen's native reasoning prompt."""
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate(self, prompt: str, n_candidates: int = 3, max_new_tokens: int = 350):
        # Native Chain-of-Thought System Prompt
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # In-GPU Batching: replicate the clean prompt N times
        input_ids = inputs["input_ids"].repeat(n_candidates, 1)
        attention_mask = inputs["attention_mask"].repeat(n_candidates, 1)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,       # Provides diverse paths while staying logically grounded
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )

        candidates = []
        for i in range(n_candidates):
            gen_tokens = outputs.sequences[i][prompt_len:]
            candidates.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        # PRM hidden state scoring
        last_step_states = outputs.hidden_states[-1][-1]
        final_vectors = last_step_states[:, -1, :].to(torch.float32)
        with torch.no_grad():
            scores = self.verifier(final_vectors).squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]

        # Extract parsed answers
        answers = [self._extract_boxed_answer(c) for c in candidates]
        
        # Consensus & Verification Logic
        counter = Counter([a for a in answers if a != "PARSE_ERROR"])
        has_majority = False
        winning_answer = None

        if counter:
            most_common_answer, count = counter.most_common(1)[0]
            if count >= 2:
                # Majority consensus achieved (2 or 3 paths agreed)
                has_majority = True
                winning_answer = most_common_answer
                pool = [i for i, a in enumerate(answers) if a == winning_answer]
                best_idx = max(pool, key=lambda idx: scores[idx])

        if not has_majority:
            # Tie / No consensus: PRM score decides the winner
            best_idx = int(torch.tensor(scores).argmax().item())
            winning_answer = answers[best_idx]

        return {
            "selected_solution": candidates[best_idx],
            "selected_score": scores[best_idx],
            "extracted_answer": winning_answer,
            "has_consensus": has_majority,
            "all_candidates": candidates,
            "all_scores": scores,
            "all_answers": answers
        }