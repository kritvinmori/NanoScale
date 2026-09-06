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

    def _extract_answer(self, text: str) -> str:
        """Robust parser for boxed, bolded, or unit-suffixed numerical answers."""
        # 1. Standard or unclosed LaTeX boxed
        boxed = re.findall(r"\\boxed\{?\$?(-?\d+(?:\.\d+)?)", text)
        if boxed:
            return boxed[-1].strip()

        # 2. Bolded markdown numbers, e.g. **12 days** or **36**
        bolded = re.findall(r"\*\*(-?\d+(?:\.\d+)?)\s*(?:days?|hours?|pencils?|dollars?|\$)?\*\*", text, re.IGNORECASE)
        if bolded:
            return bolded[-1].strip()

        # 3. Explicit result phrases
        phrases = re.findall(r"(?:take|takes|equals|is|answer is|=)\s*\*?\*?\$?(-?\d+(?:\.\d+)?)\s*(?:days?|hours?|pencils?|dollars?)", text, re.IGNORECASE)
        if phrases:
            return phrases[-1].strip()

        # 4. Fallback: extract last standalone number in the response
        numbers = re.findall(r"\b(-?\d+(?:\.\d+)?)\b", text)
        return numbers[-1].strip() if numbers else "PARSE_ERROR"

    def run_baseline(self, prompt: str, max_tokens: int = 300) -> str:
        messages = [
            {"role": "system", "content": "Solve concisely step-by-step. Put the final numerical answer inside \\boxed{}."},
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

    def generate(self, prompt: str, n_candidates: int = 3, max_new_tokens: int = 300):
        system_instruction = (
            "You are a precise mathematical optimizer.\n"
            "Solve the problem step-by-step using standard arithmetic:\n"
            "- If this is a rate/worker problem, apply the compound work invariant: (Workers * Days) / Work = Constant.\n"
            "- State each calculation on a clean line.\n"
            "- End your response with: Final Answer: \\boxed{your_number}"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        input_ids = inputs["input_ids"].repeat(n_candidates, 1)
        attention_mask = inputs["attention_mask"].repeat(n_candidates, 1)

        # High-speed generation without caching intermediate hidden layers
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        candidates = []
        for i in range(n_candidates):
            gen_tokens = outputs[i][prompt_len:]
            candidates.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        # Sub-millisecond forward pass to extract final token hidden states for PRM scoring
        eval_inputs = outputs[:, -64:]
        with torch.no_grad():
            forward_out = self.model(eval_inputs, output_hidden_states=True)
            final_vectors = forward_out.hidden_states[-1][:, -1, :].to(torch.float32)
            scores = self.verifier(final_vectors).squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]

        # Extract parsed answers
        answers = [self._extract_answer(c) for c in candidates]
        valid_answers = [a for a in answers if a != "PARSE_ERROR"]

        has_majority = False
        winning_answer = None

        if valid_answers:
            most_common_answer, count = Counter(valid_answers).most_common(1)[0]
            if count >= 2:
                has_majority = True
                winning_answer = most_common_answer
                pool = [i for i, a in enumerate(answers) if a == winning_answer]
                best_idx = max(pool, key=lambda idx: scores[idx])

        if not has_majority:
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