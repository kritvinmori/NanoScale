import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .verifier import StepVerifier

class NanoScaleEngine:
    """
    Test-Time Compute Search Engine.
    Generates N parallel reasoning paths on Qwen2.5-1.5B and selects
    the highest-confidence candidate using the trained PRM StepVerifier.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", verifier_path: str = "verifier_head.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NanoScale] Initializing engine on: {self.device}")

        # 1. 4-bit Quantization Configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. Load Base Model & Tokenizer
        print("[NanoScale] Loading quantized base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

        # 3. Load Trained PRM Verifier Head
        print("[NanoScale] Loading trained PRM verifier head...")
        self.verifier = StepVerifier(hidden_dim=1536).to(self.device)
        self.verifier.load(verifier_path, map_location=self.device)

    def generate(self, prompt: str, n_candidates: int = 3, max_new_tokens: int = 256):
        """
        Executes parallel candidate generation and PRM verification.
        """
        # Format prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        prompt_len = inputs["input_ids"].shape[1]

        # In-GPU Batching: duplicate prompt across N candidate rows
        input_ids = inputs["input_ids"].repeat(n_candidates, 1)
        attention_mask = inputs["attention_mask"].repeat(n_candidates, 1)

        # Generate N reasoning trajectories simultaneously
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )

        # Extract generated tokens
        sequences = outputs.sequences
        candidates = []
        for i in range(n_candidates):
            gen_tokens = sequences[i][prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            candidates.append(text)

        # Extract last hidden state of final generated token across all candidates
        # outputs.hidden_states is a tuple of generation steps
        last_step_states = outputs.hidden_states[-1][-1] # Last layer, last generated step
        final_vectors = last_step_states[:, -1, :].to(torch.float32)

        # Score candidate trajectories in parallel (< 2ms)
        with torch.no_grad():
            scores = self.verifier(final_vectors).squeeze(-1).tolist()

        # Select candidate with highest PRM score
        best_idx = int(torch.tensor(scores).argmax().item())

        return {
            "selected_solution": candidates[best_idx],
            "selected_score": scores[best_idx],
            "all_candidates": candidates,
            "all_scores": scores
        }