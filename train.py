import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nanoscale.verifier import StepVerifier
import os

# 1. Hardware & Model Configuration
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 6
LEARNING_RATE = 1e-4

print(f"--> Initializing training pipeline on: {DEVICE}")

# 2. Load 4-bit Quantized Base Qwen to extract hidden states with low VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"--> Loading base model ({MODEL_ID}) in 4-bit NF4...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    output_hidden_states=True
)

# Freeze base model parameters (we only train the verifier head)
for param in base_model.parameters():
    param.requires_grad = False

base_model.eval()

# 3. Curated Step-Reasoning Dataset (Positive vs. Flawed Steps)
# Training samples pair a reasoning step with a label (1.0 = sound, 0.0 = flawed)
training_samples = [
    # Positive logic steps (Label 1.0)
    {"text": "Problem: Solve 3x + 12 = 27. Step: Subtract 12 from both sides: 3x = 15. Divide by 3: x = 5.", "label": 1.0},
    {"text": "Problem: What is 15% of 80? Step: 10% is 8, 5% is 4. Total = 8 + 4 = 12.", "label": 1.0},
    {"text": "Problem: Probability of an even number on a standard die. Step: Evens are {2,4,6}. Probability is 3/6 = 1/2.", "label": 1.0},
    {"text": "Problem: Cheaper way to buy 26 pencils at 6 for $3 or $0.75 each. Step: 4 packs (24 pencils) cost $12. 2 singles cost $1.50. Total = $13.50, which is cheaper than 5 packs ($15).", "label": 1.0},
    {"text": "Problem: Car travels 60 mph for 2.5 hours. Step: Distance = Speed * Time = 60 * 2.5 = 150 miles.", "label": 1.0},
    {"text": "Problem: Factor x^2 - 16. Step: Difference of squares gives (x - 4)(x + 4).", "label": 1.0},
    {"text": "Problem: Perimeter of a rectangle with length 8 and width 5. Step: Perimeter = 2*(8 + 5) = 2*13 = 26.", "label": 1.0},
    {"text": "Problem: If 5 pens cost $15, how much do 9 cost? Step: Cost per pen = 15/5 = $3. Total for 9 = 9 * 3 = $27.", "label": 1.0},
    {"text": "Problem: Solve for y: 2y - 4 = 10. Step: 2y = 14, therefore y = 7.", "label": 1.0},
    {"text": "Problem: 20% discount on a $50 shirt. Step: Discount = 0.20 * 50 = $10. Sale price = 50 - 10 = $40.", "label": 1.0},

    # Negative / Flawed steps (Label 0.0)
    {"text": "Problem: Solve 3x + 12 = 27. Step: Subtract 12 from both sides to get 3x = 18.", "label": 0.0},
    {"text": "Problem: What is 15% of 80? Step: 15% of 80 is 80 - 15 = 65.", "label": 0.0},
    {"text": "Problem: Probability of an even number on a standard die. Step: Evens are {2,4}, so probability is 2/6 = 1/3.", "label": 0.0},
    {"text": "Problem: Cheaper way to buy 26 pencils at 6 for $3 or $0.75 each. Step: Buy 5 packs for $15, because you cannot buy individual pencils.", "label": 0.0},
    {"text": "Problem: Cheaper way to buy 26 pencils at 6 for $3 or $0.75 each. Step: 26 individual pencils cost 26 * 0.75 = $19.50, which is the only option.", "label": 0.0},
    {"text": "Problem: Car travels 60 mph for 2.5 hours. Step: Distance = 60 / 2.5 = 24 miles.", "label": 0.0},
    {"text": "Problem: Factor x^2 - 16. Step: This factors into (x - 8)(x + 2).", "label": 0.0},
    {"text": "Problem: Perimeter of a rectangle with length 8 and width 5. Step: Perimeter = 8 * 5 = 40.", "label": 0.0},
    {"text": "Problem: If 5 pens cost $15, how much do 9 cost? Step: 15 * 5 = 75, so 9 pens cost 75 / 9 = 8.33.", "label": 0.0},
    {"text": "Problem: Solve for y: 2y - 4 = 10. Step: 2y = 6, therefore y = 3.", "label": 0.0},
    {"text": "Problem: 20% discount on a $50 shirt. Step: Sale price = 50 - 20 = $30.", "label": 0.0},
]

class ReasoningDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.data = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.float32)
        }

# Prepare DataLoader
dataset = ReasoningDataset(training_samples, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Initialize StepVerifier Head
verifier = StepVerifier(hidden_dim=1536).to(DEVICE)
optimizer = torch.optim.AdamW(verifier.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# 5. Supervised Training Loop
print("\n--> Starting StepVerifier Training Loop...")
verifier.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(-1)

        # Extract hidden states from Qwen
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            # Pull last hidden state of the final token in the sequence
            last_hidden_state = outputs.hidden_states[-1][:, -1, :].to(torch.float32)

        # Forward pass through StepVerifier
        optimizer.zero_grad()
        predictions = verifier(last_hidden_state)
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

# 6. Save the trained weights
OUTPUT_PATH = "verifier_head.pt"
verifier.save(OUTPUT_PATH)
print(f"\n[Success] PRM Head trained and saved to {OUTPUT_PATH}")