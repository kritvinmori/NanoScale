import torch
import torch.nn as nn

class StepVerifier(nn.Module):
    """
    Process Reward Model (PRM) Head.
    Takes a 1536-dimensional hidden state vector from Qwen2.5-1.5B 
    and predicts whether the reasoning trajectory is mathematically sound.
    """
    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        
        # 2-layer Multi-Layer Perceptron (MLP)
        self.classifier = nn.Sequential(
            # Layer 1: Compress latent space from 1536 down to 256
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),          # Smooth non-linear activation
            nn.Dropout(0.1),     # Prevents overfitting during training
            
            # Layer 2: Compress 256 representations down to 1 scalar logit
            nn.Linear(256, 1),
            
            # Sigmoid squashes the logit strictly between 0.0 and 1.0
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:  [batch_size, hidden_dim]
        Output shape: [batch_size, 1]
        """
        return self.classifier(x)

    def save(self, path: str = "verifier_head.pt"):
        """Saves trained weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"[NanoScale] Weights successfully saved to: {path}")

    def load(self, path: str = "verifier_head.pt", map_location: str = "cuda"):
        """Loads trained weights onto device."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
        print(f"[NanoScale] Weights successfully loaded from: {path}")