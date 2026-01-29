import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    # 2-layer MLP that projects embeddings to shared space for contrastive learning
    def __init__(
            self,
            in_dim:int,
            out_dim:int = 256,
            hidden_dim:int=None,
            dropout:float = 0.1,
            normalize:bool = True,
        ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim  # Default hidden size matches input

        self.normalize = normalize

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1, eps=1e-8)  # L2 normalize for cosine similarity
        return z