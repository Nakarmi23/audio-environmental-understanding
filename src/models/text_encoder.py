import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    # Wraps Sentence-BERT model to encode text captions into embeddings
    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            device: str = "cpu",
            trainable: bool = False,
    ):
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

        # Freeze weights if not training text encoder (saves memory and compute)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
    
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, captions: list[str]) -> torch.Tensor:
        # Enable gradients only if model is trainable
        with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
            embeddings = self.model.encode(
                captions,
                convert_to_tensor=True,
                normalize_embeddings=False,
                device=self.device,
            )

        return embeddings