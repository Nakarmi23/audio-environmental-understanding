import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            audio_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
        ) -> torch.Tensor:
        if audio_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D tensors of shape (batch_size, embedding_dim)")
        if audio_embeddings.shape[0] != text_embeddings.shape[0]:
            raise ValueError("Audio and text embeddings must have the same batch size")
        
        B = audio_embeddings.shape[0]
        device = audio_embeddings.device

        logits = (audio_embeddings @ text_embeddings.t()) / self.temperature

        labels = torch.arange(B, device=device)

        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_a2t + loss_t2a)