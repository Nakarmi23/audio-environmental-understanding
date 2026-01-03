import torch
import torch.nn as nn

from src.preprocessing.audio_features import LogMelExtractor
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder
from src.models.projection import ProjectionHead


class MultimodalModel(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        audio_emb_dim: int = 512,
        text_model_name: str = "all-MiniLM-L6-v2",
        shared_dim: int = 256,
        device: str = "cpu",
        text_trainable: bool = False,
    ):
        super().__init__()
        self.device = device

        self.logmel = LogMelExtractor(sample_rate=sample_rate, n_mels=n_mels).to(device)

        self.audio_encoder = AudioEncoder(n_mels=n_mels, embedding_dim=audio_emb_dim)

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            device=device,
            trainable=text_trainable,
        )
        text_emb_dim = self.text_encoder.embedding_dim

        self.audio_proj = ProjectionHead(in_dim=audio_emb_dim, out_dim=shared_dim)
        self.text_proj = ProjectionHead(in_dim=text_emb_dim, out_dim=shared_dim)

        self.to(device)

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = waveforms.to(self.device)

        logmel = self.logmel(waveforms)


        if logmel.ndim == 3:
            logmel = logmel.unsqueeze(1)

        audio_emb = self.audio_encoder(logmel)
        audio_z = self.audio_proj(audio_emb)
        return audio_z

    @torch.no_grad()
    def encode_text(self, captions: list[str]) -> torch.Tensor:
        text_emb = self.text_encoder(captions)
        text_emb = text_emb.to(self.device)
        text_z = self.text_proj(text_emb)
        return text_z

    def forward(self, waveforms: torch.Tensor, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        audio_z = self.encode_audio(waveforms)

        text_z = self.encode_text(captions)
        return audio_z, text_z
