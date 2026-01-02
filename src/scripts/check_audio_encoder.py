import torch
from src.models.audio_encoder import AudioEncoder

x = torch.randn(4, 1, 64, 500)

model = AudioEncoder(n_mels=64, embedding_dim=512)
emb = model(x)

print("Embedding shape:", emb.shape)