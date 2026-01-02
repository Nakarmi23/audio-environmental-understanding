import torch
from src.models.text_encoder import TextEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = TextEncoder(device=device)

captions = [
    "A dog barking loudly in the street",
    "Heavy rain falling on a metal roof",
    "People talking in a busy market",
]

emb = encoder(captions)

print("Embedding shape:", emb.shape)
print("Embedding dtype:", emb.dtype)
print("Device:", emb.device)
