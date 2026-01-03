import torch
from src.models.projection import ProjectionHead
from src.training.contrastive_loss import InfoNCELoss

B = 4
audio = torch.randn(B, 512)
text = torch.randn(B, 384)

audio_proj = ProjectionHead(512, 256)
text_proj = ProjectionHead(384, 256)

audio_z = audio_proj(audio)
text_z = text_proj(text)

loss_fn = InfoNCELoss(temperature=0.07)
loss = loss_fn(audio_z, text_z)

print("audio_z:", audio_z.shape)
print("text_z:", text_z.shape)
print("loss:", loss.item())
