import torch
import torch.nn as nn
import torchaudio

class SpecAugment(nn.Module):
    # Applies frequency and time masking to spectrograms for data augmentation
    def __init__(
        self,
        time_mask_param: int = 30,
        freq_mask_param: int = 8,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        p: float = 0.5,  # Probability of applying augmentation
    ):
        super().__init__()
        self.p = p
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only augment during training
        if not self.training:
            return x
        if torch.rand(1).item() > self.p:
            return x

        # Handle 4D tensors by temporarily removing channel dimension
        squeeze_channel = False
        if x.ndim == 4:
            x = x.squeeze(1)
            squeeze_channel = True

        out = x
        # Apply multiple frequency and time masks
        for _ in range(self.num_freq_masks):
            out = self.freq_mask(out)
        for _ in range(self.num_time_masks):
            out = self.time_mask(out)

        if squeeze_channel:
            out = out.unsqueeze(1)

        return out
