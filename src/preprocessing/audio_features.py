import torch
import torch.nn as nn
import torchaudio

class LogMelExtractor(nn.Module):
    # Converts raw audio waveforms to log-mel spectrograms for CNN processing
    def __init__(self, sample_rate=16000, n_mels=64, win_length_ms=25, hop_length_ms=10,n_fft=1024):
        super().__init__()
        # Convert milliseconds to samples
        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)

        mel = torch.clamp(mel, min=1e-10)  # Avoid log(0) errors

        log_mel = self.db(mel)

        # Replace any NaN/Inf values with zeros for numerical stability
        log_mel = torch.nan_to_num(
            log_mel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return log_mel