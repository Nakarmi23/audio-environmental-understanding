import torch
import torchaudio

class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_mels=64, win_length_ms=25, hop_length_ms=10):
        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=1024,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel(waveform)
        log_mel_spec = self.db(mel_spec)
        return log_mel_spec