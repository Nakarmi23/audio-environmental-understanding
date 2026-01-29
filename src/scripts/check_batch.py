from src.datasets.audiocaps_dataset import AudioCapsDataset
from src.datasets.collate import collate_audio_text
from torch.utils.data import DataLoader
from src.preprocessing.audio_features import LogMelExtractor

ds = AudioCapsDataset(
    "data/audiocaps/audiocaps_train.tsv",
    "data",
    target_sr=16000,
    clip_seconds=10.0,
)
dl = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_audio_text,
)

extractor = LogMelExtractor(
    sample_rate=16000,
    n_mels=64,
)

batch = next(iter(dl))
waveforms = batch["waveform"]
log_mel_specs = extractor(waveforms)

print("Waveforms shape:", waveforms.shape)
print("Log-Mel Spectrograms shape:", log_mel_specs.shape)
print("Captions:", batch["caption"])