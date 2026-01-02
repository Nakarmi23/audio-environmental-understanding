from src.datasets.audiocaps_dataset import AudioCapsDataset

ds = AudioCapsDataset(
    tsv_path="data/audiocaps/audiocaps_train.tsv",
    root_dir="data",
    target_sr=16000,
    clip_seconds=10.0,
)

sample = ds[0]
print("uid:", sample["uid"])
print("waveform shape:", sample["waveform"].shape)
print("caption:", sample["caption"])
print("duration:", sample["duration"])
print("path:", sample["audio_path"])
