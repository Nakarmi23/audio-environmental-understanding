from torch.utils.data import DataLoader

from datasets.audiocaps_dataset import AudioCapsDataset

def make_loader(dataset: AudioCapsDataset, batch_size=16, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
