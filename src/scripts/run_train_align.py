import torch
from torch.utils.data import DataLoader

from src.datasets.audiocaps_dataset import AudioCapsDataset
from src.datasets.collate import collate_audio_text
from src.models.multimodal_model import MultimodalModel
from src.training.train_align import train_align


def main():
    # Main training script for audio-text alignment
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load training and validation datasets
    train_ds = AudioCapsDataset("data/audiocaps/audiocaps_train.tsv", root_dir="data", target_sr=16000, clip_seconds=10.0)
    val_ds   = AudioCapsDataset("data/audiocaps/audiocaps_val.tsv", root_dir="data", target_sr=16000, clip_seconds=10.0)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_audio_text)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_audio_text)

    # Initialize multimodal model with frozen text encoder
    model = MultimodalModel(
        sample_rate=16000,
        n_mels=64,
        audio_emb_dim=512,
        text_model_name="all-MiniLM-L6-v2",
        shared_dim=256,
        device=device,
        text_trainable=False,  # Keep text encoder frozen
    )

    # Train with contrastive learning
    train_align(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=3e-4,
        weight_decay=1e-4,
        temperature=0.07,  # Temperature for InfoNCE loss
        device=device,
        out_dir="runs/align",
        log_every=50,
        use_amp=True,
    )


if __name__ == "__main__":
    main()
