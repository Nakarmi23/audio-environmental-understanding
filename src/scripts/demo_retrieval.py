import torch
from torch.utils.data import DataLoader

from src.models.multimodal_model import MultimodalModel
from src.datasets.audiocaps_dataset import AudioCapsDataset
from src.datasets.collate import collate_audio_text


@torch.no_grad()
def main():
    # Demo: retrieve top-5 captions for a random audio sample
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MultimodalModel(device=device).to(device)
    ckpt = torch.load("runs/align/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load test dataset
    test_dataset = AudioCapsDataset(
        tsv_path="data/audiocaps/audiocaps_test.tsv",
        root_dir="data",
    )

    loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_audio_text,
    )

    # Encode all text captions from the dataset
    all_text_embs = []
    all_captions = []

    for batch in loader:
        caps = batch["caption"]
        text_z = model.encode_text(caps)   # (B, D)
        all_text_embs.append(text_z.cpu())
        all_captions.extend(caps)

    all_text_embs = torch.cat(all_text_embs, dim=0)  # (N, D)

    random_index = torch.randint(0, len(test_dataset), (1,)).item()

    # Pick one audio sample and encode it
    sample = test_dataset[random_index]
    wave = sample["waveform"].unsqueeze(0).to(device)
    gt_caption = sample["caption"]

    audio_z = model.encode_audio(wave).cpu()  # (1, D)

    # Compute similarity with all text captions and get top-5
    sims = (audio_z @ all_text_embs.T).squeeze(0)  # (N,)
    topk = torch.topk(sims, k=5)

    print("\n=== QUALITATIVE RETRIEVAL RESULT ===")
    print("Audio sample index in test set:", sample["uid"])
    print("Ground truth caption:")
    print(" >", gt_caption)
    print("\nTop-5 retrieved captions:")

    for rank, idx in enumerate(topk.indices.tolist(), start=1):
        print(f"{rank}. {all_captions[idx]}")


if __name__ == "__main__":
    main()
