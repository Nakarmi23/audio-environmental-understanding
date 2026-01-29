import torch
from torch.utils.data import DataLoader

from src.models.multimodal_model import MultimodalModel
from src.evaluation.retrieval import evaluate_retrieval
from src.datasets.audiocaps_dataset import AudioCapsDataset
from src.datasets.collate import collate_audio_text


def main():
    # Evaluate trained model on test set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "runs/align/best.pt"

    # Load trained model from checkpoint
    model = MultimodalModel(
        device=device,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_dataset = AudioCapsDataset("data/audiocaps/audiocaps_test.tsv", root_dir="data",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_audio_text,
        pin_memory=True,
    )

    with torch.no_grad():
        metrics = evaluate_retrieval(model, test_loader)

    print("\n===== TEST SET RESULTS =====")
    print(f"N = {metrics['N']}")
    print(f"a2t R@1  = {metrics['a2t_R@1']:.3f}")
    print(f"a2t R@5  = {metrics['a2t_R@5']:.3f}")
    print(f"a2t R@10 = {metrics['a2t_R@10']:.3f}")
    print("----------------------------")
    print(f"t2a R@1  = {metrics['t2a_R@1']:.3f}")
    print(f"t2a R@5  = {metrics['t2a_R@5']:.3f}")
    print(f"t2a R@10 = {metrics['t2a_R@10']:.3f}")


if __name__ == "__main__":
    main()
