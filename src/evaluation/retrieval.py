import torch


@torch.no_grad()
def recall_at_k(sim: torch.Tensor, k: int) -> float:
    topk = sim.topk(k, dim=1).indices
    targets = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    hits = (topk == targets).any(dim=1).float().mean().item()
    return hits


@torch.no_grad()
def evaluate_retrieval(model, dataloader, max_batches: int = None):
    model.eval()

    audio_all = []
    text_all = []

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        wave = batch["waveform"]
        caps = batch["caption"]

        audio_z, text_z = model(wave, caps)

        audio_all.append(audio_z)
        text_all.append(text_z)

    audio_z = torch.cat(audio_all, dim=0)
    text_z = torch.cat(text_all, dim=0)

    sim = audio_z @ text_z.T

    results = {
        "a2t_R@1": recall_at_k(sim, 1),
        "a2t_R@5": recall_at_k(sim, 5),
        "a2t_R@10": recall_at_k(sim, 10),
        "t2a_R@1": recall_at_k(sim.T, 1),
        "t2a_R@5": recall_at_k(sim.T, 5),
        "t2a_R@10": recall_at_k(sim.T, 10),
        "N": sim.size(0),
    }
    return results
