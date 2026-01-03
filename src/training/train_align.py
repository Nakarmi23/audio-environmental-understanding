import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.contrastive_loss import InfoNCELoss
from src.evaluation.retrieval import evaluate_retrieval

from src.utils.metrics_logger import MetricsLogger


def train_align(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader ,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    device: str = "cpu",
    out_dir: str = "runs/align",
    log_every: int = 50,
    use_amp: bool = True,
):
    metrics_logger = MetricsLogger(f"{out_dir}/metrics.csv")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Only train audio encoder + projection heads
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    params = list(model.audio_encoder.parameters()) + list(model.audio_proj.parameters()) + list(model.text_proj.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = InfoNCELoss(temperature=temperature)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    global_step = 0
    best_val = -1.0

    model.train()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for bi, batch in enumerate(train_loader, start=1):
            wave = batch["waveform"].to(device)   # (B,1,T)
            caps = batch["caption"]               # list[str]

            optimizer.zero_grad(set_to_none=True)



            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                audio_z, text_z = model(wave, caps)      # (B,D), (B,D)
                if not torch.isfinite(audio_z).all():
                    print("NaN/Inf in audio_z")
                    print("audio_z min/max:", audio_z.min().item(), audio_z.max().item())
                    break

                if not torch.isfinite(text_z).all():
                    print("NaN/Inf in text_z")
                    print("text_z min/max:", text_z.min().item(), text_z.max().item())
                    break
                loss = loss_fn(audio_z, text_z)

            scaler.scale(loss).backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                avg = running / bi
                print(f"[Epoch {epoch}/{epochs}] step={global_step} batch={bi} loss={avg:.4f}")

        # ---- Validation retrieval
        if val_loader is not None:
            model.eval()
            metrics = evaluate_retrieval(model, val_loader, max_batches=None)
            print(f"[VAL] epoch={epoch} N={metrics['N']} "
                  f"a2t R@1={metrics['a2t_R@1']:.3f} R@5={metrics['a2t_R@5']:.3f} R@10={metrics['a2t_R@10']:.3f} | "
                  f"t2a R@1={metrics['t2a_R@1']:.3f} R@5={metrics['t2a_R@5']:.3f} R@10={metrics['t2a_R@10']:.3f}")
            
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": running / bi,
                "a2t_R@1": metrics["a2t_R@1"],
                "a2t_R@5": metrics["a2t_R@5"],
                "a2t_R@10": metrics["a2t_R@10"],
                "t2a_R@1": metrics["t2a_R@1"],
                "t2a_R@5": metrics["t2a_R@5"],
                "t2a_R@10": metrics["t2a_R@10"],
            }

            metrics_logger.log(epoch_metrics)

            # Save best by a2t_R@10 (you can change criterion)
            score = metrics["a2t_R@10"]
            if score > best_val:
                best_val = score
                ckpt_path = os.path.join(out_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                    "metrics": metrics,
                }, ckpt_path)
                print(f"Saved best checkpoint to: {ckpt_path}")

        # Save last
        last_path = os.path.join(out_dir, "last.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
        }, last_path)

    print(f"Training complete. Best a2t_R@10 = {best_val:.3f}")
