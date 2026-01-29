import torch

def collate_audio_text(batch):
    # Collate function to batch audio waveforms and text captions together
    waveforms = torch.stack([item["waveform"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    uids = [item["uid"] for item in batch]
    return {
        "waveform": waveforms,
        "caption": captions,
        "uid": uids,
        }