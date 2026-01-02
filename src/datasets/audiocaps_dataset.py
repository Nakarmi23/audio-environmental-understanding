from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from typing import Dict, Tuple, Any

import torch
import torchaudio
from torch.utils.data import Dataset

@dataclass
class AudioCapsItem:
    uid: int
    audio_path: Path
    caption: str
    duration: float

class AudioCapsDataset(Dataset):
    def __init__(
            self,
            tsv_path: str ,
            root_dir: str ,
            target_sr: int = 16000,
            clip_seconds: float = 10.0,
    ):
        self.tsv_path = Path(tsv_path)
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds
        self.clip_samples = int(target_sr * clip_seconds)

        self.df = pd.read_csv(self.tsv_path, sep='\t', dtype={"uniq_id": int})
        required = {"uniq_id", "audio", "text", "duration"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in TSV: {missing}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, sr = torchaudio.load(str(path))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        n = waveform.shape[1]
        if n < self.clip_samples:
            pad = self.clip_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif n > self.clip_samples:
            waveform = waveform[:, :self.clip_samples]
        
        return waveform
    
    def __getitem__(self, index) -> Dict[str, Any]:
        row = self.df.iloc[index]

        uid = int(row["uniq_id"])
        rel_audio = str(row["audio"]).strip()
        caption = str(row["text"]).strip()
        duration = float(row["duration"])

        audio_path = (self.root_dir / rel_audio).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        waveform = self._load_audio(audio_path)

        return {
            "uid": uid,
            "waveform": waveform,
            "caption": caption,
            "duration": duration,
            "audio_path": str(audio_path),
        }