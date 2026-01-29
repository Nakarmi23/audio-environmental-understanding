# Audio-Environmental Understanding

A multimodal deep learning system for audio-text alignment and cross-modal retrieval using contrastive learning. The project implements an audio encoder and text encoder with projection heads to learn a shared embedding space where audio and text representations are aligned.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
  - [Download](#download)
  - [Dataset Structure](#dataset-structure)
  - [Expected Data Directory](#expected-data-directory)
  - [Audio Specifications](#audio-specifications)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
  - [Python Version](#python-version)
  - [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Qualitative Demo](#qualitative-demo)
  - [Verification Scripts](#verification-scripts)
  - [Visualization](#visualization)
- [Models](#models)
  - [Audio Encoder](#audio-encoder)
  - [Text Encoder](#text-encoder)
  - [Projection Heads](#projection-heads)
  - [Loss Function](#loss-function)
- [Outputs](#outputs)
  - [Training Outputs](#training-outputs)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Console Output](#console-output)

## Overview

This project addresses the task of **audio-text retrieval** by training a multimodal model that can:
- **Audio-to-Text Retrieval (A2T)**: Given an audio clip, find the most relevant text descriptions
- **Text-to-Audio Retrieval (T2A)**: Given a text query, find the most relevant audio clips

The system uses:
- **CNN-based Audio Encoder**: Processes log-mel spectrograms to extract audio features
- **Pre-trained Text Encoder**: Uses Sentence-BERT (MiniLM) for text embeddings
- **Contrastive Learning**: InfoNCE loss to align audio and text in a shared embedding space
- **SpecAugment**: Data augmentation for improved robustness

## Dataset

The project uses the **AudioCaps** dataset, which contains audio clips paired with descriptive captions.

### Download
We use the AudioCaps dataset from Kaggle: **[AudioCaps Dataset](https://www.kaggle.com/datasets/nickkar30/audiocaps)**

This Kaggle version is particularly convenient because it provides the actual audio files directly. The original AudioCaps dataset only provides YouTube links to the audio, which requires manually downloading each clip from YouTube. Using this pre-downloaded version saves significant time and avoids issues with missing or unavailable YouTube videos.

### Dataset Structure
The dataset is expected in TSV (Tab-Separated Values) format with the following columns:
- `uniq_id`: Unique identifier for each sample
- `audio`: Relative path to the audio file
- `text`: Text caption describing the audio
- `duration`: Duration of the audio clip in seconds

### Expected Data Directory
After downloading and extracting the dataset, copy all files and folders from the AudioCaps download into `data/audiocaps/`. The final structure should be:

```
data/
└── audiocaps/
    ├── audiocaps_train.tsv
    ├── audiocaps_val.tsv
    ├── audiocaps_test.tsv
    ├── audio/                # Audio files directory
    │   ├── train/            # Training audio files
    │   ├── val/              # Validation audio files
    │   └── test/             # Test audio files
    └── [other files from the dataset]
```

### Audio Specifications
- **Sample Rate**: 16,000 Hz
- **Clip Duration**: 10 seconds (padded or truncated)
- **Format**: Mono audio (stereo is automatically converted)

## Project Structure

```
audio-environmental-understanding/
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── src/
│   ├── datasets/
│   │   ├── audiocaps_dataset.py    # AudioCaps dataset loader
│   │   ├── collate.py              # Batch collation functions
│   │   └── loader.py               # DataLoader utilities
│   ├── models/
│   │   ├── audio_encoder.py        # CNN-based audio encoder
│   │   ├── text_encoder.py         # Sentence-BERT text encoder
│   │   ├── projection.py           # Projection heads for alignment
│   │   └── multimodal_model.py     # Main multimodal model
│   ├── preprocessing/
│   │   ├── audio_features.py       # Log-mel spectrogram extraction
│   │   └── spec_augment.py         # SpecAugment data augmentation
│   ├── training/
│   │   ├── train_align.py          # Training loop for alignment
│   │   └── contrastive_loss.py     # InfoNCE contrastive loss
│   ├── evaluation/
│   │   └── retrieval.py            # Retrieval evaluation metrics (Recall@K)
│   ├── scripts/
│   │   ├── run_train_align.py      # Main training script
│   │   ├── eval_test_retrieval.py  # Test set evaluation
│   │   ├── demo_retrieval.py       # Qualitative retrieval demo
│   │   ├── check_dataset.py        # Dataset verification
│   │   ├── check_audio_encoder.py  # Audio encoder testing
│   │   ├── check_text_encoder.py   # Text encoder testing
│   │   ├── check_batch.py          # Batch processing verification
│   │   └── check_contrastive.py    # Contrastive loss verification
│   ├── utils/
│   │   └── metrics_logger.py       # CSV-based metrics logging
│   └── metrics_visualization.ipynb # Jupyter notebook for visualizations
└── runs/                      # Training outputs (created automatically)
    └── align/
        ├── best.pt            # Best model checkpoint
        ├── last.pt            # Last epoch checkpoint
        └── metrics.csv        # Training metrics log
```

## Requirements

### Python Version
- Python 3.8+

### Dependencies
All required packages are listed in `requirements.txt`:
- **PyTorch** (≥2.0.0): Deep learning framework
- **torchaudio** (≥2.0.0): Audio processing
- **pandas** (≥1.5.0, <3.0.0): Data handling
- **numpy** (≥1.23.0): Numerical operations
- **matplotlib** (≥3.7): Visualization
- **sentence-transformers** (2.2.2): Pre-trained text encoders
- **transformers** (4.26.1): Hugging Face transformers
- **huggingface_hub** (0.13.4): Model hub integration

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd audio-environmental-understanding
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

#### For CPU-only:
```bash
pip install -r requirements.txt
```

#### For GPU (CUDA):
If you have an NVIDIA GPU with CUDA support, install the CUDA-enabled version of PyTorch for significantly faster training:

```bash
# For CUDA 11.8
pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install -r requirements.txt
```

Check your CUDA version with `nvcc --version` or `nvidia-smi`. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for other CUDA versions.

### 4. Prepare the Dataset
- Download the AudioCaps dataset from [Kaggle](https://www.kaggle.com/datasets/nickkar30/audiocaps)
- Extract the downloaded files
- Create the directory: `mkdir -p data/audiocaps`
- Copy **all files and folders** from the extracted AudioCaps dataset into `data/audiocaps/`
- Verify that `data/audiocaps/` contains the TSV files, the `audio/` folder with `train/`, `val/`, and `test/` subdirectories, and any other dataset files

## Usage

### Training

Run the main training script to train the audio-text alignment model:

```bash
python -m src.scripts.run_train_align
```

**Training Configuration:**
- **Batch Size**: 16
- **Epochs**: 15
- **Learning Rate**: 3e-4
- **Weight Decay**: 1e-4
- **Temperature**: 0.07 (for InfoNCE loss)
- **Mixed Precision**: Enabled (AMP)
- **Output Directory**: `runs/align/`

**What Gets Trained:**
- Audio encoder (CNN)
- Audio projection head
- Text projection head
- Text encoder is **frozen** (pre-trained weights)

### Evaluation

Evaluate the trained model on the test set:

```bash
python -m src.scripts.eval_test_retrieval
```

This script:
- Loads the best checkpoint from `runs/align/best.pt`
- Evaluates on the test set
- Reports Recall@1, Recall@5, and Recall@10 for both A2T and T2A retrieval

### Qualitative Demo

Run a qualitative retrieval demo to see top-5 retrieved captions for a random audio sample:

```bash
python -m src.scripts.demo_retrieval
```

### Verification Scripts

Test individual components:

```bash
# Check dataset loading
python -m src.scripts.check_dataset

# Check audio encoder
python -m src.scripts.check_audio_encoder

# Check text encoder
python -m src.scripts.check_text_encoder

# Check batch collation
python -m src.scripts.check_batch

# Check contrastive loss
python -m src.scripts.check_contrastive
```

### Visualization

Open the Jupyter notebook for metrics visualization:

```bash
jupyter notebook src/metrics_visualization.ipynb
```

The notebook visualizes training metrics from `runs/align/metrics.csv`.

## Models

### Audio Encoder
- **Architecture**: 4-layer CNN with BatchNorm and MaxPooling
- **Input**: Log-mel spectrogram (64 mel bins)
- **Output**: 512-dimensional embedding
- **Layers**:
  - Conv2D: 1→32→64→128→256 channels
  - Global Average Pooling
  - Fully Connected: 256→512

### Text Encoder
- **Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Output**: 384-dimensional embedding
- **Training**: Frozen (pre-trained weights used)

### Projection Heads
- **Architecture**: 2-layer MLP with BatchNorm and ReLU
- **Input Dimensions**: 
  - Audio: 512
  - Text: 384
- **Output Dimension**: 256 (shared embedding space)

### Loss Function
- **InfoNCE (NT-Xent)**: Normalized Temperature-scaled Cross-Entropy
- **Temperature**: 0.07
- **Purpose**: Pull positive pairs together, push negative pairs apart

## Outputs

### Training Outputs

All training artifacts are saved to `runs/align/`:

#### Checkpoints
- **`best.pt`**: Best model based on validation A2T Recall@10
  - Contains: model state, optimizer state, epoch, metrics
- **`last.pt`**: Checkpoint from the last epoch
  - Contains: model state, optimizer state, epoch, best validation score

#### Metrics Log
- **`metrics.csv`**: Training and validation metrics per epoch
  - Columns:
    - `epoch`: Epoch number
    - `train_loss`: Average training loss
    - `a2t_R@1`, `a2t_R@5`, `a2t_R@10`: Audio-to-text retrieval recall
    - `t2a_R@1`, `t2a_R@5`, `t2a_R@10`: Text-to-audio retrieval recall

### Evaluation Metrics

The model is evaluated using **Recall@K** metrics:

- **Recall@1**: Percentage of queries where the correct match is in the top-1 result
- **Recall@5**: Percentage of queries where the correct match is in the top-5 results
- **Recall@10**: Percentage of queries where the correct match is in the top-10 results

Both directions are evaluated:
- **A2T (Audio-to-Text)**: Given audio, retrieve text
- **T2A (Text-to-Audio)**: Given text, retrieve audio

### Console Output

During training, you'll see logs like:
```
[Epoch 1/15] step=50 batch=50 loss=2.3456
[VAL] epoch=1 N=495 a2t R@1=0.123 R@5=0.345 R@10=0.456 | t2a R@1=0.111 R@5=0.333 R@10=0.444
Saved best checkpoint to: runs/align/best.pt
```
