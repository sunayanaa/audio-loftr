# Robust Audio Alignment via Audio LoFTR

Code accompanying the paper submitted to Elsevier Signal Processing:


## Overview

Audio LoFTR adapts the detector-free LoFTR visual correspondence
architecture to the problem of temporal audio alignment under
time-stretch and pitch-shift distortion. A CNN mel-spectrogram
encoder with frequency pooling feeds a cross-attention Transformer
with Rotary Position Embeddings (RoPE), producing a coarse alignment
path. A spectral DTW fine stage refines this to sub-frame precision.

The key findings reproduced by this code are:

- The coarse stage achieves 11–17 ms MAE across stretch rates
  0.8×–1.4×, outperforming MFCC+DTW (18–33 ms) throughout.
- Under combined time-stretch (1.2×) and pitch-shift (±4 semitones),
  the coarse stage maintains 12–13 ms MAE while Spec+DTW degrades
  to 81–286 ms.
- Replacing RoPE with sinusoidal absolute positional encoding (APE)
  degrades coarse MAE from ~17 ms to 575 ms at 0.8× stretch,
  confirming that RoPE is necessary for rate generalisation.

---

## Repository Structure

```
.
├── README.md
├── demo_coarse_to_fine_v5.py   # Model definition, training, and main evaluation
├── eval_multirate_v1.py        # Multi-rate evaluation (Table 1, Figure 2)
├── eval_pitch_v1.py            # Pitch-shift robustness (Table 2, Figure 3)
├── eval_pitch_v2.py            # Chroma fine-stage experiment (Section 6.1)
└── ablation_rope_v1.py         # RoPE vs APE ablation (Table 3, Figure 4)
```

---

## Requirements

All scripts are designed to run in Google Colab with a T4 GPU.
Dependencies are available in the standard Colab environment:

```
torch >= 2.0
librosa >= 0.10
numpy
scipy
matplotlib
```

No additional installation is required in Colab.

---

## Data

All experiments use the **GTZAN Genre Collection** (Jazz subset),
publicly available at:
http://marsyas.info/downloads/datasets.html

Download the Jazz sub-genre (100 tracks, ~30 s each, 22 050 Hz mono)
and place the `.au` files in a folder accessible from Colab, e.g.:

```
/content/drive/MyDrive/GTZAN/jazz/
```

The scripts discover all audio files in that folder automatically
and support `.au`, `.wav`, `.mp3`, and `.flac` formats.

---

## Reproducing the Paper Results

All scripts read from and write to Google Drive. Set the path at
the top of each script by editing `DRIVE_BASE`:

```python
DRIVE_BASE = "/content/drive/MyDrive/GTZAN"
```

### Step 1 — Train the model and run the main evaluation

```bash
# Run in Colab
demo_coarse_to_fine_v5.py
```

This script:
- Creates a stable 80/20 track-level split and saves it as
  `ctf_v5_split.json` (reused by all subsequent scripts).
- Trains the Audio LoFTR model for 50 epochs (~2 hours on T4 GPU).
- Saves the trained checkpoint as `model_ctf_v5.pth`.
- Evaluates all methods on 60 test windows at stretch rate 1.2×.
- Saves per-window results to `ctf_v5_results.json`.
- Saves the pipeline figure to `ctf_v5_figure.png`.

Training resumes automatically from the last saved checkpoint if
interrupted. Evaluation resumes from the cached JSON file.

**Expected output (rate = 1.2×, 60 windows):**

| Method | Mean MAE (ms) | Median (ms) |
|---|---|---|
| MFCC + DTW | 20.0 | 15.1 |
| Spec + DTW | 6.4 | 6.3 |
| LoFTR Coarse | 11.7 | 11.7 |
| LoFTR Fine | 6.4 | 6.3 |

---

### Step 2 — Multi-rate evaluation

```bash
eval_multirate_v1.py
```

Requires: `model_ctf_v5.pth` and `ctf_v5_split.json` from Step 1.

Evaluates stretch rates 0.8×, 0.9×, 1.1×, 1.2×, 1.3×, 1.4× on
the same 60 test windows. Results are cached to
`ctf_multirate_results.json` and the figure saved to
`ctf_multirate_figure.png`.

**Expected output (mean MAE ms):**

| Method | 0.8× | 0.9× | 1.1× | 1.2× | 1.3× | 1.4× |
|---|---|---|---|---|---|---|
| MFCC + DTW | 33.0 | 23.2 | 18.9 | 20.0 | 22.9 | 25.0 |
| Spec + DTW | 6.3 | 5.6 | 5.8 | 6.4 | 7.0 | 7.5 |
| LoFTR Coarse | 16.9 | 12.8 | 11.7 | 11.7 | 13.5 | 16.1 |
| LoFTR Fine | 6.3 | 5.6 | 5.8 | 6.4 | 7.0 | 7.5 |

---

### Step 3 — Pitch-shift robustness

```bash
eval_pitch_v1.py
```

Requires: `model_ctf_v5.pth` and `ctf_v5_split.json` from Step 1.

Evaluates stretch rate 1.2× combined with pitch shifts −4, −2, −1,
0, +1, +2, +4 semitones. Results cached to `ctf_pitch_results.json`,
figure to `ctf_pitch_figure.png`.

**Expected output (mean MAE ms):**

| Method | −4st | −2st | −1st | 0st | +1st | +2st | +4st |
|---|---|---|---|---|---|---|---|
| MFCC + DTW | 210.7 | 102.3 | 76.2 | 20.0 | 61.3 | 54.8 | 113.4 |
| Spec + DTW | 172.2 | 193.1 | 81.3 | 6.4 | 91.7 | 286.4 | 162.7 |
| LoFTR Coarse | **12.9** | **12.5** | **12.7** | **11.7** | **12.8** | **12.3** | **12.6** |
| LoFTR Fine | 59.8 | 69.8 | 43.5 | 6.4 | 45.2 | 70.7 | 63.1 |

---

### Step 4 — Chroma fine-stage experiment (Section 6.1)

```bash
eval_pitch_v2.py
```

Requires: `model_ctf_v5.pth` and `ctf_v5_split.json` from Step 1.

Compares CQT-chroma fine alignment against mel-spectrogram fine
alignment under pitch shift. This experiment supports the discussion
in Section 6.1: chroma features do not improve the fine stage on
polyphonic jazz because chroma vectors lack local discriminative
power in dense harmonic textures. Results cached to
`ctf_pitch_v2_results.json`, figure to `ctf_pitch_v2_figure.png`.

---

### Step 5 — RoPE vs APE ablation

```bash
ablation_rope_v1.py
```

Requires: `ctf_v5_split.json` from Step 1.
Does **not** require the pre-trained checkpoint — trains two models
from scratch under identical conditions.

Trains and evaluates two model variants:
- `model_rope.pth` — RoPE self-attention (proposed)
- `model_ape.pth` — sinusoidal absolute positional encoding (ablation)

Both are evaluated on stretch rates 0.8×–1.4×. Results cached to
`ablation_rope_results.json`, figure to `ablation_rope_figure.png`.
Total runtime approximately 4 hours on T4 GPU (two full training
runs of 50 epochs each).

**Expected output (coarse MAE ms):**

| Model | 0.8× | 0.9× | 1.1× | 1.2× | 1.3× | 1.4× |
|---|---|---|---|---|---|---|
| LoFTR + RoPE | 16.5 | 12.8 | 11.7 | 11.8 | 14.5 | 14.8 |
| LoFTR + APE | 575.2 | 12.8 | 11.6 | 11.6 | 240.7 | 488.2 |

---

## Ground Truth Convention

All experiments use `librosa.effects.time_stretch(y, rate=r)`, which
produces a signal of duration `len(y) / r` seconds — i.e. a rate
greater than 1.0 produces a *shorter* query. The ground-truth
correspondence is therefore:

```
j_gt(i) = i / rate
```

where `i` is the reference frame index and `j` is the query frame
index. This convention is used consistently in all five scripts.
Frame durations used for MAE conversion are:

| Stage | Hop (samples) | Frame duration |
|---|---|---|
| MFCC | 512 | 23.22 ms |
| Mel / Fine | 256 | 11.61 ms |
| Coarse (×4 downsample) | 1024 | 46.44 ms |

---


---

## License

This code is released for research reproducibility under the MIT
License. The GTZAN dataset is subject to its own terms of use; please
consult the original dataset documentation before any commercial use.