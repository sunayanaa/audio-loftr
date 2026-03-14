# ablation_rope_v1.py
#
# Ablation: RoPE (relative) vs Sinusoidal APE (absolute) positional encoding.
#
# Both variants are trained from scratch under IDENTICAL conditions:
#   - Same 80/20 track split (ctf_v5_split.json)
#   - Same training rates [0.9, 1.0, 1.1, 1.2]
#   - Same 50 epochs, AdamW lr=5e-4, cosine LR, batch=4, grad clip=1.0
#   - Same BCE loss on coarse GT matrix (j_gt = i / rate)
#
# Evaluation:
#   - 60 test windows, rate=1.2 (same as v5)
#   - Also multi-rate: 0.8, 0.9, 1.1, 1.2, 1.3, 1.4
#
# Outputs:
#   - ablation_rope_results.json   (per-window MAE cache for both variants)
#   - ablation_rope_figure.png     (bar + line plots)
#   - Console LaTeX table
#
# IMPORTANT: This script trains TWO models sequentially. Each takes ~same
# time as v5 training. Run with GPU. Total runtime ~2× v5 training time.
#
# Checkpoints:
#   model_rope.pth   — RoPE variant
#   model_ape.pth    — APE variant

import os, json, random, math
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist

# ── Colab drive ────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
DRIVE_BASE     = "/content/drive/MyDrive/GTZAN"
DRIVE_DATA     = os.path.join(DRIVE_BASE, "jazz")
DRIVE_SPLIT    = os.path.join(DRIVE_BASE, "ctf_v5_split.json")
CKPT_ROPE      = os.path.join(DRIVE_BASE, "model_rope.pth")
CKPT_APE       = os.path.join(DRIVE_BASE, "model_ape.pth")
DRIVE_RESULTS  = os.path.join(DRIVE_BASE, "ablation_rope_results.json")
DRIVE_FIG      = os.path.join(DRIVE_BASE, "ablation_rope_figure.png")
os.makedirs(DRIVE_BASE, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050
N_MELS         = 128
HOP_LENGTH     = 256
N_MFCC         = 20
HOP_MFCC       = 512
COARSE_SCALE   = 4
TUBE_RADIUS    = 12
CLIP_DUR       = 8.0
CLIP_SAMPLES   = int(CLIP_DUR * SAMPLE_RATE)
TRAIN_RATES    = [0.9, 1.0, 1.1, 1.2]
EVAL_RATES     = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
EPOCHS         = 50
BATCH_SIZE     = 4
SEED           = 42
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FRAME_MS_MFCC   = HOP_MFCC  / SAMPLE_RATE * 1000   # 23.22 ms
FRAME_MS_FINE   = HOP_LENGTH / SAMPLE_RATE * 1000   # 11.61 ms
FRAME_MS_COARSE = HOP_LENGTH * COARSE_SCALE / SAMPLE_RATE * 1000  # 46.44 ms

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device : {DEVICE}")
print(f"Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  Rates : {TRAIN_RATES}")

# ── Positional encoding helpers ────────────────────────────────────────────

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionEmbedding(nn.Module):
    """RoPE — applied inside self-attention to Q and K."""
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t     = torch.arange(x.shape[1], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        return x * emb.cos() + rotate_half(x) * emb.sin()


class SinusoidalAPE(nn.Module):
    """
    Standard sinusoidal absolute positional encoding (Vaswani et al. 2017).
    Added to token embeddings ONCE before the transformer layers.
    Uses the same sin/cos formulation as the original Transformer paper:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        """x: [B, L, D]  →  x + PE[:L]"""
        return x + self.pe[:, :x.shape[1], :]


# ── Shared attention module ────────────────────────────────────────────────

class LoFTRAttention(nn.Module):
    """
    Multi-head attention.
    When rope is not None, RoPE is applied to Q and K before the dot product.
    When rope is None (cross-attention or APE variant), no position applied here.
    """
    def __init__(self, d_model, nhead, rope=None):
        super().__init__()
        self.nhead  = nhead
        self.d_head = d_model // nhead
        self.scale  = self.d_head ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(self, x, src):
        B, L1, D = x.shape; _, L2, _ = src.shape
        q = self.q(x).view(B, L1, self.nhead, self.d_head).transpose(1, 2)
        k = self.k(src).view(B, L2, self.nhead, self.d_head).transpose(1, 2)
        v = self.v(src).view(B, L2, self.nhead, self.d_head).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q.transpose(1,2).reshape(B,L1,D))\
                    .view(B, L1, self.nhead, self.d_head).transpose(1, 2)
            k = self.rope(k.transpose(1,2).reshape(B,L2,D))\
                    .view(B, L2, self.nhead, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        return self.o((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L1, D))


# ── Transformer variants ───────────────────────────────────────────────────

class TransformerRoPE(nn.Module):
    """
    4 self + 4 cross layers.
    RoPE applied to Q, K inside every self-attention layer.
    Cross-attention has no positional encoding (same as original LoFTR).
    """
    def __init__(self, d_model, nhead, layer_names=None):
        super().__init__()
        if layer_names is None:
            layer_names = ['self', 'cross'] * 4
        self.rope   = RotaryPositionEmbedding(d_model)
        self.layers = nn.ModuleList([
            LoFTRAttention(d_model, nhead,
                           rope=self.rope if n == 'self' else None)
            for n in layer_names])
        self.names = layer_names

    def forward(self, f0, f1):
        for i, layer in enumerate(self.layers):
            if self.names[i] == 'self':
                f0 = f0 + layer(f0, f0)
                f1 = f1 + layer(f1, f1)
            else:
                f0 = f0 + layer(f0, f1)
                f1 = f1 + layer(f1, f0)
        return f0, f1


class TransformerAPE(nn.Module):
    """
    4 self + 4 cross layers.
    Sinusoidal APE added ONCE to both sequences before the first layer.
    No positional encoding inside attention (rope=None everywhere).
    This is a strict ablation: the only change from TransformerRoPE is the
    positional encoding mechanism.
    """
    def __init__(self, d_model, nhead, layer_names=None):
        super().__init__()
        if layer_names is None:
            layer_names = ['self', 'cross'] * 4
        self.ape    = SinusoidalAPE(d_model)
        self.layers = nn.ModuleList([
            LoFTRAttention(d_model, nhead, rope=None)
            for _ in layer_names])
        self.names = layer_names

    def forward(self, f0, f1):
        # APE added once at the input — position information is injected
        # into the token embeddings and propagates through all layers
        f0 = self.ape(f0)
        f1 = self.ape(f1)
        for i, layer in enumerate(self.layers):
            if self.names[i] == 'self':
                f0 = f0 + layer(f0, f0)
                f1 = f1 + layer(f1, f1)
            else:
                f0 = f0 + layer(f0, f1)
                f1 = f1 + layer(f1, f0)
        return f0, f1


# ── Full model parameterised by PE type ───────────────────────────────────

class AudioLoFTR(nn.Module):
    def __init__(self, d_model=128, nhead=4, temperature=0.1, pe='rope'):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        if pe == 'rope':
            self.transformer = TransformerRoPE(d_model, nhead)
        elif pe == 'ape':
            self.transformer = TransformerAPE(d_model, nhead)
        else:
            raise ValueError(f"Unknown pe: {pe}")
        self.T = temperature

    def _match(self, f0, f1):
        f0 = F.normalize(f0, dim=-1)
        f1 = F.normalize(f1, dim=-1)
        sim = torch.einsum("bmd,bnd->bmn", f0, f1) / self.T
        return F.softmax(sim, dim=1) * F.softmax(sim, dim=2)

    def forward(self, img0, img1):
        x0 = self.enc(img0).mean(dim=2).transpose(1, 2)
        x1 = self.enc(img1).mean(dim=2).transpose(1, 2)
        f0, f1 = self.transformer(x0, x1)
        return {'conf_matrix': self._match(f0, f1)}


# ── Audio / feature helpers ────────────────────────────────────────────────

def audio_to_mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                        n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(m, ref=np.max)

def mel_db_to_tensor(m_db):
    m = (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)
    return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

def load_window(track_fname, start_sample):
    path = os.path.join(DRIVE_DATA, track_fname)
    y, _ = librosa.load(path, sr=SAMPLE_RATE,
                         offset=start_sample / SAMPLE_RATE,
                         duration=CLIP_DUR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)

def calc_mae(path, rate, frame_ms):
    i    = path[:, 0].astype(np.float64)
    j    = path[:, 1].astype(np.float64)
    j_gt = i / rate
    return float(np.mean(np.abs(j - j_gt)) * frame_ms)

def coarse_align(model, y_ref, y_qry):
    with torch.no_grad():
        conf = model(
            mel_db_to_tensor(audio_to_mel_db(y_ref)),
            mel_db_to_tensor(audio_to_mel_db(y_qry))
        )['conf_matrix'][0].cpu().numpy()
    cost = -np.log(conf + 1e-8)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()

def mfcc_dtw(y_ref, y_qry):
    m1 = librosa.feature.mfcc(y=y_ref, sr=SAMPLE_RATE,
                                n_mfcc=N_MFCC, hop_length=HOP_MFCC)
    m2 = librosa.feature.mfcc(y=y_qry, sr=SAMPLE_RATE,
                                n_mfcc=N_MFCC, hop_length=HOP_MFCC)
    cost = cdist(m1.T, m2.T, metric='euclidean').astype(np.float32)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()

def spec_dtw(y_ref, y_qry):
    s1 = audio_to_mel_db(y_ref)
    s2 = audio_to_mel_db(y_qry)
    cost = cdist(s1.T, s2.T, metric='cosine').astype(np.float32)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()


# ── Dataset ────────────────────────────────────────────────────────────────

class JazzWindowDataset(Dataset):
    def __init__(self, windows, rates=TRAIN_RATES):
        self.items = [(w, r) for w in windows for r in rates]
        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        (track, start), rate = self.items[idx]
        try:
            y_ref   = load_window(track, start)
            y_qry   = librosa.effects.time_stretch(y_ref, rate=rate)

            def norm(m):
                return (m - m.min()) / (m.max() - m.min() + 1e-8)

            t_ref = torch.from_numpy(norm(audio_to_mel_db(y_ref))).float().unsqueeze(0)
            t_qry = torch.from_numpy(norm(audio_to_mel_db(y_qry))).float().unsqueeze(0)

            Wc_ref = t_ref.shape[2] // COARSE_SCALE
            Wc_qry = t_qry.shape[2] // COARSE_SCALE
            gt = torch.zeros(Wc_ref, Wc_qry)
            for i in range(Wc_ref):
                j = int(round(i / rate))
                if 0 <= j < Wc_qry:
                    gt[i, j] = 1.0

            return t_ref, t_qry, gt, float(rate)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


def collate_pad(batch):
    max_w_ref = max(b[0].shape[2] for b in batch)
    max_w_qry = max(b[1].shape[2] for b in batch)
    max_wc_r  = max(b[2].shape[0] for b in batch)
    max_wc_q  = max(b[2].shape[1] for b in batch)
    t_refs, t_qrys, gts, rates = [], [], [], []
    for t_r, t_q, gt, rate in batch:
        t_refs.append(F.pad(t_r, (0, max_w_ref - t_r.shape[2])))
        t_qrys.append(F.pad(t_q, (0, max_w_qry - t_q.shape[2])))
        gts.append(F.pad(gt,   (0, max_wc_q - gt.shape[1],
                                 0, max_wc_r - gt.shape[0])))
        rates.append(rate)
    return torch.stack(t_refs), torch.stack(t_qrys), torch.stack(gts), rates


# ── Generic training function ──────────────────────────────────────────────

def train_model(pe_type, ckpt_path, train_windows):
    model = AudioLoFTR(pe=pe_type).to(DEVICE)
    start_ep = 0

    if os.path.exists(ckpt_path):
        ckpt     = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        start_ep = ckpt.get('epoch', EPOCHS)
        print(f"  [{pe_type.upper()}] Checkpoint loaded (epoch {start_ep})")
    else:
        print(f"  [{pe_type.upper()}] No checkpoint — training from scratch")

    if start_ep >= EPOCHS:
        print(f"  [{pe_type.upper()}] Already trained to {EPOCHS} epochs, skipping.")
        model.eval()
        return model

    dataset = JazzWindowDataset(train_windows)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=collate_pad, num_workers=0)
    opt   = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=EPOCHS - start_ep, eta_min=1e-5)
    crit  = nn.BCELoss()

    model.train()
    for ep in range(start_ep, EPOCHS):
        tot = 0.0
        for t_ref, t_qry, gt, _ in loader:
            t_ref = t_ref.to(DEVICE)
            t_qry = t_qry.to(DEVICE)
            gt    = gt.to(DEVICE)
            opt.zero_grad()
            conf = model(t_ref, t_qry)['conf_matrix']
            # Crop to min size in case of rounding differences
            min_r = min(conf.shape[1], gt.shape[1])
            min_c = min(conf.shape[2], gt.shape[2])
            loss  = crit(conf[:, :min_r, :min_c], gt[:, :min_r, :min_c])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        avg = tot / len(loader)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{pe_type.upper()}] Epoch {ep+1:3d}/{EPOCHS}  loss={avg:.4f}")
        # Save checkpoint every 10 epochs
        if (ep + 1) % 10 == 0:
            torch.save({'model': model.state_dict(), 'epoch': ep + 1}, ckpt_path)
            print(f"  [{pe_type.upper()}] Checkpoint saved → {ckpt_path}")

    torch.save({'model': model.state_dict(), 'epoch': EPOCHS}, ckpt_path)
    print(f"  [{pe_type.upper()}] Training complete → {ckpt_path}")
    model.eval()
    return model


# ── Load split ─────────────────────────────────────────────────────────────
assert os.path.exists(DRIVE_SPLIT), \
    f"Split file not found: {DRIVE_SPLIT}\nRun demo_coarse_to_fine_v5.py first."

with open(DRIVE_SPLIT) as f:
    split = json.load(f)
train_tracks = split['train']
test_tracks  = split['test']
print(f"Split: {len(train_tracks)} train / {len(test_tracks)} test tracks")

audio_exts = ('.au', '.wav', '.mp3', '.flac')

def get_windows(track_fname):
    path = os.path.join(DRIVE_DATA, track_fname)
    dur  = librosa.get_duration(path=path)
    wins = []; start = 0.0
    while start + CLIP_DUR <= dur:
        wins.append((track_fname, int(start * SAMPLE_RATE)))
        start += CLIP_DUR
    return wins

train_windows = []
for t in train_tracks:
    train_windows.extend(get_windows(t))
test_windows = []
for t in test_tracks:
    test_windows.extend(get_windows(t))
print(f"Train windows : {len(train_windows)}  |  Test windows : {len(test_windows)}")


# ── Train both models ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Train RoPE model")
print("="*60)
model_rope = train_model('rope', CKPT_ROPE, train_windows)

print("\n" + "="*60)
print("STEP 2: Train APE model")
print("="*60)
model_ape = train_model('ape', CKPT_APE, train_windows)


# ── Evaluation ─────────────────────────────────────────────────────────────
# Load cache
if os.path.exists(DRIVE_RESULTS):
    with open(DRIVE_RESULTS) as f:
        results = json.load(f)
    print(f"\nCache loaded: {DRIVE_RESULTS}")
else:
    results = {}

def save_cache():
    with open(DRIVE_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

METHODS = ['mfcc', 'spec', 'rope', 'ape']
METHOD_LABELS = {
    'mfcc': 'MFCC + DTW',
    'spec': 'Spec + DTW',
    'rope': 'LoFTR + RoPE  (proposed)',
    'ape':  'LoFTR + APE   (ablation)',
}

# Ensure all keys exist
for rate in EVAL_RATES:
    rkey = str(rate)
    if rkey not in results:
        results[rkey] = {m: {} for m in METHODS}
    for m in METHODS:
        if m not in results[rkey]:
            results[rkey][m] = {}

print("\n" + "="*60)
print("STEP 3: Evaluate on test windows")
print("="*60)

for rate in EVAL_RATES:
    rkey = str(rate)
    n_done = len(results[rkey]['rope'])
    print(f"\n── Rate x{rate}  ({n_done}/{len(test_windows)} cached) ────────────")

    for (track, start) in test_windows:
        wkey = f"{track}@{start}"
        # Skip only if ALL methods have results for this window at this rate
        if all(wkey in results[rkey][m] for m in METHODS):
            continue
        try:
            y_ref = load_window(track, start)
            y_qry_raw = librosa.effects.time_stretch(y_ref, rate=rate)

            # Baselines
            if wkey not in results[rkey]['mfcc']:
                path_m = mfcc_dtw(y_ref, y_qry_raw)
                results[rkey]['mfcc'][wkey] = calc_mae(path_m, rate, FRAME_MS_MFCC)

            if wkey not in results[rkey]['spec']:
                path_s = spec_dtw(y_ref, y_qry_raw)
                results[rkey]['spec'][wkey] = calc_mae(path_s, rate, FRAME_MS_FINE)

            # LoFTR variants — coarse alignment only
            if wkey not in results[rkey]['rope']:
                path_r = coarse_align(model_rope, y_ref, y_qry_raw)
                results[rkey]['rope'][wkey] = calc_mae(path_r, rate, FRAME_MS_COARSE)

            if wkey not in results[rkey]['ape']:
                path_a = coarse_align(model_ape, y_ref, y_qry_raw)
                results[rkey]['ape'][wkey] = calc_mae(path_a, rate, FRAME_MS_COARSE)

            save_cache()

            r_mae = results[rkey]['rope'][wkey]
            a_mae = results[rkey]['ape'][wkey]
            print(f"  {track} +{start//SAMPLE_RATE:2.0f}s | "
                  f"RoPE={r_mae:.1f}  APE={a_mae:.1f}  ms")

        except Exception as e:
            print(f"  ERROR {wkey} rate={rate}: {e}")
            continue


# ── Aggregate results ──────────────────────────────────────────────────────
mean_t = {m: {} for m in METHODS}
std_t  = {m: {} for m in METHODS}
for rate in EVAL_RATES:
    rkey = str(rate)
    for m in METHODS:
        vals = list(results[rkey][m].values())
        mean_t[m][rate] = float(np.mean(vals)) if vals else float('nan')
        std_t[m][rate]  = float(np.std(vals))  if vals else float('nan')


# ── Console table ──────────────────────────────────────────────────────────
sep = "="*80
print(f"\n{sep}")
print("RoPE vs APE Ablation  —  Coarse alignment MAE (ms)")
print(f"{'Method':<30}" + "".join(f"{r:>8.1f}×" for r in EVAL_RATES))
print("-"*80)
for m in METHODS:
    row = f"{METHOD_LABELS[m]:<30}"
    for r in EVAL_RATES:
        row += f"{mean_t[m][r]:>9.1f}"
    print(row)
print(sep)

# Relative comparison: RoPE vs APE at each rate
print("\nRoPE vs APE (positive = RoPE better):")
for r in EVAL_RATES:
    delta = mean_t['ape'][r] - mean_t['rope'][r]
    winner = "RoPE better" if delta > 0 else "APE better " if delta < 0 else "equal"
    print(f"  x{r}: {delta:+.1f} ms  ({winner})")

# Worst-rate degradation relative to x1.2 (R2's concern about r=0.8)
ref_rate = 1.2
print(f"\nDegradation relative to x{ref_rate} baseline:")
for m in ['rope', 'ape']:
    ref_mae = mean_t[m].get(ref_rate, float('nan'))
    worst   = max(mean_t[m][r] for r in EVAL_RATES)
    worst_r = max(EVAL_RATES, key=lambda r: mean_t[m][r])
    print(f"  {METHOD_LABELS[m]}: "
          f"best={ref_mae:.1f}ms @ x{ref_rate},  "
          f"worst={worst:.1f}ms @ x{worst_r}")


# ── LaTeX table ────────────────────────────────────────────────────────────
n_wins = len(test_windows); n_trk = len(test_tracks)
rate_hdrs = " & ".join(f"\\textbf{{${r:.1f}\\times$}}" for r in EVAL_RATES)

latex  = f"\n\\begin{{table}}[t]\n"
latex += (f"\\caption{{RoPE vs.\\ APE ablation --- coarse-stage MAE (ms) under "
          f"time-stretch. {n_wins} windows from {n_trk} test tracks, "
          f"{CLIP_DUR:.0f}\\,s clips.}}\n")
latex += "\\centering\n"
latex += "\\begin{tabular}{l" + "c"*len(EVAL_RATES) + "}\n\\hline\n"
latex += f"\\textbf{{Method}} & {rate_hdrs} \\\\\n\\hline\n"
for m in METHODS:
    cells = [f"{mean_t[m][r]:.1f}" for r in EVAL_RATES]
    if m == 'rope':
        label = f"\\textbf{{{METHOD_LABELS[m]}}}"
    else:
        label = METHOD_LABELS[m]
    latex += f"{label} & " + " & ".join(cells) + " \\\\\n"
    if m == 'spec':
        latex += "\\hline\n"
latex += "\\hline\n\\end{tabular}\n\\label{tab:rope_ablation}\n\\end{table}\n"
print(latex)


# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"RoPE vs APE Ablation — coarse-stage MAE  "
    f"({len(test_windows)} windows, {len(test_tracks)} tracks)",
    fontsize=12)

colors  = {'mfcc': '#7090C0', 'spec': '#70C080',
           'rope': '#D65F5F', 'ape':  '#E09020'}
markers = {'mfcc': 'o', 'spec': 's', 'rope': '^', 'ape': 'D'}
lwidths = {'mfcc': 1.2, 'spec': 1.2, 'rope': 2.5, 'ape': 2.0}
lstyles = {'mfcc': '--', 'spec': '--', 'rope': '-', 'ape': '-'}

rates_x = EVAL_RATES

# Panel 1: all methods
ax = axes[0]
for m in METHODS:
    ys = [mean_t[m][r] for r in rates_x]
    es = [std_t[m][r]  for r in rates_x]
    ax.errorbar(rates_x, ys, yerr=es,
                label=METHOD_LABELS[m],
                color=colors[m], marker=markers[m],
                linewidth=lwidths[m], linestyle=lstyles[m],
                markersize=7, capsize=3)
ax.set_xlabel('Time-stretch rate')
ax.set_ylabel('Mean MAE (ms)')
ax.set_title('All Methods')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xticks(rates_x)

# Panel 2: RoPE vs APE only (zoom in)
ax = axes[1]
for m in ['rope', 'ape']:
    ys = [mean_t[m][r] for r in rates_x]
    es = [std_t[m][r]  for r in rates_x]
    ax.errorbar(rates_x, ys, yerr=es,
                label=METHOD_LABELS[m],
                color=colors[m], marker=markers[m],
                linewidth=lwidths[m], linestyle=lstyles[m],
                markersize=8, capsize=3)
ax.set_xlabel('Time-stretch rate')
ax.set_ylabel('Mean MAE (ms)')
ax.set_title('LoFTR Coarse: RoPE vs APE (zoom)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(rates_x)

# Annotate the delta at each point
for r in rates_x:
    delta = mean_t['ape'][r] - mean_t['rope'][r]
    y_mid = (mean_t['rope'][r] + mean_t['ape'][r]) / 2
    color = '#2a7' if delta > 0 else '#c33'
    ax.annotate(f"{delta:+.0f}ms",
                xy=(r, y_mid), fontsize=7, ha='center', color=color,
                xytext=(0, 8), textcoords='offset points')

plt.tight_layout()
plt.savefig(DRIVE_FIG, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {DRIVE_FIG}")
print("Done.")
