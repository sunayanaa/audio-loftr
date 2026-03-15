# demo_coarse_to_fine_v5.py  —  Path A: 8-second windowed evaluation
#
# Design:
#   - Extract non-overlapping 8s windows from each jazz track
#   - Track-level 80/20 split (no track leakage across train/test)
#   - Train on windows from 80 train tracks with rate augmentation
#   - Evaluate on windows from 20 test tracks at rate=1.2
#   - Query is NOT trimmed — reference (8s) maps into longer query
#   - GT: j_gt = i * rate  (all ref frames have valid GT in longer query)
#   - Baselines: MFCC+DTW, Spec+DTW at correct frame resolutions
#   - Full resume: checkpoint saved every epoch, results cached per-window

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
from datetime import datetime


# ── Colab drive ────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
DRIVE_BASE    = "/content/drive/MyDrive/audio-loftr"
DRIVE_DATA    = os.path.join(DRIVE_BASE, "jazz")
DRIVE_CKPT    = os.path.join(DRIVE_BASE, "model_ctf_v5.pth")
DRIVE_SPLIT   = os.path.join(DRIVE_BASE, "ctf_v5_split.json")
DRIVE_RESULTS = os.path.join(DRIVE_BASE, "ctf_v5_results.json")
DRIVE_FIG     = os.path.join(DRIVE_BASE, "ctf_v5_figure.png")
os.makedirs(DRIVE_BASE, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
N_MELS        = 128
HOP_LENGTH    = 256
N_MFCC        = 20
HOP_MFCC      = 512           # librosa default — used only for MFCC baseline
COARSE_SCALE  = 4
TUBE_RADIUS   = 12
CLIP_DUR      = 8.0           # seconds per window
STRIDE_DUR    = 8.0           # non-overlapping
EVAL_RATE     = 1.2
TRAIN_RATES   = [0.9, 1.0, 1.1, 1.2]
EPOCHS        = 50
BATCH_SIZE    = 4
TRAIN_FRAC    = 0.80
SEED          = 42
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Correct frame durations
FRAME_MS_MFCC   = HOP_MFCC   / SAMPLE_RATE * 1000    # 23.22 ms
FRAME_MS_FINE   = HOP_LENGTH  / SAMPLE_RATE * 1000    # 11.61 ms
FRAME_MS_COARSE = HOP_LENGTH  * COARSE_SCALE / SAMPLE_RATE * 1000  # 46.44 ms

CLIP_SAMPLES  = int(CLIP_DUR * SAMPLE_RATE)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"Device            : {DEVICE}")
print(f"Clip duration     : {CLIP_DUR}s  ({CLIP_SAMPLES} samples)")
print(f"FRAME_MS_MFCC     : {FRAME_MS_MFCC:.2f} ms")
print(f"FRAME_MS_FINE     : {FRAME_MS_FINE:.2f} ms")
print(f"FRAME_MS_COARSE   : {FRAME_MS_COARSE:.2f} ms")

# ── Discover tracks and build window index ─────────────────────────────────
audio_exts = ('.au', '.wav', '.mp3', '.flac')
all_tracks  = sorted([f for f in os.listdir(DRIVE_DATA)
                       if f.lower().endswith(audio_exts)])
assert len(all_tracks) >= 10, f"Too few tracks: {len(all_tracks)}"

def get_windows(track_fname):
    """Return list of (track_fname, start_sample) for each 8s window."""
    path = os.path.join(DRIVE_DATA, track_fname)
    dur  = librosa.get_duration(path=path)
    wins = []
    start = 0.0
    while start + CLIP_DUR <= dur:
        wins.append((track_fname, int(start * SAMPLE_RATE)))
        start += STRIDE_DUR
    return wins

# ── Track-level split (stable — saved to Drive) ───────────────────────────
if os.path.exists(DRIVE_SPLIT):
    with open(DRIVE_SPLIT) as f:
        split = json.load(f)
    train_tracks = split['train']
    test_tracks  = split['test']
    print(f"Loaded split : {len(train_tracks)} train / {len(test_tracks)} test tracks")
else:
    rng = random.Random(SEED)
    sh  = all_tracks[:]
    rng.shuffle(sh)
    n_tr         = int(len(sh) * TRAIN_FRAC)
    train_tracks = sh[:n_tr]
    test_tracks  = sh[n_tr:]
    with open(DRIVE_SPLIT, 'w') as f:
        json.dump({'train': train_tracks, 'test': test_tracks}, f, indent=2)
    print(f"New split    : {len(train_tracks)} train / {len(test_tracks)} test tracks")

# Build window lists
print("Scanning tracks for windows...")
train_windows = []
for t in train_tracks:
    train_windows.extend(get_windows(t))
test_windows = []
for t in test_tracks:
    test_windows.extend(get_windows(t))
print(f"Train windows: {len(train_windows)}  |  Test windows: {len(test_windows)}")

# ── Model ──────────────────────────────────────────────────────────────────
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x):
        t     = torch.arange(x.shape[1], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        return x * emb.cos() + rotate_half(x) * emb.sin()

class LoFTRAttention(nn.Module):
    def __init__(self, d_model, nhead, rope=None):
        super().__init__()
        self.nhead  = nhead
        self.d_head = d_model // nhead
        self.scale  = self.d_head ** -0.5
        self.q  = nn.Linear(d_model, d_model, bias=False)
        self.k  = nn.Linear(d_model, d_model, bias=False)
        self.v  = nn.Linear(d_model, d_model, bias=False)
        self.o  = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope
    def forward(self, x, src):
        B, L1, D = x.shape; _, L2, _ = src.shape
        q = self.q(x).view(B,L1,self.nhead,self.d_head).transpose(1,2)
        k = self.k(src).view(B,L2,self.nhead,self.d_head).transpose(1,2)
        v = self.v(src).view(B,L2,self.nhead,self.d_head).transpose(1,2)
        if self.rope:
            q = self.rope(q.transpose(1,2).reshape(B,L1,D))\
                    .view(B,L1,self.nhead,self.d_head).transpose(1,2)
            k = self.rope(k.transpose(1,2).reshape(B,L2,D))\
                    .view(B,L2,self.nhead,self.d_head).transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        return self.o((attn.softmax(dim=-1) @ v).transpose(1,2).reshape(B,L1,D))

class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model, nhead, layer_names=['self','cross']*4):
        super().__init__()
        self.rope   = RotaryPositionEmbedding(d_model)
        self.layers = nn.ModuleList([
            LoFTRAttention(d_model, nhead,
                           rope=self.rope if n=='self' else None)
            for n in layer_names])
        self.names  = layer_names
    def forward(self, f0, f1):
        for i, layer in enumerate(self.layers):
            if self.names[i] == 'self':
                f0 = f0 + layer(f0, f0); f1 = f1 + layer(f1, f1)
            else:
                f0 = f0 + layer(f0, f1); f1 = f1 + layer(f1, f0)
        return f0, f1

class AudioLoFTR(nn.Module):
    def __init__(self, d_model=128, nhead=4, temperature=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        self.transformer = LocalFeatureTransformer(d_model, nhead)
        self.T = temperature

    def _match(self, f0, f1):
        f0 = F.normalize(f0, dim=-1)
        f1 = F.normalize(f1, dim=-1)
        sim = torch.einsum("bmd,bnd->bmn", f0, f1) / self.T
        return F.softmax(sim, dim=1) * F.softmax(sim, dim=2)

    def forward(self, img0, img1):
        x0 = self.enc(img0).mean(dim=2).transpose(1, 2)  # freq-pool→[B,Wc,D]
        x1 = self.enc(img1).mean(dim=2).transpose(1, 2)
        f0, f1 = self.transformer(x0, x1)
        return {'conf_matrix': self._match(f0, f1)}

# ── Spectrogram helpers ────────────────────────────────────────────────────
def audio_to_mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                       n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(m, ref=np.max)   # [N_MELS, T]

def mel_db_to_tensor(m_db):
    m = (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)
    return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

def load_window(track_fname, start_sample):
    """Load exactly CLIP_SAMPLES of audio starting at start_sample."""
    path = os.path.join(DRIVE_DATA, track_fname)
    offset = start_sample / SAMPLE_RATE
    y, _  = librosa.load(path, sr=SAMPLE_RATE, offset=offset,
                          duration=CLIP_DUR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)

# ── Training dataset ───────────────────────────────────────────────────────
class JazzWindowDataset(Dataset):
    def __init__(self, windows, rates=TRAIN_RATES):
        # Each (track, start) × each rate = one sample
        self.items = [(w, r) for w in windows for r in rates]
        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        (track, start), rate = self.items[idx]
        try:
            y_ref = load_window(track, start)

            # Stretch reference → query is LONGER than reference
            y_qry = librosa.effects.time_stretch(y_ref, rate=rate)
            # Do NOT trim query — keep full stretched length

            mel_ref = audio_to_mel_db(y_ref)
            mel_qry = audio_to_mel_db(y_qry)

            # Normalise
            def norm(m):
                return (m - m.min()) / (m.max() - m.min() + 1e-8)

            t_ref = torch.from_numpy(norm(mel_ref)).float().unsqueeze(0)
            t_qry = torch.from_numpy(norm(mel_qry)).float().unsqueeze(0)

            # GT coarse matrix
            # ref frame i → query frame j_gt = round(i * rate)
            # Query is longer so all j_gt are within bounds
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
    """Pad to max width in batch (ref and qry may differ in T)."""
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
    return (torch.stack(t_refs), torch.stack(t_qrys),
            torch.stack(gts),    rates)

# ── Train or resume ────────────────────────────────────────────────────────
model    = AudioLoFTR().to(DEVICE)
start_ep = 0

if os.path.exists(DRIVE_CKPT):
    ckpt     = torch.load(DRIVE_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    start_ep = ckpt.get('epoch', EPOCHS)
    print(f" Checkpoint loaded (epoch {start_ep}): {DRIVE_CKPT}")
else:
    print("No checkpoint — training from scratch.")

if start_ep < EPOCHS:
    print(f"\nTraining: {len(train_windows)} windows × {len(TRAIN_RATES)} rates"
          f" = {len(train_windows)*len(TRAIN_RATES)} samples")
    print(f"Epochs {start_ep}→{EPOCHS}, batch={BATCH_SIZE}, device={DEVICE}")

    dataset = JazzWindowDataset(train_windows)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=collate_pad, num_workers=0)
    opt     = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=EPOCHS - start_ep, eta_min=1e-5)
    crit    = nn.BCELoss()

    model.train()
    for ep in range(start_ep, EPOCHS):
        tot = 0.0
        for t_ref, t_qry, gt, _ in loader:
            t_ref = t_ref.to(DEVICE)
            t_qry = t_qry.to(DEVICE)
            gt    = gt.to(DEVICE)
            opt.zero_grad()
            conf  = model(t_ref, t_qry)['conf_matrix']
            mh    = min(conf.shape[1], gt.shape[1])
            mw    = min(conf.shape[2], gt.shape[2])
            loss  = crit(conf[:, :mh, :mw], gt[:, :mh, :mw])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot  += loss.item()
        sched.step()
        avg = tot / max(len(loader), 1)
        lr  = sched.get_last_lr()[0]
        current_datetime = datetime.now()
        print(current_datetime.strftime("%d-%m-%Y %H:%M:%S"), end="")
        print(f"  Epoch {ep+1:3d}/{EPOCHS}: loss={avg:.4f}  lr={lr:.2e}")
        # Save every epoch — safe against disconnects
        torch.save({'model': model.state_dict(), 'epoch': ep + 1}, DRIVE_CKPT)

    print(f"Training done. Checkpoint: {DRIVE_CKPT}")

model.eval()

# ── Alignment functions ────────────────────────────────────────────────────
def coarse_align(y_ref, y_qry):
    with torch.no_grad():
        conf = model(
            mel_db_to_tensor(audio_to_mel_db(y_ref)),
            mel_db_to_tensor(audio_to_mel_db(y_qry))
        )['conf_matrix'][0].cpu().numpy()
    cost = -np.log(conf + 1e-8)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy(), conf

def fine_align(y_ref, y_qry, path_coarse):
    """Single global DTW through tube mask built from coarse path."""
    spec_ref = audio_to_mel_db(y_ref)   # [N_MELS, T_ref]
    spec_qry = audio_to_mel_db(y_qry)   # [N_MELS, T_qry]  (T_qry > T_ref)
    T_ref = spec_ref.shape[1]
    T_qry = spec_qry.shape[1]

    tube = np.full((T_ref, T_qry), 10.0, dtype=np.float32)

    for ic, jc in path_coarse:
        i_fine = ic * COARSE_SCALE
        j_fine = jc * COARSE_SCALE
        i0 = max(0, i_fine - TUBE_RADIUS)
        i1 = min(T_ref, i_fine + TUBE_RADIUS + 1)
        j0 = max(0, j_fine - TUBE_RADIUS)
        j1 = min(T_qry, j_fine + TUBE_RADIUS + 1)
        local = cdist(spec_ref[:, i0:i1].T,
                      spec_qry[:, j0:j1].T,
                      metric='cosine').astype(np.float32)
        tube[i0:i1, j0:j1] = np.minimum(tube[i0:i1, j0:j1], local)

    _, wp = librosa.sequence.dtw(C=tube, backtrack=True)
    return wp[::-1].copy()

def mfcc_dtw(y_ref, y_qry):
    m1   = librosa.feature.mfcc(y=y_ref, sr=SAMPLE_RATE,
                                 n_mfcc=N_MFCC, hop_length=HOP_MFCC)
    m2   = librosa.feature.mfcc(y=y_qry, sr=SAMPLE_RATE,
                                 n_mfcc=N_MFCC, hop_length=HOP_MFCC)
    cost = cdist(m1.T, m2.T, metric='euclidean').astype(np.float32)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()

def spec_dtw(y_ref, y_qry):
    s1   = audio_to_mel_db(y_ref)
    s2   = audio_to_mel_db(y_qry)
    cost = cdist(s1.T, s2.T, metric='cosine').astype(np.float32)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()

def calc_mae(path, rate, frame_ms):
    """
    MAE in ms.
    ref frame i → GT query frame j_gt = i / rate
    Query is longer than ref so all i have valid GT.
    path[:,0] = ref indices, path[:,1] = query indices.
    """
    i    = path[:, 0].astype(np.float64)
    j    = path[:, 1].astype(np.float64)
    j_gt = i / rate
    return float(np.mean(np.abs(j - j_gt)) * frame_ms)

# ── Load cached results ────────────────────────────────────────────────────
if os.path.exists(DRIVE_RESULTS):
    with open(DRIVE_RESULTS) as f:
        cache = json.load(f)
    done         = set(cache['done'])
    res_mfcc     = cache['mfcc']
    res_spec     = cache['spec']
    res_coarse   = cache['coarse']
    res_fine     = cache['fine']
    print(f"Cache loaded: {len(done)}/{len(test_windows)} windows done")
else:
    done       = set()
    res_mfcc   = {}
    res_spec   = {}
    res_coarse = {}
    res_fine   = {}

def save_cache():
    with open(DRIVE_RESULTS, 'w') as f:
        json.dump({'done':   list(done),
                   'mfcc':   res_mfcc,
                   'spec':   res_spec,
                   'coarse': res_coarse,
                   'fine':   res_fine}, f, indent=2)

# ── Evaluation loop ────────────────────────────────────────────────────────
print(f"\nEvaluating {len(test_windows)} test windows "
      f"(rate={EVAL_RATE}, clip={CLIP_DUR}s)...")

last = {}

for (track, start) in test_windows:
    key = f"{track}@{start}"
    if key in done:
        continue

    try:
        y_ref = load_window(track, start)
        # Query is NOT trimmed — naturally longer
        y_qry = librosa.effects.time_stretch(y_ref, rate=EVAL_RATE)

        path_m = mfcc_dtw(y_ref, y_qry)
        mae_m  = calc_mae(path_m, EVAL_RATE, FRAME_MS_MFCC)

        path_s = spec_dtw(y_ref, y_qry)
        mae_s  = calc_mae(path_s, EVAL_RATE, FRAME_MS_FINE)

        path_c, conf = coarse_align(y_ref, y_qry)
        mae_c  = calc_mae(path_c, EVAL_RATE, FRAME_MS_COARSE)

        path_f = fine_align(y_ref, y_qry, path_c)
        mae_f  = calc_mae(path_f, EVAL_RATE, FRAME_MS_FINE)

        res_mfcc[key]   = mae_m
        res_spec[key]   = mae_s
        res_coarse[key] = mae_c
        res_fine[key]   = mae_f
        done.add(key)
        save_cache()

        print(f"  {track} +{start//SAMPLE_RATE:3.0f}s | "
              f"MFCC={mae_m:.0f}  Spec={mae_s:.0f}  "
              f"Coarse={mae_c:.0f}  Fine={mae_f:.0f}  ms")

        last = dict(key=key, conf=conf,
                    path_c=path_c, path_f=path_f,
                    path_s=path_s, path_m=path_m,
                    y_ref=y_ref, y_qry=y_qry)

    except Exception as e:
        print(f"  ERROR {key}: {e}")
        continue

# ── Aggregate ──────────────────────────────────────────────────────────────
def agg(d): return list(d.values())
m_v = agg(res_mfcc); s_v = agg(res_spec)
c_v = agg(res_coarse); f_v = agg(res_fine)

def stats(v):
    return np.mean(v), np.std(v), np.median(v)

m_m, m_s, m_med = stats(m_v)
s_m, s_s, s_med = stats(s_v)
c_m, c_s, c_med = stats(c_v)
f_m, f_s, f_med = stats(f_v)

reduction = (1 - f_m / c_m) * 100 if c_m > 0 else 0.0

print("\n" + "="*68)
print(f"Windows evaluated : {len(m_v)}  "
      f"(from {len(test_tracks)} test tracks, {CLIP_DUR}s clips)")
print(f"{'Method':<30} {'Mean':>8} {'Std':>8} {'Median':>8}  ms")
print("-"*68)
print(f"{'MFCC + DTW':30} {m_m:8.1f} {m_s:8.1f} {m_med:8.1f}")
print(f"{'Spec + DTW':30} {s_m:8.1f} {s_s:8.1f} {s_med:8.1f}")
print(f"{'Audio LoFTR  Coarse':30} {c_m:8.1f} {c_s:8.1f} {c_med:8.1f}")
print(f"{'Audio LoFTR  Fine':30} {f_m:8.1f} {f_s:8.1f} {f_med:8.1f}")
print(f"Coarse→Fine reduction : {reduction:.1f}%")
print("="*68)

# ── LaTeX ─────────────────────────────────────────────────────────────────
def fmt(m, s): return f"{m:.1f} $\\pm$ {s:.1f}"
n_tracks = len(test_tracks)
n_wins   = len(m_v)

latex = f"""
\\begin{{table}}[h]
\\caption{{Alignment MAE on Real Jazz (N={n_wins} windows from {n_tracks} tracks,
          {CLIP_DUR:.0f}\\,s clips, rate={EVAL_RATE})}}
\\centering
\\begin{{tabular}}{{l c c}}
\\hline
\\textbf{{Method}} & \\textbf{{MAE (ms)}} & \\textbf{{Median (ms)}}\\\\
\\hline
MFCC + DTW                         & {fmt(m_m,m_s)} & {m_med:.1f} \\\\
Spec + DTW                         & {fmt(s_m,s_s)} & {s_med:.1f} \\\\
Audio LoFTR --- Coarse only        & {fmt(c_m,c_s)} & {c_med:.1f} \\\\
\\textbf{{Audio LoFTR --- Coarse+Fine}} & \\textbf{{{fmt(f_m,f_s)}}} & \\textbf{{{f_med:.1f}}} \\\\
\\hline
\\end{{tabular}}
\\label{{tab:main_results}}
\\end{{table}}
"""
print(latex)

# ── Figure ─────────────────────────────────────────────────────────────────
if last:
    conf    = last['conf']
    path_c  = last['path_c']
    path_f  = last['path_f']
    y_ref   = last['y_ref']
    y_qry   = last['y_qry']
    spec_ref = audio_to_mel_db(y_ref)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"Path A — 8s clip  |  rate={EVAL_RATE}  |  {last['key']}",
                 fontsize=10)

    # Panel 1: coarse attention + path + GT
    ax = axes[0]
    im = ax.imshow(conf, origin='lower', aspect='auto', cmap='magma')
    ax.plot(path_c[:, 1], path_c[:, 0], 'c-', lw=1.5,
            label=f"Coarse MAE={res_coarse[last['key']]:.0f}ms")
    gt_j = np.arange(conf.shape[0]) / EVAL_RATE
    ax.plot(gt_j, np.arange(conf.shape[0]), 'r--', lw=1.5, alpha=0.8,
            label='GT')
    ax.set_title('Stage 1: Coarse Attention', fontsize=11)
    ax.set_xlabel('Query (coarse frames)')
    ax.set_ylabel('Ref (coarse frames)')
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: fine path on reference spectrogram
    ax = axes[1]
    ax.imshow(spec_ref, origin='lower', aspect='auto', cmap='viridis',
              extent=[0, spec_ref.shape[1], 0, N_MELS])
    step = max(1, len(path_f) // 600)
    pf   = path_f[::step]
    ax.scatter(pf[:, 0],
               pf[:, 1] * N_MELS / max(last['y_qry'].shape[0]//HOP_LENGTH, 1),
               s=2, c='red', alpha=0.6,
               label=f"Fine MAE={res_fine[last['key']]:.0f}ms")
    gt_fine = np.arange(spec_ref.shape[1]) / EVAL_RATE
    ax.plot(np.arange(spec_ref.shape[1]),
            gt_fine * N_MELS / max(spec_ref.shape[1] * EVAL_RATE, 1),
            'r--', lw=1, alpha=0.5, label='GT')
    ax.set_title('Stage 2: Fine Alignment', fontsize=11)
    ax.set_xlabel('Ref time (fine frames)')
    ax.set_ylabel('Mel bin')
    ax.legend(fontsize=8)

    # Panel 3: MAE distribution box plot across all windows
    ax = axes[2]
    data   = [m_v, s_v, c_v, f_v]
    labels = ['MFCC\n+DTW', 'Spec\n+DTW', 'LoFTR\nCoarse', 'LoFTR\nFine']
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(f'MAE Distribution ({len(m_v)} windows)', fontsize=11)
    ax.set_ylabel('MAE (ms)')
    ax.grid(axis='y', alpha=0.3)
    for i, (m, med) in enumerate([(m_m,m_med),(s_m,s_med),(c_m,c_med),(f_m,f_med)], 1):
        ax.text(i, ax.get_ylim()[1]*0.95, f'μ={m:.0f}', ha='center',
                fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(DRIVE_FIG, dpi=150, bbox_inches='tight')
    print(f"\n Figure saved: {DRIVE_FIG}")

print("\nDone.")
