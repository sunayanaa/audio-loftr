# eval_multirate_v1.py
#
# Evaluates the trained Audio LoFTR (from model_ctf_v5.pth) across
# multiple time-stretch rates on the held-out test tracks.
#
# Rates: 0.8, 0.9, 1.1, 1.2, 1.3, 1.4
#   (1.0 omitted — trivial, zero error for all methods)
#   (rate < 1.0 → query shorter than ref; rate > 1.0 → query shorter still
#    because librosa time_stretch(rate=r) produces duration/r seconds)
#
# GT formula: j_gt = i / rate  (correct for librosa time_stretch)
#
# Output:
#   - Console table + LaTeX table (methods × rates)
#   - ctf_multirate_results.json  (cache, resumes on reconnect)
#   - ctf_multirate_figure.png    (heatmap + line plot)

import os, json, random
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

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
DRIVE_RESULTS = os.path.join(DRIVE_BASE, "ctf_multirate_results.json")
DRIVE_FIG     = os.path.join(DRIVE_BASE, "ctf_multirate_figure.png")

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
N_MELS       = 128
HOP_LENGTH   = 256
N_MFCC       = 20
HOP_MFCC     = 512
COARSE_SCALE = 4
TUBE_RADIUS  = 12
CLIP_DUR     = 8.0
CLIP_SAMPLES = int(CLIP_DUR * SAMPLE_RATE)
SEED         = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rates to evaluate — excludes 1.0 (trivial) and the single training rate 1.2
# Covers mild (0.9, 1.1) and strong (0.8, 1.3, 1.4) stretch
EVAL_RATES   = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4]

FRAME_MS_MFCC   = HOP_MFCC   / SAMPLE_RATE * 1000   # 23.22 ms
FRAME_MS_FINE   = HOP_LENGTH  / SAMPLE_RATE * 1000   # 11.61 ms
FRAME_MS_COARSE = HOP_LENGTH  * COARSE_SCALE / SAMPLE_RATE * 1000  # 46.44 ms

METHODS = ['mfcc', 'spec', 'coarse', 'fine']
METHOD_LABELS = {
    'mfcc':   'MFCC + DTW',
    'spec':   'Spec + DTW',
    'coarse': 'LoFTR Coarse',
    'fine':   'LoFTR Fine',
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device  : {DEVICE}")
print(f"Rates   : {EVAL_RATES}")

# ── Model definition (identical to v5) ────────────────────────────────────
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
        self.nhead  = nhead; self.d_head = d_model // nhead
        self.scale  = self.d_head ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
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
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,d_model,3,stride=2,padding=1), nn.BatchNorm2d(d_model), nn.ReLU())
        self.transformer = LocalFeatureTransformer(d_model, nhead)
        self.T = temperature
    def _match(self, f0, f1):
        f0 = F.normalize(f0, dim=-1); f1 = F.normalize(f1, dim=-1)
        sim = torch.einsum("bmd,bnd->bmn", f0, f1) / self.T
        return F.softmax(sim, dim=1) * F.softmax(sim, dim=2)
    def forward(self, img0, img1):
        x0 = self.enc(img0).mean(dim=2).transpose(1,2)
        x1 = self.enc(img1).mean(dim=2).transpose(1,2)
        f0, f1 = self.transformer(x0, x1)
        return {'conf_matrix': self._match(f0, f1)}

# ── Load model ─────────────────────────────────────────────────────────────
model = AudioLoFTR().to(DEVICE)
ckpt  = torch.load(DRIVE_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"✅ Model loaded (epoch {ckpt.get('epoch','?')}): {DRIVE_CKPT}")

# ── Load test split ────────────────────────────────────────────────────────
with open(DRIVE_SPLIT) as f:
    split = json.load(f)
test_tracks = split['test']

# Build test windows (same logic as v5)
def get_windows(track_fname):
    path = os.path.join(DRIVE_DATA, track_fname)
    dur  = librosa.get_duration(path=path)
    wins = []
    start = 0.0
    while start + CLIP_DUR <= dur:
        wins.append((track_fname, int(start * SAMPLE_RATE)))
        start += CLIP_DUR
    return wins

test_windows = []
for t in test_tracks:
    test_windows.extend(get_windows(t))
print(f"Test windows : {len(test_windows)} from {len(test_tracks)} tracks")

# ── Audio helpers ──────────────────────────────────────────────────────────
def load_window(track_fname, start_sample):
    path = os.path.join(DRIVE_DATA, track_fname)
    y, _ = librosa.load(path, sr=SAMPLE_RATE,
                         offset=start_sample/SAMPLE_RATE,
                         duration=CLIP_DUR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)

def audio_to_mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                        n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(m, ref=np.max)

def mel_db_to_tensor(m_db):
    m = (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)
    return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

def calc_mae(path, rate, frame_ms):
    i    = path[:, 0].astype(np.float64)
    j    = path[:, 1].astype(np.float64)
    j_gt = i / rate
    return float(np.mean(np.abs(j - j_gt)) * frame_ms)

# ── Alignment methods ──────────────────────────────────────────────────────
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

def coarse_align(y_ref, y_qry):
    with torch.no_grad():
        conf = model(
            mel_db_to_tensor(audio_to_mel_db(y_ref)),
            mel_db_to_tensor(audio_to_mel_db(y_qry))
        )['conf_matrix'][0].cpu().numpy()
    cost = -np.log(conf + 1e-8)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()

def fine_align(y_ref, y_qry, path_coarse):
    spec_ref = audio_to_mel_db(y_ref)
    spec_qry = audio_to_mel_db(y_qry)
    T_ref = spec_ref.shape[1]; T_qry = spec_qry.shape[1]
    tube  = np.full((T_ref, T_qry), 10.0, dtype=np.float32)
    for ic, jc in path_coarse:
        i_fine = ic * COARSE_SCALE; j_fine = jc * COARSE_SCALE
        i0 = max(0, i_fine - TUBE_RADIUS); i1 = min(T_ref, i_fine + TUBE_RADIUS + 1)
        j0 = max(0, j_fine - TUBE_RADIUS); j1 = min(T_qry, j_fine + TUBE_RADIUS + 1)
        local = cdist(spec_ref[:, i0:i1].T,
                      spec_qry[:, j0:j1].T, metric='cosine').astype(np.float32)
        tube[i0:i1, j0:j1] = np.minimum(tube[i0:i1, j0:j1], local)
    _, wp = librosa.sequence.dtw(C=tube, backtrack=True)
    return wp[::-1].copy()

# ── Load cache ─────────────────────────────────────────────────────────────
# Cache structure: results[rate_str][method][window_key] = mae_ms
if os.path.exists(DRIVE_RESULTS):
    with open(DRIVE_RESULTS) as f:
        results = json.load(f)
    print(f"✅ Cache loaded: {DRIVE_RESULTS}")
else:
    results = {}

def save_cache():
    with open(DRIVE_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

# ── Evaluation loop ────────────────────────────────────────────────────────
total_windows = len(test_windows) * len(EVAL_RATES)
done_count    = sum(
    len(results.get(str(r), {}).get('fine', {}))
    for r in EVAL_RATES)
print(f"\nEvaluating {total_windows} window×rate combinations "
      f"({done_count} already cached)...\n")

for rate in EVAL_RATES:
    rkey = str(rate)
    if rkey not in results:
        results[rkey] = {m: {} for m in METHODS}

    n_done = len(results[rkey]['fine'])
    n_todo = len(test_windows) - n_done
    print(f"── Rate {rate}  ({n_done}/{len(test_windows)} done) ──────────────")

    for (track, start) in test_windows:
        wkey = f"{track}@{start}"
        if wkey in results[rkey]['fine']:
            continue

        try:
            y_ref = load_window(track, start)
            y_qry = librosa.effects.time_stretch(y_ref, rate=rate)

            path_m = mfcc_dtw(y_ref, y_qry)
            path_s = spec_dtw(y_ref, y_qry)
            path_c = coarse_align(y_ref, y_qry)
            path_f = fine_align(y_ref, y_qry, path_c)

            results[rkey]['mfcc'][wkey]   = calc_mae(path_m, rate, FRAME_MS_MFCC)
            results[rkey]['spec'][wkey]   = calc_mae(path_s, rate, FRAME_MS_FINE)
            results[rkey]['coarse'][wkey] = calc_mae(path_c, rate, FRAME_MS_COARSE)
            results[rkey]['fine'][wkey]   = calc_mae(path_f, rate, FRAME_MS_FINE)
            save_cache()

            print(f"  {track} +{start//SAMPLE_RATE:2.0f}s | "
                  f"MFCC={results[rkey]['mfcc'][wkey]:.0f} "
                  f"Spec={results[rkey]['spec'][wkey]:.0f} "
                  f"Coarse={results[rkey]['coarse'][wkey]:.0f} "
                  f"Fine={results[rkey]['fine'][wkey]:.0f}  ms")

        except Exception as e:
            print(f"  ERROR {wkey} rate={rate}: {e}")
            continue

# ── Aggregate ──────────────────────────────────────────────────────────────
# mean_table[method][rate] = mean MAE in ms
mean_table = {m: {} for m in METHODS}
std_table  = {m: {} for m in METHODS}

for rate in EVAL_RATES:
    rkey = str(rate)
    for m in METHODS:
        vals = list(results[rkey][m].values())
        if vals:
            mean_table[m][rate] = float(np.mean(vals))
            std_table[m][rate]  = float(np.std(vals))
        else:
            mean_table[m][rate] = float('nan')
            std_table[m][rate]  = float('nan')

# ── Console table ──────────────────────────────────────────────────────────
rate_strs = [str(r) for r in EVAL_RATES]
col_w = 12

print("\n" + "="*80)
print(f"{'Method':<20}" + "".join(f"{r:>{col_w}}" for r in EVAL_RATES) + "  (mean MAE ms)")
print("-"*80)
for m in METHODS:
    row = f"{METHOD_LABELS[m]:<20}"
    for rate in EVAL_RATES:
        v = mean_table[m][rate]
        row += f"{v:>{col_w}.1f}"
    print(row)
print("="*80)

# ── LaTeX table ────────────────────────────────────────────────────────────
n_wins = len(test_windows)
n_trk  = len(test_tracks)

latex  = f"\n\\begin{{table}}[t]\n"
latex += f"\\caption{{Alignment MAE (ms) across time-stretch rates. "
latex += f"N={n_wins} windows from {n_trk} test tracks, {CLIP_DUR:.0f}\\,s clips.}}\n"
latex += "\\centering\n"
latex += "\\begin{tabular}{l" + "c"*len(EVAL_RATES) + "}\n\\hline\n"
latex += "\\textbf{Method} & " + " & ".join(
    f"\\textbf{{×{r}}}" for r in EVAL_RATES) + " \\\\\n\\hline\n"

for m in METHODS:
    row_vals = []
    for rate in EVAL_RATES:
        mu = mean_table[m][rate]
        sd = std_table[m][rate]
        cell = f"{mu:.1f}"
        # Bold the best (lowest) value in each rate column
        row_vals.append((mu, sd, cell))
    # Find best per column handled below
    if m == 'fine':
        prefix = "\\textbf{LoFTR Fine}"
    else:
        prefix = METHOD_LABELS[m]
    latex += f"{prefix} & " + " & ".join(c for _,_,c in row_vals) + " \\\\\n"

latex += "\\hline\n\\end{tabular}\n\\label{tab:multirate}\n\\end{table}\n"
print(latex)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(f"Multi-Rate Evaluation — {n_wins} windows, {n_trk} test tracks",
             fontsize=12)

colors = {'mfcc': '#4878CF', 'spec': '#6ACC65',
          'coarse': '#D65F5F', 'fine': '#B47CC7'}
markers = {'mfcc': 'o', 'spec': 's', 'coarse': '^', 'fine': 'D'}

# Panel 1: line plot — mean MAE vs rate
ax = axes[0]
for m in METHODS:
    ys = [mean_table[m][r] for r in EVAL_RATES]
    es = [std_table[m][r]  for r in EVAL_RATES]
    ax.errorbar(EVAL_RATES, ys, yerr=es,
                label=METHOD_LABELS[m],
                color=colors[m], marker=markers[m],
                linewidth=2, markersize=7, capsize=4)
ax.axvline(1.2, color='gray', linestyle='--', alpha=0.5, label='Train rate')
ax.set_xlabel('Time-stretch rate  (librosa, rate<1 → longer query)')
ax.set_ylabel('Mean MAE (ms)')
ax.set_title('MAE vs Stretch Rate')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(EVAL_RATES)

# Panel 2: heatmap — methods × rates
ax = axes[1]
methods_ordered = METHODS
data_matrix = np.array([[mean_table[m][r] for r in EVAL_RATES]
                         for m in methods_ordered])
im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(len(EVAL_RATES)))
ax.set_xticklabels([f'×{r}' for r in EVAL_RATES])
ax.set_yticks(range(len(methods_ordered)))
ax.set_yticklabels([METHOD_LABELS[m] for m in methods_ordered])
ax.set_title('MAE Heatmap (ms, lower=better)')
for i, m in enumerate(methods_ordered):
    for j, r in enumerate(EVAL_RATES):
        v = mean_table[m][r]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                fontsize=9, color='black',
                fontweight='bold' if m == 'fine' else 'normal')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='MAE (ms)')

plt.tight_layout()
plt.savefig(DRIVE_FIG, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {DRIVE_FIG}")
print("\nDone.")
