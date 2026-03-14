# eval_pitch_v2.py
#
# Compares two fine-stage feature choices under pitch shift:
#   - fine_mel  : cosine DTW on mel-spectrogram (original, pitch-sensitive)
#   - fine_chroma: cosine DTW on chroma features (pitch-class invariant)
#
# Both use the same coarse path from Audio LoFTR for the tube.
# Also adds chroma_dtw as a standalone baseline (chroma without LoFTR coarse).
#
# Conditions: stretch x1.2 + pitch shifts [-4,-2,-1,0,+1,+2,+4] semitones
# GT formula: j_gt = i / rate
#
# Output:
#   - Console + LaTeX table
#   - ctf_pitch_v2_results.json  (cache)
#   - ctf_pitch_v2_figure.png

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
DRIVE_BASE    = "/content/drive/MyDrive/GTZAN"
DRIVE_DATA    = os.path.join(DRIVE_BASE, "jazz")
DRIVE_CKPT    = os.path.join(DRIVE_BASE, "model_ctf_v5.pth")
DRIVE_SPLIT   = os.path.join(DRIVE_BASE, "ctf_v5_split.json")
DRIVE_RESULTS = os.path.join(DRIVE_BASE, "ctf_pitch_v2_results.json")
DRIVE_FIG     = os.path.join(DRIVE_BASE, "ctf_pitch_v2_figure.png")

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
N_MELS       = 128
N_CHROMA     = 12
HOP_LENGTH   = 256          # fine-stage hop (mel and chroma)
N_MFCC       = 20
HOP_MFCC     = 512
COARSE_SCALE = 4
TUBE_RADIUS  = 12
CLIP_DUR     = 8.0
CLIP_SAMPLES = int(CLIP_DUR * SAMPLE_RATE)
SEED         = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STRETCH_RATE = 1.2
PITCH_SHIFTS = [-4, -2, -1, 0, 1, 2, 4]

FRAME_MS_MFCC   = HOP_MFCC   / SAMPLE_RATE * 1000
FRAME_MS_FINE   = HOP_LENGTH  / SAMPLE_RATE * 1000
FRAME_MS_COARSE = HOP_LENGTH  * COARSE_SCALE / SAMPLE_RATE * 1000

# Methods evaluated
METHODS = ['mfcc', 'spec', 'chroma', 'coarse', 'fine_mel', 'fine_chroma']
METHOD_LABELS = {
    'mfcc':        'MFCC + DTW',
    'spec':        'Spec + DTW',
    'chroma':      'Chroma + DTW',
    'coarse':      'LoFTR Coarse',
    'fine_mel':    'LoFTR + Mel Fine',
    'fine_chroma': 'LoFTR + Chroma Fine',
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device        : {DEVICE}")
print(f"Stretch rate  : {STRETCH_RATE}")
print(f"Pitch shifts  : {PITCH_SHIFTS} semitones")

# ── Model (identical to v5) ────────────────────────────────────────────────
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

model = AudioLoFTR().to(DEVICE)
ckpt  = torch.load(DRIVE_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded (epoch {ckpt.get('epoch','?')})")

# ── Load split + build test windows ───────────────────────────────────────
with open(DRIVE_SPLIT) as f:
    split = json.load(f)
test_tracks = split['test']

def get_windows(track_fname):
    path = os.path.join(DRIVE_DATA, track_fname)
    dur  = librosa.get_duration(path=path)
    wins = []; start = 0.0
    while start + CLIP_DUR <= dur:
        wins.append((track_fname, int(start * SAMPLE_RATE)))
        start += CLIP_DUR
    return wins

test_windows = []
for t in test_tracks:
    test_windows.extend(get_windows(t))
print(f"Test windows : {len(test_windows)} from {len(test_tracks)} tracks")

# ── Audio/feature helpers ──────────────────────────────────────────────────
def load_window(track_fname, start_sample):
    path = os.path.join(DRIVE_DATA, track_fname)
    y, _ = librosa.load(path, sr=SAMPLE_RATE,
                         offset=start_sample / SAMPLE_RATE,
                         duration=CLIP_DUR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)

def make_query(y_ref, stretch_rate, pitch_semitones):
    y = librosa.effects.time_stretch(y_ref, rate=stretch_rate)
    if pitch_semitones != 0:
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=pitch_semitones)
    return y

def audio_to_mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                        n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(m, ref=np.max)

def audio_to_chroma(y):
    """
    CQT-based chroma with L2 normalisation per frame.
    CQT chroma is more pitch-invariant than STFT chroma
    because its log-frequency bins align naturally with semitones.
    """
    c = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE,
                                    hop_length=HOP_LENGTH,
                                    n_chroma=N_CHROMA)
    # L2-normalise each frame to remove loudness variation
    norms = np.linalg.norm(c, axis=0, keepdims=True) + 1e-8
    return c / norms   # [12, T]

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

def chroma_dtw(y_ref, y_qry):
    """Standalone Chroma+DTW baseline — no LoFTR."""
    c1 = audio_to_chroma(y_ref)
    c2 = audio_to_chroma(y_qry)
    cost = cdist(c1.T, c2.T, metric='cosine').astype(np.float32)
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

def fine_align_features(y_ref, y_qry, path_coarse, feat_ref, feat_qry):
    """
    Generic fine alignment through coarse tube using arbitrary feature arrays.
    feat_ref, feat_qry: [D, T] arrays (mel, chroma, or any other features)
    """
    T_ref = feat_ref.shape[1]
    T_qry = feat_qry.shape[1]
    tube  = np.full((T_ref, T_qry), 10.0, dtype=np.float32)

    for ic, jc in path_coarse:
        i_fine = ic * COARSE_SCALE
        j_fine = jc * COARSE_SCALE
        i0 = max(0, i_fine - TUBE_RADIUS)
        i1 = min(T_ref, i_fine + TUBE_RADIUS + 1)
        j0 = max(0, j_fine - TUBE_RADIUS)
        j1 = min(T_qry, j_fine + TUBE_RADIUS + 1)
        local = cdist(feat_ref[:, i0:i1].T,
                      feat_qry[:, j0:j1].T,
                      metric='cosine').astype(np.float32)
        tube[i0:i1, j0:j1] = np.minimum(tube[i0:i1, j0:j1], local)

    _, wp = librosa.sequence.dtw(C=tube, backtrack=True)
    return wp[::-1].copy()

# ── Load cache ─────────────────────────────────────────────────────────────
if os.path.exists(DRIVE_RESULTS):
    with open(DRIVE_RESULTS) as f:
        results = json.load(f)
    print(f"Cache loaded: {DRIVE_RESULTS}")
else:
    results = {}

def save_cache():
    with open(DRIVE_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

# ── Evaluation loop ────────────────────────────────────────────────────────
total = len(test_windows) * len(PITCH_SHIFTS)
done  = sum(len(results.get(str(p), {}).get('fine_chroma', {}))
            for p in PITCH_SHIFTS)
print(f"\nEvaluating {total} combinations ({done} cached)...\n")

for pitch in PITCH_SHIFTS:
    pkey = str(pitch)
    if pkey not in results:
        results[pkey] = {m: {} for m in METHODS}
    # Ensure all method keys exist (for resuming older partial caches)
    for m in METHODS:
        if m not in results[pkey]:
            results[pkey][m] = {}

    n_done = len(results[pkey]['fine_chroma'])
    print(f"── Pitch {pitch:+d} semitones  "
          f"({n_done}/{len(test_windows)} done) ─────────────")

    for (track, start) in test_windows:
        wkey = f"{track}@{start}"
        if wkey in results[pkey]['fine_chroma']:
            continue
        try:
            y_ref = load_window(track, start)
            y_qry = make_query(y_ref, STRETCH_RATE, pitch)

            # Compute features once, reuse across methods
            mel_ref   = audio_to_mel_db(y_ref)
            mel_qry   = audio_to_mel_db(y_qry)
            chroma_ref = audio_to_chroma(y_ref)
            chroma_qry = audio_to_chroma(y_qry)

            # Baselines
            path_m = mfcc_dtw(y_ref, y_qry)
            path_s = spec_dtw(y_ref, y_qry)
            path_ch = chroma_dtw(y_ref, y_qry)

            # LoFTR coarse (uses mel encoder — same as training)
            path_c = coarse_align(y_ref, y_qry)

            # Fine: mel features through coarse tube
            path_fm = fine_align_features(y_ref, y_qry, path_c, mel_ref, mel_qry)

            # Fine: chroma features through coarse tube
            path_fc = fine_align_features(y_ref, y_qry, path_c, chroma_ref, chroma_qry)

            results[pkey]['mfcc'][wkey]        = calc_mae(path_m,  STRETCH_RATE, FRAME_MS_MFCC)
            results[pkey]['spec'][wkey]        = calc_mae(path_s,  STRETCH_RATE, FRAME_MS_FINE)
            results[pkey]['chroma'][wkey]      = calc_mae(path_ch, STRETCH_RATE, FRAME_MS_FINE)
            results[pkey]['coarse'][wkey]      = calc_mae(path_c,  STRETCH_RATE, FRAME_MS_COARSE)
            results[pkey]['fine_mel'][wkey]    = calc_mae(path_fm, STRETCH_RATE, FRAME_MS_FINE)
            results[pkey]['fine_chroma'][wkey] = calc_mae(path_fc, STRETCH_RATE, FRAME_MS_FINE)
            save_cache()

            fc = results[pkey]['fine_chroma'][wkey]
            fm = results[pkey]['fine_mel'][wkey]
            cc = results[pkey]['coarse'][wkey]
            print(f"  {track} +{start//SAMPLE_RATE:2.0f}s | "
                  f"Coarse={cc:.0f}  MelFine={fm:.0f}  ChromaFine={fc:.0f}  ms")

        except Exception as e:
            print(f"  ERROR {wkey} pitch={pitch}: {e}")
            continue

# ── Aggregate ──────────────────────────────────────────────────────────────
mean_t = {m: {} for m in METHODS}
std_t  = {m: {} for m in METHODS}

for pitch in PITCH_SHIFTS:
    pkey = str(pitch)
    for m in METHODS:
        vals = list(results[pkey][m].values())
        mean_t[m][pitch] = float(np.mean(vals)) if vals else float('nan')
        std_t[m][pitch]  = float(np.std(vals))  if vals else float('nan')

# ── Console table ──────────────────────────────────────────────────────────
col_w = 9
sep   = "="*80
print(f"\n{sep}")
print(f"Stretch x{STRETCH_RATE}  |  pitch shift on query  |  MAE in ms")
print(f"{'Method':<24}" +
      "".join(f"{p:>+{col_w}}st" for p in PITCH_SHIFTS))
print("-"*80)
for m in METHODS:
    row = f"{METHOD_LABELS[m]:<24}"
    for p in PITCH_SHIFTS:
        row += f"{mean_t[m][p]:>{col_w}.1f}"
    print(row)
print(sep)

# Improvement of chroma-fine over mel-fine
print("\nChroma Fine vs Mel Fine improvement (positive = chroma better):")
for p in PITCH_SHIFTS:
    delta = mean_t['fine_mel'][p] - mean_t['fine_chroma'][p]
    print(f"  {p:+d}st : {delta:+.1f} ms  "
          f"({'chroma better' if delta > 0 else 'mel better' if delta < 0 else 'equal'})")

# ── LaTeX table ────────────────────────────────────────────────────────────
n_wins = len(test_windows); n_trk = len(test_tracks)
pitch_hdrs = " & ".join(f"\\textbf{{{p:+d}st}}" for p in PITCH_SHIFTS)

latex  = f"\n\\begin{{table}}[t]\n"
latex += (f"\\caption{{Alignment MAE (ms) under combined time-stretch "
          f"($\\times${STRETCH_RATE}) and pitch-shift. "
          f"N={n_wins} windows from {n_trk} test tracks, {CLIP_DUR:.0f}\\,s clips.}}\n")
latex += "\\centering\n"
latex += "\\begin{tabular}{l" + "c"*len(PITCH_SHIFTS) + "}\n\\hline\n"
latex += f"\\textbf{{Method}} & {pitch_hdrs} \\\\\n\\hline\n"

for m in METHODS:
    cells = [f"{mean_t[m][p]:.1f}" for p in PITCH_SHIFTS]
    if m in ('coarse', 'fine_chroma'):
        label = f"\\textbf{{{METHOD_LABELS[m]}}}"
    else:
        label = METHOD_LABELS[m]
    latex += f"{label} & " + " & ".join(cells) + " \\\\\n"
    if m == 'chroma':   # separator before LoFTR methods
        latex += "\\hline\n"

latex += "\\hline\n\\end{tabular}\n\\label{tab:pitch_chroma}\n\\end{table}\n"
print(latex)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(
    f"Chroma vs Mel Fine Stage  —  stretch x{STRETCH_RATE} + pitch shift  "
    f"({len(test_windows)} windows, {len(test_tracks)} tracks)",
    fontsize=12)

# Colour scheme: baselines grey-ish, LoFTR methods vivid
colors = {
    'mfcc':        '#7090C0',
    'spec':        '#70C080',
    'chroma':      '#C09040',
    'coarse':      '#D65F5F',
    'fine_mel':    '#B090D0',
    'fine_chroma': '#9030A0',
}
markers = {'mfcc':'o','spec':'s','chroma':'P',
           'coarse':'^','fine_mel':'D','fine_chroma':'*'}
lwidths = {'mfcc':1.2,'spec':1.2,'chroma':1.2,
           'coarse':2.0,'fine_mel':1.8,'fine_chroma':2.5}
lstyles = {'mfcc':'--','spec':'--','chroma':'--',
           'coarse':'-','fine_mel':':','fine_chroma':'-'}

# Panel 1: all methods
ax = axes[0]
for m in METHODS:
    ys = [mean_t[m][p] for p in PITCH_SHIFTS]
    es = [std_t[m][p]  for p in PITCH_SHIFTS]
    ax.errorbar(PITCH_SHIFTS, ys, yerr=es,
                label=METHOD_LABELS[m],
                color=colors[m], marker=markers[m],
                linewidth=lwidths[m], linestyle=lstyles[m],
                markersize=7 if m != 'fine_chroma' else 10,
                capsize=3)
ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Pitch shift (semitones)')
ax.set_ylabel('Mean MAE (ms)')
ax.set_title('All Methods')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xticks(PITCH_SHIFTS)

# Panel 2: LoFTR methods only (zoom in)
ax = axes[1]
loftr_methods = ['coarse', 'fine_mel', 'fine_chroma']
for m in loftr_methods:
    ys = [mean_t[m][p] for p in PITCH_SHIFTS]
    es = [std_t[m][p]  for p in PITCH_SHIFTS]
    ax.errorbar(PITCH_SHIFTS, ys, yerr=es,
                label=METHOD_LABELS[m],
                color=colors[m], marker=markers[m],
                linewidth=lwidths[m], linestyle=lstyles[m],
                markersize=7 if m != 'fine_chroma' else 10,
                capsize=3)
# Add chroma baseline for reference
ys_ch = [mean_t['chroma'][p] for p in PITCH_SHIFTS]
ax.plot(PITCH_SHIFTS, ys_ch, label='Chroma+DTW (baseline)',
        color=colors['chroma'], marker=markers['chroma'],
        linewidth=1.2, linestyle='--', markersize=6)
ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Pitch shift (semitones)')
ax.set_ylabel('Mean MAE (ms)')
ax.set_title('LoFTR Methods (zoom)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(PITCH_SHIFTS)

plt.tight_layout()
plt.savefig(DRIVE_FIG, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {DRIVE_FIG}")
print("\nDone.")
