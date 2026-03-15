# ============================================================
# Program    : MAPS-eval_v1.py
# Version    : 1.0
# Description:
#   Cross-dataset generalisation experiment for Audio LoFTR
#   using the MAPS ENSTDkCl subset (real Yamaha Disklavier,
#   close-microphone, 44.1 kHz stereo).
#
#   The Audio LoFTR model (model_ctf_v5.pth) was trained
#   exclusively on GTZAN Jazz. This script applies it — without
#   any fine-tuning — to 30 classical piano MUS recordings,
#   constituting a zero-shot transfer test across instrument
#   timbre (jazz ensemble → solo piano), recording condition
#   (studio → Disklavier), and sample rate (22.05 → 44.1 kHz,
#   resampled transparently by librosa).
#
#   Experimental protocol matches eval_multirate_v1.py exactly:
#     - Non-overlapping 8-second windows from MUS files only
#       (ENSTDkCl/MUS/MAPS_MUS-*.wav — 30 pieces, 68–634 s each)
#     - Time-stretch rates: 0.8, 0.9, 1.1, 1.2, 1.3, 1.4
#     - GT convention: j_gt = i / rate  (librosa time_stretch)
#     - Four methods: MFCC+DTW, Spec+DTW, LoFTR Coarse, LoFTR Fine
#     - Results capped at MAX_WINDOWS=60 for direct comparability
#       with the GTZAN 60-window evaluation in Table 1
#
#   Outputs written to DRIVE_BASE:
#     MAPS-eval_v1_results.json      per-window MAE cache (resumes)
#     MAPS-eval_v1_figure.png        line plot + heatmap
#     MAPS-eval_v1_figure_box.png    per-rate box plots
#     Console + LaTeX table          ready for paper Section 5.4
#
#   Prerequisites:
#     - model_ctf_v5.pth  (produced by demo_coarse_to_fine_v5.py)
#     - ENSTDkCl.zip      (upload to DRIVE_BASE before running)
#     - MAPS-explore_v1.py must have been run at least once so
#       that /content/maps_enstdkcl is already extracted, saving
#       the 3-minute unzip time on re-runs.
# ============================================================

import os, json, random, zipfile, collections
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

# ── 0. Mount Drive ─────────────────────────────────────────────────────────
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except ImportError:
    pass

# ── 1. Paths ───────────────────────────────────────────────────────────────
DRIVE_BASE    = "/content/drive/MyDrive/audio-loftr"
DRIVE_CKPT    = os.path.join(DRIVE_BASE, "model_ctf_v5.pth")
ZIP_PATH      = os.path.join(DRIVE_BASE, "ENSTDkCl.zip")
EXTRACT_DIR   = "/content/maps_enstdkcl"          # Colab scratch — fast local I/O
DRIVE_RESULTS = os.path.join(DRIVE_BASE, "MAPS-eval_v1_results.json")
DRIVE_FIG     = os.path.join(DRIVE_BASE, "MAPS-eval_v1_figure.png")
DRIVE_FIG_BOX = os.path.join(DRIVE_BASE, "MAPS-eval_v1_figure_box.png")
os.makedirs(DRIVE_BASE, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ── 2. Config (identical to demo_coarse_to_fine_v5.py) ────────────────────
SAMPLE_RATE   = 22050          # librosa resamples 44.1 kHz → 22.05 kHz
N_MELS        = 128
HOP_LENGTH    = 256
N_MFCC        = 20
HOP_MFCC      = 512
COARSE_SCALE  = 4
TUBE_RADIUS   = 12
CLIP_DUR      = 8.0            # seconds per window
CLIP_SAMPLES  = int(CLIP_DUR * SAMPLE_RATE)
EVAL_RATES    = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
MAX_WINDOWS   = 60             # matches GTZAN eval count for direct comparison
SEED          = 42
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FRAME_MS_MFCC   = HOP_MFCC  / SAMPLE_RATE * 1000   # 23.22 ms
FRAME_MS_FINE   = HOP_LENGTH / SAMPLE_RATE * 1000   # 11.61 ms
FRAME_MS_COARSE = HOP_LENGTH * COARSE_SCALE / SAMPLE_RATE * 1000  # 46.44 ms

METHODS = ['mfcc', 'spec', 'coarse', 'fine']
METHOD_LABELS = {
    'mfcc':   'MFCC + DTW',
    'spec':   'Spec + DTW',
    'coarse': 'LoFTR Coarse',
    'fine':   'LoFTR Fine',
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device      : {DEVICE}")
print(f"Eval rates  : {EVAL_RATES}")
print(f"Max windows : {MAX_WINDOWS}")

# ── 3. Model definition (byte-for-byte identical to demo_coarse_to_fine_v5) ─
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
        x0 = self.enc(img0).mean(dim=2).transpose(1, 2)
        x1 = self.enc(img1).mean(dim=2).transpose(1, 2)
        f0, f1 = self.transformer(x0, x1)
        return {'conf_matrix': self._match(f0, f1)}

# ── 4. Load checkpoint ──────────────────────────────────────────────────────
assert os.path.exists(DRIVE_CKPT), \
    f"Checkpoint not found: {DRIVE_CKPT}\nRun demo_coarse_to_fine_v5.py first."
model = AudioLoFTR().to(DEVICE)
ckpt  = torch.load(DRIVE_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"✅ Model loaded (epoch {ckpt.get('epoch','?')}): {DRIVE_CKPT}")

# ── 5. Locate MUS files ─────────────────────────────────────────────────────
# Internal zip structure confirmed by MAPS-explore_v1.py:
#   ENSTDkCl/MUS/MAPS_MUS-<piece>_ENSTDkCl.wav
# The category token is the PARENT DIRECTORY name ('MUS'),
# not part of the filename itself — hence the directory-level check.
print("\n" + "="*60)
print("Locating MUS files ...")
print("="*60)

# Extract zip if the MUS folder is not yet present on disk
mus_dir = os.path.join(EXTRACT_DIR, "ENSTDkCl", "MUS")
if not os.path.isdir(mus_dir):
    assert os.path.exists(ZIP_PATH), \
        f"Zip not found: {ZIP_PATH}\n" \
        "Please upload ENSTDkCl.zip to DRIVE_BASE and re-run."
    print(f"  Extracting {ZIP_PATH}  →  {EXTRACT_DIR}")
    print("  (~3 minutes — 2.6 GB from Drive to Colab local disk)")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("  Extraction complete.")
else:
    print(f"  MUS directory already present: {mus_dir}")

# Collect all WAV files whose immediate parent directory is 'MUS'
mus_files = sorted([
    os.path.join(root, fname)
    for root, dirs, files in os.walk(EXTRACT_DIR)
    for fname in files
    if fname.lower().endswith('.wav')
    and os.path.basename(root).upper() == 'MUS'
])

assert len(mus_files) > 0, \
    f"No MUS wav files found under {EXTRACT_DIR}.\n" \
    "Run MAPS-explore_v1.py to verify the zip contents."

print(f"  MUS wav files found : {len(mus_files)}")
for p in mus_files:
    print(f"    {os.path.basename(p)}")

# ── 6. Build window list ────────────────────────────────────────────────────
print("\n" + "="*60)
print("Building window index ...")
print("="*60)

all_windows = []
track_info  = {}
for path in mus_files:
    try:
        dur = librosa.get_duration(path=path)
    except Exception as e:
        print(f"  [WARN] Cannot read {os.path.basename(path)}: {e}")
        continue
    wins = []
    start = 0.0
    while start + CLIP_DUR <= dur:
        wins.append((path, int(start * SAMPLE_RATE)))
        start += CLIP_DUR
    track_info[os.path.basename(path)] = {
        'duration_s': round(dur, 1),
        'n_windows' : len(wins),
    }
    all_windows.extend(wins)

print(f"\n  {'Piece':<50}  {'Dur(s)':>7}  {'Wins':>4}")
print("  " + "-"*65)
for fname, info in sorted(track_info.items()):
    print(f"  {fname:<50}  {info['duration_s']:>7.1f}  {info['n_windows']:>4}")
print("  " + "-"*65)
total_available = sum(v['n_windows'] for v in track_info.values())
print(f"  {'TOTAL':<50}  {'':>7}  {total_available:>4}")

# Stable shuffle → cap at MAX_WINDOWS for parity with GTZAN evaluation
rng = random.Random(SEED)
shuffled = all_windows[:]
rng.shuffle(shuffled)
eval_windows = shuffled[:MAX_WINDOWS]
eval_windows.sort(key=lambda x: (x[0], x[1]))   # sort for readable progress

n_eval_tracks = len(set(p for p, _ in eval_windows))
print(f"\n  Windows selected for eval : {len(eval_windows)} / {total_available} "
      f"(cap={MAX_WINDOWS}, from {n_eval_tracks} tracks)")

# ── 7. Audio helpers ────────────────────────────────────────────────────────
def load_window(abs_path, start_sample):
    """Load 8-second mono window; librosa resamples 44.1 kHz → 22.05 kHz."""
    y, _ = librosa.load(abs_path, sr=SAMPLE_RATE,
                        offset=start_sample / SAMPLE_RATE,
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

def mfcc_dtw(y_ref, y_qry):
    m1   = librosa.feature.mfcc(y=y_ref, sr=SAMPLE_RATE,
                                 n_mfcc=N_MFCC, hop_length=HOP_MFCC)
    m2   = librosa.feature.mfcc(y=y_qry, sr=SAMPLE_RATE,
                                 n_mfcc=N_MFCC, hop_length=HOP_MFCC)
    cost = cdist(m1.T, m2.T, metric='cosine').astype(np.float32)
    _, wp = librosa.sequence.dtw(C=cost, backtrack=True)
    return wp[::-1].copy()

def spec_dtw(y_ref, y_qry):
    s1   = audio_to_mel_db(y_ref)
    s2   = audio_to_mel_db(y_qry)
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
    return wp[::-1].copy(), conf

def fine_align(y_ref, y_qry, path_coarse):
    spec_ref = audio_to_mel_db(y_ref)
    spec_qry = audio_to_mel_db(y_qry)
    T_ref    = spec_ref.shape[1]
    T_qry    = spec_qry.shape[1]
    tube     = np.full((T_ref, T_qry), 10.0, dtype=np.float32)
    for ic, jc in path_coarse:
        i_fine = ic * COARSE_SCALE
        j_fine = jc * COARSE_SCALE
        i0 = max(0, i_fine - TUBE_RADIUS); i1 = min(T_ref, i_fine + TUBE_RADIUS + 1)
        j0 = max(0, j_fine - TUBE_RADIUS); j1 = min(T_qry, j_fine + TUBE_RADIUS + 1)
        local = cdist(spec_ref[:, i0:i1].T,
                      spec_qry[:, j0:j1].T, metric='cosine').astype(np.float32)
        tube[i0:i1, j0:j1] = np.minimum(tube[i0:i1, j0:j1], local)
    _, wp = librosa.sequence.dtw(C=tube, backtrack=True)
    return wp[::-1].copy()

# ── 8. Load result cache ─────────────────────────────────────────────────────
if os.path.exists(DRIVE_RESULTS):
    with open(DRIVE_RESULTS) as f:
        results = json.load(f)
    done_total = sum(
        len(results.get(str(r), {}).get('fine', {})) for r in EVAL_RATES)
    print(f"\n✅ Cache loaded — {done_total} window×rate entries already done.")
else:
    results    = {}
    done_total = 0
    print("\nNo cache — starting fresh.")

def save_cache():
    with open(DRIVE_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

# ── 9. Evaluation loop ───────────────────────────────────────────────────────
total_combos = len(eval_windows) * len(EVAL_RATES)
print(f"\nEvaluating {len(eval_windows)} windows × {len(EVAL_RATES)} rates "
      f"= {total_combos} combinations  ({done_total} already cached)\n")

for rate in EVAL_RATES:
    rkey = str(rate)
    if rkey not in results:
        results[rkey] = {m: {} for m in METHODS}

    n_done = len(results[rkey].get('fine', {}))
    print(f"── Rate ×{rate}  ({n_done}/{len(eval_windows)} cached) ──────────────")

    for (abs_path, start) in eval_windows:
        # Cache key: filename only (not full path) — portable across Colab sessions
        wkey = f"{os.path.basename(abs_path)}@{start}"
        if wkey in results[rkey].get('fine', {}):
            continue

        try:
            y_ref          = load_window(abs_path, start)
            y_qry          = librosa.effects.time_stretch(y_ref, rate=rate)

            path_m         = mfcc_dtw(y_ref, y_qry)
            path_s         = spec_dtw(y_ref, y_qry)
            path_c, conf   = coarse_align(y_ref, y_qry)
            path_f         = fine_align(y_ref, y_qry, path_c)

            results[rkey]['mfcc'][wkey]   = calc_mae(path_m, rate, FRAME_MS_MFCC)
            results[rkey]['spec'][wkey]   = calc_mae(path_s, rate, FRAME_MS_FINE)
            results[rkey]['coarse'][wkey] = calc_mae(path_c, rate, FRAME_MS_COARSE)
            results[rkey]['fine'][wkey]   = calc_mae(path_f, rate, FRAME_MS_FINE)
            save_cache()

            fname_short = os.path.basename(abs_path).replace(
                'MAPS_MUS-', '').replace('_ENSTDkCl.wav', '')
            print(f"  {fname_short:<35} +{start//SAMPLE_RATE:3.0f}s | "
                  f"MFCC={results[rkey]['mfcc'][wkey]:5.0f}  "
                  f"Spec={results[rkey]['spec'][wkey]:5.0f}  "
                  f"Coarse={results[rkey]['coarse'][wkey]:5.0f}  "
                  f"Fine={results[rkey]['fine'][wkey]:5.0f}  ms")

        except Exception as e:
            print(f"  ERROR {wkey} rate={rate}: {e}")
            continue

# ── 10. Aggregate ─────────────────────────────────────────────────────────────
mean_table = {m: {} for m in METHODS}
std_table  = {m: {} for m in METHODS}
med_table  = {m: {} for m in METHODS}

for rate in EVAL_RATES:
    rkey = str(rate)
    for m in METHODS:
        vals = list(results.get(rkey, {}).get(m, {}).values())
        if vals:
            mean_table[m][rate] = float(np.mean(vals))
            std_table[m][rate]  = float(np.std(vals))
            med_table[m][rate]  = float(np.median(vals))
        else:
            mean_table[m][rate] = float('nan')
            std_table[m][rate]  = float('nan')
            med_table[m][rate]  = float('nan')

n_wins = len(eval_windows)
n_trks = len(set(p for p, _ in eval_windows))

# ── 11. Console table ─────────────────────────────────────────────────────────
col_w = 10
sep   = "=" * 82
print("\n" + sep)
print(f"MAPS ENSTDkCl — Zero-shot cross-dataset transfer")
print(f"  {n_wins} windows · {n_trks} tracks · 8 s clips · "
      f"44.1 kHz stereo resampled to 22.05 kHz mono")
print(f"  Model trained on GTZAN Jazz only — no piano audio seen during training")
print(sep)
print(f"{'Method':<20}" + "".join(f"{'×'+str(r):>{col_w}}" for r in EVAL_RATES)
      + "   (mean MAE ms)")
print("-" * 82)
for m in METHODS:
    marker = " ◀" if m == 'fine' else ""
    row    = f"{METHOD_LABELS[m]:<20}"
    for rate in EVAL_RATES:
        row += f"{mean_table[m][rate]:>{col_w}.1f}"
    print(row + marker)
print(sep)

# ── 12. LaTeX table ───────────────────────────────────────────────────────────
latex  = "\n% ── Table: MAPS ENSTDkCl zero-shot transfer (MAPS-eval_v1.py) ──\n"
latex += "\\begin{table}[t]\n"
latex += ("\\caption{Alignment MAE (ms) on MAPS ENSTDkCl under time-stretch "
          "(zero-shot cross-dataset transfer). "
          f"N\\,=\\,{n_wins} windows drawn from {n_trks} classical piano pieces "
          f"({CLIP_DUR:.0f}\\,s clips, 44.1\\,kHz stereo resampled to "
          "22\\,kHz mono). "
          "The model was trained exclusively on GTZAN Jazz; "
          "no MAPS audio was seen during training.}\n")
latex += "\\label{tab:maps_multirate}\n"
latex += "\\centering\n"
latex += "\\begin{tabular}{l" + "c" * len(EVAL_RATES) + "}\n"
latex += "\\hline\n"
latex += ("\\textbf{Method} & "
          + " & ".join(f"\\textbf{{$\\times${r}}}" for r in EVAL_RATES)
          + " \\\\\n\\hline\n")

for m in METHODS:
    cells = [f"{mean_table[m][r]:.1f}" for r in EVAL_RATES]
    label = f"\\textbf{{{METHOD_LABELS[m]}}}" if m == 'fine' else METHOD_LABELS[m]
    latex += label + " & " + " & ".join(cells) + " \\\\\n"

latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
print(latex)

# ── 13. Figure A: line plot + heatmap ─────────────────────────────────────────
colors  = {'mfcc': '#4878CF', 'spec': '#6ACC65',
           'coarse': '#D65F5F', 'fine': '#B47CC7'}
markers = {'mfcc': 'o', 'spec': 's', 'coarse': '^', 'fine': 'D'}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    f"MAPS ENSTDkCl — Zero-shot cross-dataset transfer  "
    f"({n_wins} windows, {n_trks} piano tracks)\n"
    f"Model trained on GTZAN Jazz only",
    fontsize=11)

# Panel 1 — mean MAE ± std vs stretch rate
ax = axes[0]
for m in METHODS:
    ys = [mean_table[m][r] for r in EVAL_RATES]
    es = [std_table[m][r]  for r in EVAL_RATES]
    ax.errorbar(EVAL_RATES, ys, yerr=es,
                label=METHOD_LABELS[m],
                color=colors[m], marker=markers[m],
                linewidth=2, markersize=7, capsize=4)
ax.axvline(1.2, color='gray', linestyle='--', alpha=0.5,
           label='GTZAN training rate')
ax.set_xlabel('Time-stretch rate')
ax.set_ylabel('Mean MAE (ms)')
ax.set_title('MAE vs Stretch Rate — MAPS ENSTDkCl (zero-shot)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(EVAL_RATES)
ax.set_xticklabels([f'×{r}' for r in EVAL_RATES])

# Panel 2 — heatmap
ax = axes[1]
data_matrix = np.array([[mean_table[m][r] for r in EVAL_RATES]
                         for m in METHODS])
im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(len(EVAL_RATES)))
ax.set_xticklabels([f'×{r}' for r in EVAL_RATES])
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
ax.set_title('MAE Heatmap (ms, lower = better)')
for i, m in enumerate(METHODS):
    for j, r in enumerate(EVAL_RATES):
        v = mean_table[m][r]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                fontsize=9, color='black',
                fontweight='bold' if m == 'fine' else 'normal')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='MAE (ms)')

plt.tight_layout()
plt.savefig(DRIVE_FIG, dpi=150, bbox_inches='tight')
print(f"Figure saved      : {DRIVE_FIG}")

# ── 14. Figure B: per-rate box plots ─────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle(
    f"MAPS ENSTDkCl — MAE distributions per stretch rate  "
    f"(zero-shot: model trained on GTZAN Jazz only)",
    fontsize=12)

bcolors = [colors[m] for m in METHODS]
blabels = [METHOD_LABELS[m] for m in METHODS]

for idx, rate in enumerate(EVAL_RATES):
    ax   = axes2.flatten()[idx]
    rkey = str(rate)
    data = [list(results.get(rkey, {}).get(m, {}).values()) for m in METHODS]
    bp   = ax.boxplot(data, tick_labels=blabels, patch_artist=True, notch=False)
    for patch, col in zip(bp['boxes'], bcolors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_title(f'Rate ×{rate}', fontsize=11)
    ax.set_ylabel('MAE (ms)')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=8)
    ylim = ax.get_ylim()
    for i, m in enumerate(METHODS, 1):
        mu = mean_table[m][rate]
        if not np.isnan(mu):
            ax.text(i, ylim[1] * 0.95, f'μ={mu:.0f}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(DRIVE_FIG_BOX, dpi=150, bbox_inches='tight')
print(f"Box-plot figure saved : {DRIVE_FIG_BOX}")

print("\nDone.")