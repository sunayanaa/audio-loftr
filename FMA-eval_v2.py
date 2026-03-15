# ============================================================
# Program    : FMA-eval_v2.py
# Version    : 2.0
# Description:
#   Self-contained zero-shot cross-dataset generalisation
#   experiment using the FMA (Free Music Archive) small subset.
#
#   This script requires no prior execution of FMA-explore_v1.py.
#   It handles the full pipeline end-to-end:
#
#     1. Mounts Google Drive (if running in Colab)
#     2. Extracts fma_small.zip from DRIVE_BASE to Colab local
#        disk (/content/fma_small) — skipped if already present
#     3. Verifies the extracted contents and reports a file count
#     4. Selects 60 tracks (one 8-second window each, offset +8s)
#        from the 8,000 available MP3s using a fixed random seed
#     5. Runs four alignment methods across six time-stretch rates:
#          MFCC+DTW, Spec+DTW, LoFTR Coarse, LoFTR Fine
#     6. Caches per-window results to Drive after every window
#        so Colab disconnects are harmless — re-run to resume
#     7. Produces console + LaTeX table and two figures
#
#   FMA dataset properties (confirmed by FMA-explore_v1.py):
#     - 8,000 MP3 tracks, all exactly 30 seconds
#     - 44.1 kHz stereo (few tracks at 48 kHz — both resampled
#       transparently by librosa to 22.05 kHz mono)
#     - Two-level numeric folder structure:
#       fma_small/NNN/NNNNNN.mp3
#     - Eight genres: Hip-Hop, Pop, Folk, Experimental, Rock,
#       International, Electronic, Instrumental
#     - No metadata CSV inside the zip (genre labels require the
#       separate fma_metadata download; not needed here)
#
#   Experimental protocol matches eval_multirate_v1.py exactly:
#     - 8-second windows, offset +8s from track start
#     - Time-stretch rates: 0.8, 0.9, 1.1, 1.2, 1.3, 1.4
#     - GT convention: j_gt = i / rate  (librosa time_stretch)
#     - MAX_WINDOWS = 60 for direct comparability with GTZAN
#       (60 windows, 20 tracks) and MAPS (60 windows, 25 tracks)
#
#   The Audio LoFTR model (model_ctf_v5.pth) was trained
#   exclusively on GTZAN Jazz. No FMA audio was seen during
#   training. This is a zero-shot transfer test across three
#   axes: instrument timbre (jazz ensemble → mixed genres),
#   recording condition (studio → web-sourced), and genre
#   (single genre → eight genres).
#
#   Outputs written to DRIVE_BASE:
#     FMA-eval_v2_results.json      per-window MAE cache
#     FMA-eval_v2_figure.png        line plot + heatmap
#     FMA-eval_v2_figure_box.png    per-rate box plots
#     Console + LaTeX table         ready for paper Section 5.5
#
#   Prerequisites:
#     - model_ctf_v5.pth in DRIVE_BASE
#       (produced by demo_coarse_to_fine_v5.py)
#     - fma_small.zip in DRIVE_BASE
#       (download from https://os.unil.cloud.switch.ch/fma/
#        fma_small.zip and upload to Google Drive)
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
ZIP_PATH      = os.path.join(DRIVE_BASE, "fma_small.zip")
EXTRACT_DIR   = "/content/fma_small"          # Colab local disk — fast I/O
DRIVE_RESULTS = os.path.join(DRIVE_BASE, "FMA-eval_v2_results.json")
DRIVE_FIG     = os.path.join(DRIVE_BASE, "FMA-eval_v2_figure.png")
DRIVE_FIG_BOX = os.path.join(DRIVE_BASE, "FMA-eval_v2_figure_box.png")
os.makedirs(DRIVE_BASE,  exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ── 2. Config (identical to demo_coarse_to_fine_v5.py) ────────────────────
SAMPLE_RATE   = 22050
N_MELS        = 128
HOP_LENGTH    = 256
N_MFCC        = 20
HOP_MFCC      = 512
COARSE_SCALE  = 4
TUBE_RADIUS   = 12
CLIP_DUR      = 8.0
CLIP_SAMPLES  = int(CLIP_DUR * SAMPLE_RATE)
EVAL_RATES    = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
MAX_WINDOWS   = 60             # one window per track, 60 tracks
WINDOW_OFFSET_S = 8.0          # start at +8s to avoid fade-in/out
WINDOW_OFFSET   = int(WINDOW_OFFSET_S * SAMPLE_RATE)
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
print(f"Max windows : {MAX_WINDOWS}  (one per track, offset +{WINDOW_OFFSET_S:.0f}s)")

# ── 3. Extract fma_small.zip ────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 1 — Extraction")
print("="*60)

# The zip unpacks to fma_small/NNN/NNNNNN.mp3
# Check for the presence of that inner folder to detect prior extraction
fma_inner = os.path.join(EXTRACT_DIR, "fma_small")

if os.path.isdir(fma_inner) and any(
        f.endswith('.mp3')
        for _, _, files in os.walk(fma_inner)
        for f in files):
    existing = [
        os.path.join(root, f)
        for root, _, files in os.walk(fma_inner)
        for f in files if f.endswith('.mp3')
    ]
    print(f"  Already extracted — {len(existing)} MP3 files found.")
    print(f"  Skipping extraction.")
else:
    assert os.path.exists(ZIP_PATH), (
        f"\n[ERROR] Zip not found: {ZIP_PATH}\n"
        "Please upload fma_small.zip to your Drive at that path.\n"
        "Download from: https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    )
    zip_size_gb = os.path.getsize(ZIP_PATH) / 1024**3
    print(f"  Found: {ZIP_PATH}  ({zip_size_gb:.1f} GB)")
    print(f"  Extracting to {EXTRACT_DIR} ...")
    print(f"  This takes approximately 8–12 minutes on Colab.")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        members = zf.namelist()
        mp3_members = [m for m in members if m.endswith('.mp3')]
        total = len(mp3_members)
        # Extract with progress reporting every 500 files
        for i, member in enumerate(members):
            zf.extract(member, EXTRACT_DIR)
            if member.endswith('.mp3') and (i == 0 or
                    sum(1 for m in members[:i+1] if m.endswith('.mp3'))
                    % 500 == 0):
                done = sum(1 for m in members[:i+1] if m.endswith('.mp3'))
                print(f"    {done}/{total} MP3s extracted ...", end='\r')
    print()
    existing = [
        os.path.join(root, f)
        for root, _, files in os.walk(fma_inner)
        for f in files if f.endswith('.mp3')
    ]
    print(f"  Done. {len(existing)} MP3 files extracted.")

# ── 4. Collect and verify all MP3 paths ────────────────────────────────────
print("\n" + "="*60)
print("Step 2 — File inventory")
print("="*60)

all_mp3s = sorted([
    os.path.join(root, f)
    for root, _, files in os.walk(EXTRACT_DIR)
    for f in files if f.lower().endswith('.mp3')
])

assert len(all_mp3s) > 0, (
    f"No MP3 files found under {EXTRACT_DIR}.\n"
    "The zip may not have extracted correctly — delete the folder "
    f"at {EXTRACT_DIR} and re-run to force re-extraction."
)

# Report folder distribution
folder_counts = collections.Counter(
    os.path.basename(os.path.dirname(p)) for p in all_mp3s
)
print(f"  Total MP3 files : {len(all_mp3s)}")
print(f"  Subfolders      : {len(folder_counts)}  "
      f"(range: {min(folder_counts.keys())} – {max(folder_counts.keys())})")
print(f"  Files per folder: min={min(folder_counts.values())}  "
      f"max={max(folder_counts.values())}  "
      f"median={int(np.median(list(folder_counts.values())))}")

# ── 5. Select 60 tracks ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 3 — Window selection")
print("="*60)

rng = random.Random(SEED)
shuffled = all_mp3s[:]
rng.shuffle(shuffled)
selected_tracks = shuffled[:MAX_WINDOWS]
selected_tracks.sort()

# Each entry: (abs_path, start_sample)
eval_windows = [(p, WINDOW_OFFSET) for p in selected_tracks]

print(f"  Tracks available         : {len(all_mp3s)}")
print(f"  Tracks selected          : {len(eval_windows)}")
print(f"  Window offset            : +{WINDOW_OFFSET_S:.0f}s  "
      f"(seconds 8–16 of each 30-s track)")
print(f"\n  Selected track IDs (first 10):")
for p, _ in eval_windows[:10]:
    tid = os.path.splitext(os.path.basename(p))[0]
    folder = os.path.basename(os.path.dirname(p))
    print(f"    {folder}/{tid}.mp3")
print(f"    ... ({len(eval_windows) - 10} more)")

# ── 6. Model definition (byte-for-byte identical to demo_coarse_to_fine_v5) ─
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

# ── 7. Load checkpoint ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 4 — Load model checkpoint")
print("="*60)

assert os.path.exists(DRIVE_CKPT), (
    f"Checkpoint not found: {DRIVE_CKPT}\n"
    "Run demo_coarse_to_fine_v5.py first to train and save the model."
)
model = AudioLoFTR().to(DEVICE)
ckpt  = torch.load(DRIVE_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"  ✅ Model loaded (epoch {ckpt.get('epoch','?')}): {DRIVE_CKPT}")

# ── 8. Audio helpers ────────────────────────────────────────────────────────
def load_window(abs_path, start_sample):
    """Load 8-second mono window; librosa resamples any SR → 22.05 kHz."""
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

# ── 9. Load result cache ─────────────────────────────────────────────────────
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

# ── 10. Evaluation loop ──────────────────────────────────────────────────────
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
        # Cache key: six-digit track ID + offset — portable across sessions
        track_id = os.path.splitext(os.path.basename(abs_path))[0]
        wkey     = f"{track_id}@{start}"
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

            print(f"  {track_id}  +{start//SAMPLE_RATE:.0f}s | "
                  f"MFCC={results[rkey]['mfcc'][wkey]:5.0f}  "
                  f"Spec={results[rkey]['spec'][wkey]:5.0f}  "
                  f"Coarse={results[rkey]['coarse'][wkey]:5.0f}  "
                  f"Fine={results[rkey]['fine'][wkey]:5.0f}  ms")

        except Exception as e:
            print(f"  ERROR {wkey} rate={rate}: {e}")
            continue

# ── 11. Aggregate ─────────────────────────────────────────────────────────────
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

# ── 12. Console table ─────────────────────────────────────────────────────────
col_w = 10
sep   = "=" * 82
print("\n" + sep)
print("FMA Small — Zero-shot cross-dataset transfer")
print(f"  {n_wins} windows · {n_trks} tracks · 8 s clips · "
      f"44.1 kHz stereo resampled to 22.05 kHz mono")
print(f"  8 genres: Hip-Hop, Pop, Folk, Experimental, Rock, "
      f"International, Electronic, Instrumental")
print(f"  Model trained on GTZAN Jazz only — no FMA data seen during training")
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

# Report fine vs spec difference — key diagnostic for paper interpretation
print("\nLoFTR Fine vs Spec+DTW mean difference per rate (ms):")
for rate in EVAL_RATES:
    diff = mean_table['fine'][rate] - mean_table['spec'][rate]
    print(f"  ×{rate}:  {diff:+.2f} ms")

# ── 13. LaTeX table ───────────────────────────────────────────────────────────
latex  = "\n% ── Table: FMA Small zero-shot transfer (FMA-eval_v2.py) ──\n"
latex += "\\begin{table}[t]\n"
latex += ("\\caption{Alignment MAE (ms) on FMA Small under time-stretch "
          "(zero-shot cross-dataset transfer). "
          f"$N\\,=\\,{n_wins}$ windows from {n_trks} tracks across eight genres "
          f"({CLIP_DUR:.0f}\\,s clips, 44.1\\,kHz stereo resampled to "
          "22\\,kHz mono). "
          "The model was trained exclusively on GTZAN Jazz; "
          "no FMA audio was seen during training.}\n")
latex += "\\label{tab:fma_multirate}\n"
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

# ── 14. Figure A: line plot + heatmap ─────────────────────────────────────────
colors  = {'mfcc': '#4878CF', 'spec': '#6ACC65',
           'coarse': '#D65F5F', 'fine': '#B47CC7'}
markers = {'mfcc': 'o', 'spec': 's', 'coarse': '^', 'fine': 'D'}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    f"FMA Small — Zero-shot cross-dataset transfer  "
    f"({n_wins} windows, {n_trks} tracks, 8 genres)\n"
    f"Model trained on GTZAN Jazz only",
    fontsize=11)

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
ax.set_title('MAE vs Stretch Rate — FMA Small (zero-shot)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(EVAL_RATES)
ax.set_xticklabels([f'×{r}' for r in EVAL_RATES])

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

# ── 15. Figure B: per-rate box plots ─────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle(
    f"FMA Small — MAE distributions per stretch rate  "
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
            ax.text(i, ylim[1] * 0.95, f'μ={mu:.0f}',
                    ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(DRIVE_FIG_BOX, dpi=150, bbox_inches='tight')
print(f"Box-plot figure saved : {DRIVE_FIG_BOX}")

print("\nDone.")