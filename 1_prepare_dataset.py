"""
Step 1 - Prepare the 4-class dataset from ESC-50.

Usage:  python 1_prepare_dataset.py  [path/to/ESC-50-master]
Output: dataset_logmel.npz, mel_filterbank.npy, feature_config.json

Pipeline: ESC-50 clip (44.1 kHz) -> 16 kHz mono -> 1 s windows
          -> RMS normalisation -> log-Mel (61 frames x 40 bands)
The feature code below is deliberately plain NumPy so it can be re-implemented
identically in ESP32-S3 firmware (same window, FFT size, hop, filterbank, log).
"""
import sys, json
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly

ROOT = sys.argv[1] if len(sys.argv) > 1 else "ESC-50-master"
SR, WIN = 16000, 16000            # 16 kHz, 1 s window
N_FFT, HOP, N_MELS = 512, 256, 40
FMIN, FMAX = 0.0, 8000.0
TARGET_RMS = 0.05                 # every window is level-normalised to this RMS
SEED = 42
CLASSES = ["Normal Ambient", "Vehicle Horn", "Construction / Drilling", "Emergency Siren"]
# ESC-50 has no "drilling" class; power-tool sounds are the closest match.
EVENT_MAP = {"car_horn": 1, "chainsaw": 2, "hand_saw": 2, "siren": 3}
# loud events that are NOT our classes and could be confused with them: leave out of Ambient
EXCLUDE = {"clock_alarm", "church_bells", "fireworks", "glass_breaking"}
EVENT_HOP = WIN // 4              # events: dense windows (0.25 s hop) -> more samples
AMBIENT_PER_CLIP = 2              # ambient: at most 2 random windows per clip
AUG_PER_EVENT = 2                 # train-only: noise-mixed copies of each event window

rng = np.random.default_rng(SEED)

def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank():
    hz = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fb = np.zeros((N_MELS, len(bins)), np.float32)
    for i in range(N_MELS):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)))
    return fb

FB = mel_filterbank()
HANN = np.hanning(N_FFT).astype(np.float32)
rms = lambda x: float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def logmel(x):
    x = x * (TARGET_RMS / rms(x))
    n = 1 + (len(x) - N_FFT) // HOP                      # 61 frames for 16000 samples
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * HANN, axis=1)) ** 2
    return np.log(power @ FB.T + 1e-6).astype(np.float32)  # (61, 40)

def load16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    return resample_poly(x, SR // np.gcd(SR, sr), sr // np.gcd(SR, sr)).astype(np.float32)

def win_starts(n, hop): return list(range(0, n - WIN + 1, hop))

meta = pd.read_csv(f"{ROOT}/meta/esc50.csv")
X, y, fold, aug = [], [], [], []
ambient_pool = []                                       # raw ambient windows (train folds) for noise mixing
event_wave = []                                         # (wave, label, fold) for train-fold augmentation

for _, r in meta.iterrows():
    cat = r["category"]
    if cat in EXCLUDE: continue
    x = load16k(f"{ROOT}/audio/{r['filename']}")
    if len(x) < WIN: continue
    starts = win_starts(len(x), EVENT_HOP if cat in EVENT_MAP else WIN // 2)
    levels = np.array([rms(x[s:s + WIN]) for s in starts])
    if cat in EVENT_MAP:                                # keep only windows that really contain the event
        keep = [s for s, l in zip(starts, levels) if l >= 0.3 * levels.max() and l > 1e-3]
        label = EVENT_MAP[cat]
    else:                                               # ambient: skip digital silence, sample a few windows
        keep = [s for s, l in zip(starts, levels) if l > 1e-4]
        keep = list(rng.permutation(keep)[:AMBIENT_PER_CLIP]) if keep else []
        label = 0
    for s in keep:
        w = x[s:s + WIN]
        X.append(logmel(w)); y.append(label); fold.append(int(r["fold"])); aug.append(0)
        if r["fold"] <= 3:                              # only training folds feed augmentation
            (ambient_pool if label == 0 else event_wave).append((w, label, int(r["fold"])))

# noise-mixed augmentation of EVENT windows, training folds only (SNR 5-20 dB)
pool = [w for w, _, _ in ambient_pool]
for w, label, f in event_wave:
    for _ in range(AUG_PER_EVENT):
        noise = pool[rng.integers(len(pool))]
        snr = rng.uniform(5, 20)
        mixed = w + noise * (rms(w) / rms(noise)) / (10 ** (snr / 20))
        X.append(logmel(mixed)); y.append(label); fold.append(f); aug.append(1)

X = np.stack(X); y = np.array(y); fold = np.array(fold); aug = np.array(aug)
np.savez_compressed("dataset_logmel.npz", X=X, y=y, fold=fold, aug=aug, classes=np.array(CLASSES))
np.save("mel_filterbank.npy", FB)
json.dump(dict(sample_rate=SR, window=WIN, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
               target_rms=TARGET_RMS, frames=int(X.shape[1]), window_fn="numpy.hanning(512) (symmetric)",
               log="ln(power @ mel_fb.T + 1e-6)"), open("feature_config.json", "w"), indent=2)

print("X shape:", X.shape)
print(f"{'class':26s}" + "".join(f"fold{f:>6d}" for f in range(1, 6)) + "   augmented")
for k, name in enumerate(CLASSES):
    print(f"{name:26s}" + "".join(f"{int(((y == k) & (fold == f) & (aug == 0)).sum()):>10d}" for f in range(1, 6))
          + f"{int(((y == k) & (aug == 1)).sum()):>12d}")
print("Saved dataset_logmel.npz, mel_filterbank.npy, feature_config.json")
