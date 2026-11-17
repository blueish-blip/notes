"""
tuner_core.py
Semua fungsi inti dari notebook "cek_nada_chord_gitar_dtw.ipynb", dipisah agar bisa dipakai
oleh dashboard Streamlit (app.py). Isinya:
  - konstanta (senar, chord), DTW dari nol, jarak Euclid
  - fitur semiton / chroma / MFCC (numpy + scipy)
  - estimasi frekuensi dasar (YIN dan jumlah harmonik)
  - perakitan genjrengan chord dari nada NSynth
  - statistik akurasi (Wilson, presisi/recall/F1, McNemar, AUC, Youden)
  - jalankan_analisis(): menjalankan seluruh analisis Bagian 1 & 2 lalu menyimpan hasil ke pickle
  - tuner_live() dan cek_chord_live(): dipakai halaman rekam langsung
"""
import os, io, json, re, time, pickle, tarfile
from math import gcd
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample, resample_poly, lfilter, fftconvolve, butter, sosfiltfilt
from scipy.fft import rfft, rfftfreq, dct
from scipy.stats import binomtest

# ----------------------------------------------------------------------------- konstanta
SR = 16000
SENAR = {40: "E2", 45: "A2", 50: "D3", 55: "G3", 59: "B3", 64: "E4"}
PITCH = list(SENAR)
LABEL = [SENAR[p] for p in PITCH]
NOMOR_SENAR = {"E2": 6, "A2": 5, "D3": 4, "G3": 3, "B3": 2, "E4": 1}
NAMA_NADA = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD = {
    "C":  [48, 52, 55, 60, 64],        # x 3 2 0 1 0
    "D":  [50, 57, 62, 66],            # x x 0 2 3 2
    "E":  [40, 47, 52, 56, 59, 64],    # 0 2 2 1 0 0
    "G":  [43, 47, 50, 55, 59, 67],    # 3 2 0 0 0 3
    "A":  [45, 52, 57, 61, 64],        # x 0 2 2 2 0
    "Am": [45, 52, 57, 60, 64],        # x 0 2 2 1 0
    "Em": [40, 47, 52, 55, 59, 64],    # 0 2 2 0 0 0
    "Dm": [50, 57, 62, 65],            # x x 0 2 3 1
}
POSISI_JARI = {"C": "x 3 2 0 1 0", "D": "x x 0 2 3 2", "E": "0 2 2 1 0 0", "G": "3 2 0 0 0 3",
               "A": "x 0 2 2 2 0", "Am": "x 0 2 2 1 0", "Em": "0 2 2 0 0 0", "Dm": "x x 0 2 3 1"}
NAMA_CHORD = list(CHORD)
NADA_DIBUTUHKAN = list(range(40, 69))
VARIASI = ["asli", "derau", "pendek", "fals"]
VARIASI2 = ["turun", "naik", "lambat", "derau"]


def midi_ke_hz(m): return 440.0 * 2 ** ((m - 69) / 12)
def hz_ke_midi(f): return 69 + 12 * np.log2(f / 440.0)
def nama_nada(m): m = int(round(m)); return f"{NAMA_NADA[m % 12]}{m // 12 - 1}"
def pc(pitches): return {p % 12 for p in pitches}


# ----------------------------------------------------------------------------- DTW dan jarak
def dtw(x, y, band=None):
    """Kembalikan (jarak DTW, matriks akumulasi D, jalur warping 0-based). Baris = waktu."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        C = np.abs(x[:, None] - y[None, :])
    else:
        C = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = 1, m
        if band is not None:
            lo = max(1, int(np.floor((i - 1) * m / n)) + 1 - band)
            hi = min(m, int(np.ceil(i * m / n)) + band)
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    if not np.isfinite(D[n, m]):
        return np.inf, D[1:, 1:], np.empty((0, 2), int)
    i, j, path = n, m, [(n - 1, m - 1)]
    while (i, j) != (1, 1):
        langkah = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        i, j = langkah[int(np.argmin([D[a, b] for a, b in langkah]))]
        path.append((i - 1, j - 1))
    return D[n, m], D[1:, 1:], np.array(path[::-1])


def dtw_norm(A, B, band=None):
    return dtw(A, B, band=band)[0] / (len(A) + len(B))


def euclid_pad(A, B):
    L = max(len(A), len(B))
    A2, B2 = np.zeros((L, A.shape[1])), np.zeros((L, B.shape[1]))
    A2[:len(A)], B2[:len(B)] = A, B
    return np.linalg.norm(A2 - B2) / L


def _regangkan(A, L=32):
    t0, t1 = np.linspace(0, 1, len(A)), np.linspace(0, 1, L)
    return np.stack([np.interp(t1, t0, A[:, d]) for d in range(A.shape[1])], axis=1)


def euclid_resample(A, B, L=32):
    return np.linalg.norm(_regangkan(A, L) - _regangkan(B, L)) / L


JARAK = {"Euclid padding": euclid_pad, "Euclid resampling": euclid_resample, "DTW": dtw_norm}


# ----------------------------------------------------------------------------- audio
def muat(sumber):
    """Baca WAV dari path atau bytes/file-like; mono, 16 kHz, dinormalkan ke [-1, 1]."""
    if isinstance(sumber, (bytes, bytearray)):
        sumber = io.BytesIO(sumber)
    sr, x = wavfile.read(sumber)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x / (np.abs(x).max() + 1e-9)


def pangkas_sunyi(x, top_db=40, frame=512):
    e = np.array([np.sum(x[i:i + frame] ** 2) for i in range(0, len(x), frame)]) + 1e-12
    aktif = np.where(10 * np.log10(e / e.max()) > -top_db)[0]
    return x[aktif[0] * frame:(aktif[-1] + 1) * frame] if len(aktif) else x


def bersihkan(x, fc=70):
    """Buang DC dan gemuruh frekuensi rendah (getaran HP, angin) di bawah fc Hz."""
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 100:
        return x
    return sosfiltfilt(butter(4, fc, "highpass", fs=SR, output="sos"), x)


def variasi(x, jenis, rng):
    if jenis == "derau":
        return x + rng.normal(0, np.sqrt(np.mean(x ** 2) / 10 ** (15 / 10)), len(x))
    if jenis == "pendek":
        return x[: int(0.8 * SR)]
    if jenis == "fals":
        r = 2 ** (rng.choice([-35, 35]) / 1200)
        return resample(x, int(len(x) / r))
    return x


# ----------------------------------------------------------------------------- fitur
N_FFT, HOP = 8192, 2048
FREK = rfftfreq(N_FFT, 1 / SR)
MIDI_BIN = np.arange(36, 97)                 # C2 .. C7
with np.errstate(divide="ignore"):
    _midi_frek = 69 + 12 * np.log2(FREK / 440.0)
M_SEMITON = np.zeros((len(FREK), len(MIDI_BIN)))
for _k, _m in enumerate(MIDI_BIN):
    M_SEMITON[np.abs(_midi_frek - _m) <= 0.5, _k] = 1.0


def _mel_filterbank(n_mel=40, fmin=40, fmax=SR / 2):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    titik = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2))
    F = np.zeros((len(FREK), n_mel))
    for k in range(n_mel):
        a, b, c = titik[k:k + 3]
        F[:, k] = np.clip(np.minimum((FREK - a) / (b - a), (c - FREK) / (c - b)), 0, None)
    return F


M_MEL = _mel_filterbank()


def stft_power(x):
    x = np.pad(np.asarray(x, float), N_FFT // 2)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    w = np.hanning(N_FFT)
    frames = np.stack([x[i:i + N_FFT] * w for i in range(0, len(x) - N_FFT + 1, HOP)])
    return np.abs(rfft(frames, axis=1)) ** 2


def _norm_frame(S):
    S = np.log1p(1000 * S / (S.max() + 1e-12))
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)


def fitur_semiton(x):
    return _norm_frame(stft_power(x) @ M_SEMITON)


def fitur_chroma(x):
    S = stft_power(x) @ M_SEMITON
    C = np.zeros((S.shape[0], 12))
    for k, m in enumerate(MIDI_BIN):
        C[:, m % 12] += S[:, k]
    return _norm_frame(C)


def fitur_mfcc(x, n=13):
    L = np.log(stft_power(x) @ M_MEL + 1e-10)
    F = dct(L, type=2, axis=1, norm="ortho")[:, :n]
    return F - F.mean(axis=0)


FITUR = {"semiton": fitur_semiton, "chroma": fitur_chroma, "MFCC": fitur_mfcc}
N_AWAL = int(1.5 * SR / HOP)                 # frame dalam 1,5 detik pertama


# ----------------------------------------------------------------------------- frekuensi dasar
def yin_f0(x, fmin=60, fmax=1000, frame=2048, hop=512, ambang=0.15, maks_detik=1.0):
    tau_min, tau_max = int(SR / fmax), int(SR / fmin)
    batas = min(len(x), int(maks_detik * SR)) - frame - tau_max
    f0 = []
    for s in range(0, max(batas, 1), hop):
        w = x[s:s + frame + tau_max]
        if len(w) < frame + tau_max:
            break
        x0 = w[:frame]
        if np.sum(x0 ** 2) < 1e-6:
            continue
        acf = fftconvolve(w, x0[::-1], mode="valid")[: tau_max + 1]
        e_geser = np.convolve(w ** 2, np.ones(frame), mode="valid")[: tau_max + 1]
        d = np.sum(x0 ** 2) + e_geser - 2 * acf
        d[0] = 0
        cmnd = d * np.arange(len(d)) / (np.cumsum(d) + 1e-12)
        cmnd[0] = 1
        kand = np.where(cmnd[tau_min:] < ambang)[0]
        if len(kand):
            tau = tau_min + kand[0]
            while tau + 1 < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
        else:
            tau = tau_min + int(np.argmin(cmnd[tau_min:]))
        if 1 <= tau < len(cmnd) - 1:
            a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
            tau = tau + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        f0.append(SR / tau)
    return float(np.median(f0)) if f0 else np.nan


def prediksi_yin(x):
    f = yin_f0(x)
    if not np.isfinite(f):
        return PITCH[0]
    return min(PITCH, key=lambda p: abs(p - hz_ke_midi(f)))


def f0_harmonik(x, f_target, lebar_cent=100, n_harm=6, detik=1.5):
    """Cari f0 di sekitar f_target (±lebar_cent) dengan menjumlahkan energi di harmoniknya."""
    seg = np.asarray(x[:int(detik * SR)], float)
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    n = 1 << 18
    S = np.abs(rfft(seg, n)); f = rfftfreq(n, 1 / SR)
    kandidat = f_target * 2 ** (np.linspace(-lebar_cent, lebar_cent, 801) / 1200)
    skor = sum(np.interp(k * kandidat, f, S) / k for k in range(1, n_harm + 1))
    return float(kandidat[int(np.argmax(skor))])


def cent_dari(f0, f_acuan):
    return 1200 * np.log2(f0 / f_acuan)


def status_cent(cent):
    if not np.isfinite(cent):
        return "f0 tidak terdeteksi"
    if abs(cent) < 5:
        return "PAS"
    return "terlalu rendah, kencangkan senar" if cent < 0 else "terlalu tinggi, kendurkan senar"


# ----------------------------------------------------------------------------- data NSynth
def ekstrak_nsynth(folder=".", log=print):
    """Ekstrak examples.json + nada gitar MIDI 40-68 dari arsip .tar.gz NSynth (sekali saja)."""
    pola_nada = "|".join(f"{p:03d}" for p in NADA_DIBUTUHKAN)
    pola = re.compile(rf"audio/guitar_.*-({pola_nada})-\d{{3}}\.wav$")
    for split in ["nsynth-valid", "nsynth-test"]:
        arsip = os.path.join(folder, f"{split}.jsonwav.tar.gz")
        penanda = os.path.join(folder, split, ".ekstrak_v2_selesai")
        if os.path.exists(penanda) or not os.path.exists(arsip):
            continue
        n = 0
        with tarfile.open(arsip, "r|gz") as tar:
            for m in tar:
                if m.name.endswith("examples.json") or pola.search(m.name):
                    if not os.path.exists(os.path.join(folder, m.name)):
                        tar.extract(m, folder)
                    n += 1
        open(penanda, "w").write("ok")
        log(f"{split}: {n} file diekstrak")


def baca_meta(folder):
    p = os.path.join(folder, "examples.json")
    if not os.path.exists(p):
        return None
    m = pd.DataFrame(json.load(open(p))).T
    m["pitch"], m["velocity"] = m.pitch.astype(int), m.velocity.astype(int)
    m = m[(m.instrument_family_str == "guitar") & m.pitch.isin(NADA_DIBUTUHKAN)].copy()
    m["path"] = [os.path.join(folder, "audio", f"{k}.wav") for k in m.index]
    m = m[[os.path.exists(q) for q in m.path]]
    m["label"] = m.pitch.map(SENAR)
    return m if len(m) else None


def karplus_strong(f0, dur=3.0, redaman=0.996, kecerahan=0.5, amp=1.0, seed=0):
    r = np.random.default_rng(seed)
    N = max(2, int(round(SR / f0 - 0.5)))
    burst = r.uniform(-1, 1, N)
    for _ in range(int(1 + 6 * (1 - kecerahan))):
        burst = 0.5 * (burst + np.roll(burst, 1))
    a = np.zeros(N + 2); a[0], a[N], a[N + 1] = 1, -redaman / 2, -redaman / 2
    x = np.zeros(int(dur * SR)); x[:N] = burst
    y = lfilter([1.0], a, x)
    return amp * y / (np.abs(y).max() + 1e-9)


def buat_data_sintetis(root):
    spesifikasi = {"nsynth-valid": [("guitar_acoustic_s00", "acoustic", 0.996, 0.55)],
                   "nsynth-test": [("guitar_acoustic_s01", "acoustic", 0.994, 0.40),
                                   ("guitar_acoustic_s02", "acoustic", 0.997, 0.70),
                                   ("guitar_electronic_s03", "electronic", 0.998, 0.85),
                                   ("guitar_acoustic_s04", "acoustic", 0.992, 0.30),
                                   ("guitar_synthetic_s05", "synthetic", 0.995, 0.60)]}
    if all(os.path.exists(os.path.join(root, s, "examples.json")) for s in spesifikasi):
        return root
    for split, instrumen in spesifikasi.items():
        os.makedirs(os.path.join(root, split, "audio"), exist_ok=True)
        meta, seed = {}, 0
        for nama, sumber, red, cer in instrumen:
            for p in NADA_DIBUTUHKAN:
                for vel in [60, 100]:
                    seed += 1
                    y = karplus_strong(midi_ke_hz(p), redaman=red, kecerahan=min(1, cer + (vel - 75) / 250),
                                       amp=0.3 + 0.6 * vel / 100, seed=seed + 1000 * len(split))
                    kunci = f"{nama}-{p:03d}-{vel:03d}"
                    wavfile.write(os.path.join(root, split, "audio", kunci + ".wav"), SR,
                                  (y * 32767 * 0.9).astype(np.int16))
                    meta[kunci] = {"note_str": kunci, "instrument_str": nama, "pitch": p, "velocity": vel,
                                   "sample_rate": SR, "instrument_family_str": "guitar",
                                   "instrument_source_str": sumber}
        json.dump(meta, open(os.path.join(root, split, "examples.json"), "w"))
    return root


def pilih_template(meta):
    m = meta.assign(bukan_akustik=(meta.instrument_source_str != "acoustic").astype(int),
                    selisih_vel=(meta.velocity - 75).abs())
    lengkap = m.groupby("instrument_str").pitch.nunique()
    lengkap = lengkap[lengkap == len(PITCH)].index
    if len(lengkap):
        inst = m[m.instrument_str.isin(lengkap)].sort_values(["bukan_akustik", "instrument_str"]).instrument_str.iloc[0]
        m = m[m.instrument_str == inst]
    return m.sort_values(["bukan_akustik", "selisih_vel"]).groupby("pitch").head(1).set_index("pitch").loc[PITCH]


# ----------------------------------------------------------------------------- perakitan chord
class _Pemuat:
    def __init__(self): self.cache = {}
    def __call__(self, path):
        if path not in self.cache:
            self.cache[path] = muat(path)
        return self.cache[path]


def punya(meta, inst, pitches):
    return set(pitches) <= set(meta.pitch[meta.instrument_str == inst])


def genjreng(nada, jeda, arah="turun", durasi=3.0, rng=None, acak_amp=0.0):
    urut = list(nada) if arah == "turun" else list(nada)[::-1]
    L = int(durasi * SR); y = np.zeros(L)
    for k, x in enumerate(urut):
        s = int(round(k * jeda * SR))
        if s >= L:
            break
        g = 1.0 + (rng.uniform(-acak_amp, acak_amp) if rng is not None and acak_amp else 0.0)
        seg = x[: L - s]; y[s:s + len(seg)] += g * seg
    return y / (np.abs(y).max() + 1e-9)


def rakit(meta, pemuat, inst, pitches, vel, jeda, arah="turun", rng=None, acak_amp=0.0):
    nada = []
    for p in pitches:
        m = meta[(meta.instrument_str == inst) & (meta.pitch == p)]
        if m.empty:
            return None
        nada.append(pemuat(m.path.iloc[int(np.argmin(np.abs(m.velocity.to_numpy() - vel)))]))
    return genjreng(nada, jeda, arah, rng=rng, acak_amp=acak_amp)


def pilih_gitar_template(meta):
    urut = (meta.drop_duplicates("instrument_str")
            .assign(bukan_akustik=lambda d: (d.instrument_source_str != "acoustic").astype(int))
            .sort_values(["bukan_akustik", "instrument_str"]).instrument_str)
    for inst in urut:
        if all(punya(meta, inst, CHORD[c]) for c in NAMA_CHORD):
            return {c: inst for c in NAMA_CHORD}
    hasil = {}
    for c in NAMA_CHORD:
        cocok = [i for i in urut if punya(meta, i, CHORD[c])]
        if not cocok:
            raise ValueError(f"Tidak ada gitar di pool template yang punya semua nada chord {c}")
        hasil[c] = cocok[0]
    return hasil


POLA_BINER = {c: np.isin(np.arange(12), list(pc(CHORD[c]))).astype(float) for c in NAMA_CHORD}


def kos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def prediksi_biner(F_chroma):
    v = F_chroma[:N_AWAL].mean(0)
    return NAMA_CHORD[int(np.argmax([kos(v, POLA_BINER[c]) for c in NAMA_CHORD]))]


def profil_bersih(x):
    v = fitur_chroma(x)[:N_AWAL].mean(0)
    v = np.clip(v - np.median(v), 0, None)
    return v / (v.max() + 1e-12)


# ----------------------------------------------------------------------------- statistik
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z ** 2 / n
    pusat = (p + z ** 2 / (2 * n)) / den
    lebar = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return pusat - lebar, pusat + lebar


def metrik_per_kelas(y, yhat, kelas, nama):
    baris = []
    for k, nm in zip(kelas, nama):
        tp = np.sum((yhat == k) & (y == k)); fp = np.sum((yhat == k) & (y != k)); fn = np.sum((yhat != k) & (y == k))
        pre = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        baris.append({"kelas": nm, "presisi": pre, "recall": rec,
                      "F1": 2 * pre * rec / (pre + rec) if pre + rec else 0.0, "jumlah": int(np.sum(y == k))})
    return pd.DataFrame(baris).set_index("kelas")


def tabel_akurasi(y, prediksi, kelas, nama):
    baris = []
    for m, yhat in prediksi.items():
        k, n = int(np.sum(yhat == y)), len(y)
        lo, hi = wilson(k, n)
        pk = metrik_per_kelas(y, yhat, kelas, nama)
        baris.append({"metode": m, "akurasi": k / n, "CI 95% bawah": lo, "CI 95% atas": hi,
                      "F1 makro": pk.loc[pk.jumlah > 0, "F1"].mean(),   # hanya kelas yang ada di data uji
                      "benar / total": f"{k} / {n}"})
    return pd.DataFrame(baris).set_index("metode").sort_values("akurasi", ascending=False)


def mcnemar(y, a, b):
    b_, c_ = int(np.sum((a == y) & (b != y))), int(np.sum((a != y) & (b == y)))
    p = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
    return b_, c_, p


def tabel_mcnemar(y, prediksi, pasangan):
    baris = []
    for a, b in pasangan:
        b_, c_, p = mcnemar(y, prediksi[a], prediksi[b])
        baris.append({"metode A": a, "metode B": b, "A benar, B salah": b_, "A salah, B benar": c_,
                      "p-value": p, "kesimpulan (α = 5%)": ("A lebih baik" if b_ > c_ else "B lebih baik")
                      if p < 0.05 else "tidak berbeda nyata"})
    return pd.DataFrame(baris)


def auc(skor, label):
    s1, s0 = skor[label == 1], skor[label == 0]
    return float(((s1[:, None] > s0[None]).sum() + 0.5 * (s1[:, None] == s0[None]).sum()) / (len(s1) * len(s0)))


def ambang_youden(skor, label):
    kandidat = np.unique(skor)
    j = [np.mean(skor[label == 1] > t) + np.mean(skor[label == 0] <= t) - 1 for t in kandidat]
    return kandidat[int(np.argmax(j))]


def kurva_roc(skor, label):
    t = np.r_[np.inf, np.sort(np.unique(skor))[::-1], -np.inf]
    return (np.array([np.mean(skor[label == 0] > v) for v in t]),
            np.array([np.mean(skor[label == 1] > v) for v in t]))


# ----------------------------------------------------------------------------- pipeline analisis
def jalankan_analisis(folder=".", maks_per_nada=40, log=print):
    """Menjalankan seluruh analisis Bagian 1 (tuner) dan Bagian 2 (chord) seperti di notebook.
    Mengembalikan dict berisi tabel hasil, template, dan contoh untuk visualisasi."""
    t_mulai = time.time()
    ekstrak_nsynth(folder, log)
    meta_v, meta_t = baca_meta(os.path.join(folder, "nsynth-valid")), baca_meta(os.path.join(folder, "nsynth-test"))
    if meta_v is not None or meta_t is not None:
        mode = "nsynth"
        if meta_v is None or meta_t is None:
            semua = meta_v if meta_v is not None else meta_t
            inst = sorted(semua.instrument_str.unique()); k = max(1, len(inst) // 3)
            meta_v, meta_t = semua[semua.instrument_str.isin(inst[:k])], semua[semua.instrument_str.isin(inst[k:])]
    else:
        mode = "sintetis"
        root = buat_data_sintetis(os.path.join(folder, "data_sintetis_v2"))
        meta_v, meta_t = baca_meta(f"{root}/nsynth-valid"), baca_meta(f"{root}/nsynth-test")
    log(f"Data: {mode}")
    pemuat = _Pemuat()
    R = {"mode": mode, "dibuat": time.strftime("%Y-%m-%d %H:%M")}

    # ======================= BAGIAN 1: TUNER =======================
    mv, mt = meta_v[meta_v.pitch.isin(PITCH)], meta_t[meta_t.pitch.isin(PITCH)]
    templ = pilih_template(mv)
    uji = (mt.sample(frac=1, random_state=0).groupby("pitch").head(maks_per_nada)
           .sort_values(["pitch", "instrument_str", "velocity"]))
    rng = np.random.default_rng(7)
    audio_templ = {p: pangkas_sunyi(pemuat(templ.loc[p, "path"])) for p in PITCH}
    data_uji = []
    for k, r in uji.iterrows():
        x = pangkas_sunyi(pemuat(r.path))
        for v in VARIASI:
            data_uji.append({"id": k, "pitch": r.pitch, "label": r.label, "instrumen": r.instrument_str,
                             "sumber": r.instrument_source_str, "velocity": r.velocity, "variasi": v,
                             "x": variasi(x, v, rng)})
    log(f"Bagian 1: {len(data_uji)} bunyi uji")
    F_templ = {nf: [f(audio_templ[p]) for p in PITCH] for nf, f in FITUR.items()}
    F_uji = {nf: [f(d["x"]) for d in data_uji] for nf, f in FITUR.items()}
    hasil = pd.DataFrame([{k: d[k] for k in ["id", "pitch", "label", "instrumen", "sumber", "velocity", "variasi"]}
                          for d in data_uji])
    for nf in FITUR:
        for nj, fj in JARAK.items():
            hasil[f"{nf} | {nj}"] = [PITCH[int(np.argmin([fj(T, F) for T in F_templ[nf]]))] for F in F_uji[nf]]
    hasil["YIN (baseline tuner)"] = [prediksi_yin(d["x"]) for d in data_uji]
    metode1 = [c for c in hasil.columns if "|" in c or c.startswith("YIN")]
    y1 = hasil.pitch.to_numpy(); pred1 = {m: hasil[m].to_numpy() for m in metode1}

    # cent dari tuner (DTW semiton menentukan senar, YIN mengukur f0)
    cent = []
    for d, p in zip(data_uji, hasil["semiton | DTW"]):
        f0 = yin_f0(d["x"])
        if np.isfinite(f0):
            f0 = f0 * 2.0 ** np.round(np.log2(midi_ke_hz(p) / f0))
        cent.append(cent_dari(f0, midi_ke_hz(p)) if np.isfinite(f0) else np.nan)
    hasil["cent"] = cent

    sakoe1 = []
    for band in [None, 10, 5, 2, 1]:
        t0 = time.time()
        pred = np.array([PITCH[int(np.argmin([dtw_norm(T, F, band) for T in F_templ["semiton"]]))] for F in F_uji["semiton"]])
        pendek = (hasil.variasi == "pendek").to_numpy()
        sakoe1.append({"jendela (frame)": "tanpa" if band is None else str(band), "akurasi": (pred == y1).mean(),
                       "akurasi 'pendek'": (pred[pendek] == y1[pendek]).mean(), "waktu (detik)": time.time() - t0})
    pasangan1 = [("semiton | DTW", "semiton | Euclid resampling"), ("semiton | DTW", "semiton | Euclid padding"),
                 ("semiton | DTW", "chroma | DTW"), ("semiton | DTW", "MFCC | DTW"), ("semiton | DTW", "YIN (baseline tuner)")]
    d_pendek = next(i for i, d in enumerate(data_uji) if d["pitch"] == 45 and d["variasi"] == "pendek")
    R["tuner"] = {
        "hasil": hasil, "metode": metode1,
        "akurasi": tabel_akurasi(y1, pred1, PITCH, LABEL),
        "per_kelas": {m: metrik_per_kelas(y1, pred1[m], PITCH, LABEL) for m in metode1},
        "mcnemar": tabel_mcnemar(y1, pred1, pasangan1),
        "mcnemar_pendek": tabel_mcnemar(y1[hasil.variasi.to_numpy() == "pendek"],
                                        {m: v[hasil.variasi.to_numpy() == "pendek"] for m, v in pred1.items()}, pasangan1[:2]),
        "sakoe": pd.DataFrame(sakoe1),
        "template": templ[["label", "instrument_str", "velocity"]].copy(),
        "n_rekaman": len(uji), "n_gitar_uji": uji.instrument_str.nunique(),
        "jumlah_uji": mt.groupby(["label", "instrument_source_str"]).size().unstack(fill_value=0).reindex(LABEL, fill_value=0),
        "contoh_pendek": {"F": F_uji["semiton"][d_pendek]},
    }

    # ======================= BAGIAN 2: CHORD =======================
    inst_templ2 = pilih_gitar_template(meta_v)
    audio_templ2 = {c: pangkas_sunyi(rakit(meta_v, pemuat, inst_templ2[c], CHORD[c], 75, 0.025)) for c in NAMA_CHORD}
    rng2 = np.random.default_rng(2026)
    data2, salah2 = [], []
    for inst in sorted(meta_t.instrument_str.unique()):
        for vel in [50, 100]:
            for c in NAMA_CHORD:
                if not punya(meta_t, inst, CHORD[c]):
                    continue
                info = {"chord": c, "instrumen": inst, "velocity": vel}
                for v in VARIASI2:
                    jeda = rng2.uniform(0.07, 0.10) if v == "lambat" else rng2.uniform(0.01, 0.04)
                    x = rakit(meta_t, pemuat, inst, CHORD[c], vel, jeda, "naik" if v == "naik" else "turun", rng2, 0.2)
                    if v == "derau":
                        x = x + rng2.normal(0, np.sqrt(np.mean(x ** 2) / 10 ** 1.5), len(x))
                    data2.append({**info, "variasi": v, "x": pangkas_sunyi(x)})
                senar = list(CHORD[c])
                for jenis in ["nada salah", "nada salah", "senar mati", "senar mati"]:
                    for _ in range(20):
                        i = int(rng2.integers(len(senar))); p_baru = senar.copy()
                        if jenis == "nada salah":
                            geser = senar[i] + int(rng2.choice([-1, 1]))
                            if geser % 12 in pc(senar) or geser not in NADA_DIBUTUHKAN:
                                continue
                            p_baru[i] = geser
                        else:
                            del p_baru[i]
                        x = rakit(meta_t, pemuat, inst, p_baru, vel, rng2.uniform(0.01, 0.04),
                                  str(rng2.choice(["turun", "naik"])), rng2, 0.2)
                        if x is None:
                            continue
                        salah2.append({**info, "variasi": jenis, "x": pangkas_sunyi(x)})
                        break
    log(f"Bagian 2: {len(data2)} chord benar, {len(salah2)} chord salah")
    F_templ2 = {nf: [f(audio_templ2[c]) for c in NAMA_CHORD] for nf, f in FITUR.items()}
    F_uji2 = {nf: [f(d["x"]) for d in data2] for nf, f in FITUR.items()}
    hasil2 = pd.DataFrame([{k: d[k] for k in ["chord", "instrumen", "velocity", "variasi"]} for d in data2])
    for nf in FITUR:
        for nj, fj in JARAK.items():
            hasil2[f"{nf} | {nj}"] = [NAMA_CHORD[int(np.argmin([fj(T, F) for T in F_templ2[nf]]))] for F in F_uji2[nf]]
    hasil2["template biner (baseline)"] = [prediksi_biner(F) for F in F_uji2["chroma"]]
    metode2 = [c for c in hasil2.columns if "|" in c or c.startswith("template")]
    y2 = hasil2.chord.to_numpy(); pred2 = {m: hasil2[m].to_numpy() for m in metode2}
    lambat = (hasil2.variasi == "lambat").to_numpy()

    # verifikasi
    def skor_verifikasi(x, c):
        Fc, Fs = fitur_chroma(x), fitur_semiton(x); i = NAMA_CHORD.index(c)
        jarak_semua = [dtw_norm(T, Fc) for T in F_templ2["chroma"]]
        return {"chroma | DTW": jarak_semua[i],
                "chroma | Euclid resampling": euclid_resample(F_templ2["chroma"][i], Fc),
                "semiton | DTW": dtw_norm(F_templ2["semiton"][i], Fs),
                "template biner (baseline)": 1 - kos(Fc[:N_AWAL].mean(0), POLA_BINER[c]),
                "terdekat = target": int(int(np.argmin(jarak_semua)) == i)}
    ver = pd.DataFrame([{"chord": d["chord"], "instrumen": d["instrumen"], "variasi": d["variasi"], "salah": lab,
                         **skor_verifikasi(d["x"], d["chord"])}
                        for kumpulan, lab in [(data2, 0), (salah2, 1)] for d in kumpulan])
    skor_list = ["chroma | DTW", "chroma | Euclid resampling", "semiton | DTW", "template biner (baseline)"]
    kelompok = {inst: i % 2 for i, inst in enumerate(sorted(ver.instrumen.unique()))}
    ver["kelompok"] = ver.instrumen.map(kelompok)
    lintas = ver.kelompok.nunique() > 1
    y_ver = ver.salah.to_numpy()
    for s in skor_list:
        ver[f"tolak | {s}"] = 0
        for g in ver.kelompok.unique():
            latih = ver[ver.kelompok != g] if lintas else ver
            t = ambang_youden(latih[s].to_numpy(), latih.salah.to_numpy())
            ver.loc[ver.kelompok == g, f"tolak | {s}"] = (ver.loc[ver.kelompok == g, s] > t).astype(int)
    ver["tolak | aturan terdekat = target"] = 1 - ver["terdekat = target"]
    baris = []
    for s in skor_list + ["aturan terdekat = target"]:
        tolak = ver[f"tolak | {s}"].to_numpy()
        k_sen, n_sen = int(((tolak == 1) & (y_ver == 1)).sum()), int((y_ver == 1).sum())
        k_spe, n_spe = int(((tolak == 0) & (y_ver == 0)).sum()), int((y_ver == 0).sum())
        baris.append({"skor": s, "ROC-AUC": auc(ver[s].to_numpy(), y_ver) if s in skor_list else np.nan,
                      "sensitivitas": k_sen / n_sen, "CI sens. bawah": wilson(k_sen, n_sen)[0],
                      "CI sens. atas": wilson(k_sen, n_sen)[1],
                      "spesifisitas": k_spe / n_spe, "CI spes. bawah": wilson(k_spe, n_spe)[0],
                      "CI spes. atas": wilson(k_spe, n_spe)[1],
                      "akurasi verifikasi": (k_sen + k_spe) / (n_sen + n_spe)})
    verif = pd.DataFrame(baris).set_index("skor")
    rinci = pd.DataFrame({s: ver.groupby("variasi")[f"tolak | {s}"].mean()
                          for s in skor_list + ["aturan terdekat = target"]}).reindex(VARIASI2 + ["nada salah", "senar mati"])
    roc = {s: kurva_roc(ver[s].to_numpy(), y_ver) for s in skor_list}

    sakoe2 = []
    for band in [None, 10, 5, 2, 1]:
        t0 = time.time()
        pred = np.array([NAMA_CHORD[int(np.argmin([dtw_norm(T, F, band) for T in F_templ2["chroma"]]))] for F in F_uji2["chroma"]])
        sakoe2.append({"jendela (frame)": "tanpa" if band is None else str(band), "akurasi": (pred == y2).mean(),
                       "akurasi 'lambat'": (pred[lambat] == y2[lambat]).mean(), "waktu (detik)": time.time() - t0})
    pasangan2 = [("chroma | DTW", "chroma | Euclid resampling"), ("chroma | DTW", "chroma | Euclid padding"),
                 ("chroma | DTW", "semiton | DTW"), ("chroma | DTW", "MFCC | DTW"), ("chroma | DTW", "template biner (baseline)")]
    i_g = next((i for i, d in enumerate(data2) if d["chord"] == "G" and d["variasi"] == "lambat"), None)
    if i_g is None:
        i_g = next(i for i, d in enumerate(data2) if d["variasi"] == "lambat")
    R["chord"] = {
        "hasil": hasil2, "metode": metode2,
        "akurasi": tabel_akurasi(y2, pred2, NAMA_CHORD, NAMA_CHORD),
        "per_kelas": {m: metrik_per_kelas(y2, pred2[m], NAMA_CHORD, NAMA_CHORD) for m in metode2},
        "mcnemar": tabel_mcnemar(y2, pred2, pasangan2),
        "mcnemar_lambat": tabel_mcnemar(y2[lambat], {m: v[lambat] for m, v in pred2.items()}, pasangan2[:2]),
        "sakoe": pd.DataFrame(sakoe2), "verif": verif, "rinci": rinci, "roc": roc,
        "skor_ver": ver[["chord", "instrumen", "variasi", "salah"] + skor_list],
        "template_gitar": sorted(set(inst_templ2.values())), "n_gitar_uji": hasil2.instrumen.nunique(),
        "n_salah": len(salah2),
        "contoh_lambat": {"chord": data2[i_g]["chord"], "F": F_uji2["chroma"][i_g]},
    }

    # template untuk aplikasi rekam langsung
    R["app"] = {
        "F_templ": {nf: [np.asarray(F, np.float32) for F in v] for nf, v in F_templ.items()},
        "F_templ2": {nf: [np.asarray(F, np.float32) for F in v] for nf, v in F_templ2.items()},
        "audio_templ": {p: np.asarray(a, np.float32) for p, a in audio_templ.items()},
        "audio_templ2": {c: np.asarray(a, np.float32) for c, a in audio_templ2.items()},
    }
    R["durasi_detik"] = time.time() - t_mulai
    log(f"Selesai dalam {R['durasi_detik']:.0f} detik")
    return R


def simpan_hasil(R, path="hasil_analisis.pkl"):
    with open(path, "wb") as f:
        pickle.dump(R, f)


def muat_hasil(path="hasil_analisis.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- aplikasi rekam langsung
def tuner_live(x, A, senar_manual=None):
    """x: audio mentah. A: R['app']. senar_manual: nama senar (mis. 'A2') untuk mode manual."""
    x = bersihkan(pangkas_sunyi(np.asarray(x, float)))
    F = fitur_semiton(x)
    jarak = np.array([dtw_norm(T, F) for T in A["F_templ"]["semiton"]])
    p_dtw = PITCH[int(np.argmin(jarak))]
    p = {v: k for k, v in SENAR.items()}[senar_manual] if senar_manual else p_dtw
    f0 = f0_harmonik(x, midi_ke_hz(p), lebar_cent=150 if senar_manual else 100)
    c = cent_dari(f0, midi_ke_hz(p))
    _, _, jalur = dtw(A["F_templ"]["semiton"][PITCH.index(p)], F)
    return {"x": x, "F": F, "jarak": jarak, "pitch_dtw": p_dtw, "pitch": p, "f0": f0, "cent": c,
            "status": status_cent(c), "jalur": jalur, "yin_lebar": yin_f0(x)}


def cek_chord_live(x, target, A, b_kurang=0.3, b_lebih=0.5):
    x = bersihkan(pangkas_sunyi(np.asarray(x, float)))
    Fc = fitur_chroma(x)
    jarak = np.array([dtw_norm(T, Fc) for T in A["F_templ2"]["chroma"]])
    terdekat = NAMA_CHORD[int(np.argmin(jarak))]
    v = profil_bersih(x); nada = pc(CHORD[target])
    kurang = [NAMA_NADA[k] for k in sorted(nada) if v[k] < b_kurang]
    lebih = [NAMA_NADA[k] for k in range(12) if k not in nada and v[k] > b_lebih]
    _, _, jalur = dtw(A["F_templ2"]["chroma"][NAMA_CHORD.index(target)], Fc)
    return {"x": x, "F": Fc, "jarak": jarak, "terdekat": terdekat, "benar": terdekat == target,
            "profil": v, "kurang": kurang, "lebih": lebih, "jalur": jalur}
