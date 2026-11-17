"""
Dashboard Streamlit: Guitar Tuner & Cek Chord Gitar berbasis DTW
Jalankan dari folder project (yang berisi nsynth-valid/ dan nsynth-test/ atau arsip .tar.gz-nya):
    streamlit run app.py
"""
import os, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import tuner_core as tc

st.set_page_config(page_title="DTW Guitar Lab", page_icon="🎸", layout="wide", initial_sidebar_state="expanded")

# ---- Desain A "Studio Terang": tanah abu kebiruan, kartu putih, aksen ungu
UNGU_UTAMA, UNGU_MUDA, UNGU_TEKS, LATAR, TEKS, REDUP = "#5B3FE0", "#EEEAFF", "#4A31C7", "#F3F4FA", "#1B1D29", "#4B5563"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp { background: #F3F4FA; font-family: Inter, system-ui, sans-serif; color: #1B1D29; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Space Grotesk', sans-serif; color: #1B1D29; letter-spacing: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 0; }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; border: 0; border-radius: 12px; min-height: 44px;
  background: transparent; color: #4B5563; font-weight: 500; padding-left: 14px; }
[data-testid="stSidebar"] .stButton button:hover { background: #F3F4FA; color: #5B3FE0; }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] { background: #EEEAFF; color: #4A31C7; font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 16px; border-color: transparent !important; }
[data-testid="stExpander"] { background: #FFFFFF; border-radius: 16px; border: 0; }
[data-testid="stMetric"] { background: #FFFFFF; border-radius: 16px; padding: 14px 16px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 14px; }
.gl-logo { display: flex; align-items: center; gap: 10px; margin: 4px 0 18px; }
.gl-logo span { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 19px; color: #1B1D29; }
.gl-kpis { display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 18px; }
.gl-kpi { flex: 1 1 190px; background: #FFFFFF; border-radius: 16px; padding: 18px; display: flex; flex-direction: column; gap: 6px; }
.gl-kpi .l, .gl-kpi .s { font-size: 13px; color: #4B5563; } .gl-kpi .s { font-size: 12px; }
.gl-kpi .v { font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; color: #1B1D29; line-height: 1.1; }
.gl-badge { display: inline-block; background: #FFFFFF; border-radius: 999px; padding: 8px 14px; font-size: 13px; color: #4B5563; }
.gl-sub { font-size: 14px; color: #4B5563; margin: -8px 0 14px; }
.gl-bars { display: flex; flex-direction: column; gap: 10px; font-size: 13px; }
.gl-bar { display: flex; align-items: center; gap: 12px; } .gl-bar .n { flex: 0 0 200px; } .gl-bar .p { flex: 0 0 56px; text-align: right; font-weight: 600; }
.gl-bar .t { flex: 1; height: 14px; background: #EEF0F6; border-radius: 7px; } .gl-bar .t div { height: 14px; border-radius: 7px; }
.gl-last { background: #5B3FE0; color: #FFFFFF; border-radius: 18px; padding: 18px; display: flex; flex-direction: column; gap: 8px; margin-top: 22px; }
.gl-last .k { font-size: 12px; color: #E1DBFF; } .gl-last .b { font-family: 'Space Grotesk', sans-serif; font-size: 34px; font-weight: 700; line-height: 1; }
.gl-last .d { font-size: 13px; }
.gl-pill { display: inline-block; align-self: flex-start; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 999px; }
.gl-ok { background: #E3F5EA; color: #17663A; } .gl-warn { background: #FFF1D6; color: #7A4D00; } .gl-bad { background: #FDE4E4; color: #9B1C1C; }
.gl-last .gl-pill { background: #FFFFFF; }
.gl-big { font-family: 'Space Grotesk', sans-serif; font-size: 56px; font-weight: 700; line-height: 1; color: #1B1D29; }
.gl-res { display: flex; flex-direction: column; gap: 6px; } .gl-res .m { font-size: 13px; color: #4B5563; } .gl-res .f { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

FOLDER_DATA = "."
FILE_HASIL = "hasil_analisis.pkl"
BIRU, JINGGA, HIJAU, MERAH, ABU, UNGU = "#5B3FE0", "#F4B740", "#1F9D55", "#C62828", "#9AA1B5", "#A594F5"
PUDAR = "#DAD4FB"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#1B1D29",
                     "axes.labelcolor": "#4B5563", "xtick.color": "#4B5563", "ytick.color": "#4B5563", "axes.edgecolor": "#C9CEDD"})


# =============================================================================== data hasil analisis
@st.cache_resource(show_spinner=False)
def muat_hasil_cache(path, mtime):
    return tc.muat_hasil(path)


def ambil_hasil():
    if not os.path.exists(FILE_HASIL):
        with st.status("Menjalankan analisis pertama kali (hanya sekali, hasilnya disimpan)...", expanded=True) as s:
            R = tc.jalankan_analisis(FOLDER_DATA, log=s.write)
            tc.simpan_hasil(R, FILE_HASIL)
            s.update(label="Analisis selesai", state="complete")
    return muat_hasil_cache(FILE_HASIL, os.path.getmtime(FILE_HASIL))


R = ambil_hasil()
T1, T2, A = R["tuner"], R["chord"], R["app"]


# =============================================================================== helper
def tampil(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def pct(v):
    return f"{v:.1%}" if pd.notna(v) else "-"


def akurasi_per_variasi(hasil, metode, kolom_benar, variasi):
    tab = pd.DataFrame({v: [(hasil.loc[hasil.variasi == v, m] == hasil.loc[hasil.variasi == v, kolom_benar]).mean()
                            for m in metode] for v in variasi}, index=metode)
    tab["rata-rata"] = [(hasil[m] == hasil[kolom_benar]).mean() for m in metode]
    return tab


def fmt_tabel(df, persen=(), desimal=(), p=()):
    f = {c: "{:.1%}" for c in persen if c in df.columns}
    f.update({c: "{:.3f}" for c in desimal if c in df.columns})
    f.update({c: "{:.4f}" for c in p if c in df.columns})
    return df.style.format(f, na_rep="-")


def fig_bar_variasi(tab, variasi, judul, acak, baseline=None):
    fitur = ["semiton", "chroma", "MFCC"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for a, nf in zip(ax, fitur):
        for i, (nj, w) in enumerate(zip(["Euclid padding", "Euclid resampling", "DTW"], ["#C9CEDD", UNGU, BIRU])):
            a.bar(np.arange(len(variasi)) + (i - 1) * 0.25, tab.loc[f"{nf} | {nj}", variasi], 0.25, color=w, label=nj)
        if baseline:
            a.plot(range(len(variasi)), tab.loc[baseline, variasi], "D--", color=JINGGA, ms=5, label=baseline)
        a.axhline(acak, ls=":", color="gray")
        a.set_xticks(range(len(variasi))); a.set_xticklabels(variasi); a.set_ylim(0, 1.05); a.set_title(f"Fitur: {nf}")
    ax[0].set_ylabel("akurasi")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(judul, y=1.02); fig.tight_layout()
    return fig


def fig_ci(tab):
    t = tab.sort_values("akurasi")
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1))
    warna = [BIRU if "DTW" in m else JINGGA if ("YIN" in m or "biner" in m) else ABU for m in t.index]
    ax.errorbar(t.akurasi, range(len(t)), xerr=[t.akurasi - t["CI 95% bawah"], t["CI 95% atas"] - t.akurasi],
                fmt="none", ecolor="gray", capsize=3)
    ax.scatter(t.akurasi, range(len(t)), c=warna, zorder=3, s=40)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.set_xlim(0, 1.02)
    ax.set_xlabel("akurasi (titik) dan selang kepercayaan 95% Wilson (garis)")
    fig.tight_layout()
    return fig


def fig_konfusi(benar, prediksi, kelas, judul):
    K = pd.crosstab(pd.Series(benar, name="b"), pd.Series(prediksi, name="p")).reindex(index=kelas, columns=kelas, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(K.values, cmap="Purples")
    for i in range(len(kelas)):
        for j in range(len(kelas)):
            if K.values[i, j]:
                ax.text(j, i, K.values[i, j], ha="center", va="center", fontsize=8,
                        color="white" if K.values[i, j] > K.values.max() / 2 else "black")
    ax.set_xticks(range(len(kelas))); ax.set_xticklabels(kelas); ax.set_yticks(range(len(kelas))); ax.set_yticklabels(kelas)
    ax.set_xlabel("prediksi"); ax.set_ylabel("sebenarnya"); ax.set_title(judul, fontsize=9)
    fig.tight_layout()
    return fig


def fig_warping(T, F, jalur, judul, label_y="frame template", label_x="frame uji"):
    C = np.sqrt(((T[:, None] - F[None]) ** 2).sum(-1))
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.imshow(C, origin="lower", aspect="auto", cmap="Purples_r")
    if len(jalur):
        ax.plot(jalur[:, 1], jalur[:, 0], color=JINGGA, lw=2.5)
    ax.set_title(judul, fontsize=9); ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    fig.tight_layout()
    return fig


def fig_fitur(F, judul, sumbu="semiton"):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.imshow(F.T, origin="lower", aspect="auto", cmap="magma")
    if sumbu == "semiton":
        idx = [k for k, m in enumerate(tc.MIDI_BIN) if m % 12 == 0]
        ax.set_yticks(idx); ax.set_yticklabels([tc.nama_nada(tc.MIDI_BIN[k]) for k in idx])
    else:
        ax.set_yticks(range(12)); ax.set_yticklabels(tc.NAMA_NADA, fontsize=7)
    ax.set_xlabel("frame (≈0,13 detik)"); ax.set_title(judul, fontsize=9)
    fig.tight_layout()
    return fig


def svg_img(svg, lebar, alt=""):
    b = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b}" width="{lebar}" alt="{alt}" style="max-width:100%">'


def svg_gauge(cent):
    c = float(np.clip(cent, -50, 50)) if np.isfinite(cent) else 0.0
    th = np.deg2rad(180 - (c + 50) * 1.8)
    x, y = 100 + 66 * np.cos(th), 100 - 66 * np.sin(th)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 124">
<path d="M20 100A80 80 0 0 1 63.7 28.7" fill="none" stroke="#E4E6EF" stroke-width="12"/>
<path d="M63.7 28.7A80 80 0 0 1 87.5 21" fill="none" stroke="#F4B740" stroke-width="12"/>
<path d="M87.5 21A80 80 0 0 1 112.5 21" fill="none" stroke="#1F9D55" stroke-width="12"/>
<path d="M112.5 21A80 80 0 0 1 136.3 28.7" fill="none" stroke="#F4B740" stroke-width="12"/>
<path d="M136.3 28.7A80 80 0 0 1 180 100" fill="none" stroke="#E4E6EF" stroke-width="12"/>
<path d="M100 100L{x:.1f} {y:.1f}" stroke="#1B1D29" stroke-width="3" stroke-linecap="round"/>
<circle cx="100" cy="100" r="6" fill="#1B1D29"/>
<g font-family="Arial, sans-serif" font-size="9" fill="#4B5563" text-anchor="middle">
<text x="20" y="116">-50</text><text x="100" y="12">0</text><text x="180" y="116">+50</text></g></svg>"""


def svg_chord(nama):
    jari = tc.POSISI_JARI[nama].split()
    bag = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 112">',
           '<rect x="20" y="18" width="110" height="4" fill="#1B1D29"/>',
           '<g stroke="#C5C9D6" stroke-width="1.5"><path d="M20 48h110M20 74h110M20 100h110"/></g>',
           '<g stroke="#6B7280" stroke-width="1.5"><path d="M20 20v82M42 20v82M64 20v82M86 20v82M108 20v82M130 20v82"/></g>']
    for i, f in enumerate(jari):
        x = 20 + 22 * i
        if f == "x":
            bag.append(f'<path d="M{x-4} 5l8 8M{x+4} 5l-8 8" stroke="#6B7280" stroke-width="1.5"/>')
        elif f == "0":
            bag.append(f'<circle cx="{x}" cy="9" r="4.5" fill="none" stroke="#6B7280" stroke-width="1.5"/>')
        else:
            bag.append(f'<circle cx="{x}" cy="{9 + 26 * int(f)}" r="8" fill="#5B3FE0"/>')
    return "".join(bag) + "</svg>"


def kpi_html(kartu):
    return '<div class="gl-kpis">' + "".join(
        f'<div class="gl-kpi"><span class="l">{l}</span><span class="v">{v}</span><span class="s">{s}</span></div>'
        for l, v, s in kartu) + "</div>"


def bars_html(baris):
    return '<div class="gl-bars">' + "".join(
        f'<div class="gl-bar"><span class="n">{n}</span><div class="t"><div style="width:{v * 100:.1f}%;background:{w}"></div></div>'
        f'<span class="p">{v:.1%}</span></div>' for n, v, w in baris) + "</div>"


def ci_teks(tab, m):
    return f"CI 95%: {tab.loc[m, 'CI 95% bawah']:.1%} – {tab.loc[m, 'CI 95% atas']:.1%}"


def pill(teks, jenis="ok"):
    return f'<span class="gl-pill gl-{jenis}">{teks}</span>'


def judul(teks, sub=None):
    st.title(teks)
    if sub:
        st.markdown(f'<p class="gl-sub">{sub}</p>', unsafe_allow_html=True)


def fig_jarak(jarak, label, terpilih, judul):
    fig, ax = plt.subplots(figsize=(5, 3))
    warna = [BIRU if l == terpilih else PUDAR for l in label]
    ax.bar(label, jarak, color=warna)
    ax.set_ylabel("jarak DTW (makin kecil makin mirip)"); ax.set_title(judul, fontsize=9)
    fig.tight_layout()
    return fig


def fig_gelombang_spektrum(x, f_target=None, f0=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(np.arange(len(x)) / tc.SR, x, lw=0.5, color=BIRU)
    ax[0].set_xlabel("detik"); ax[0].set_title("Gelombang suara (setelah dipangkas & difilter)", fontsize=9)
    seg = x[: int(1.5 * tc.SR)]
    f = np.fft.rfftfreq(len(seg), 1 / tc.SR); S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    ax[1].semilogy(f, S + 1e-9, color=BIRU, lw=0.8); ax[1].set_xlim(0, 1200); ax[1].set_xlabel("Hz")
    if f_target:
        for k in range(1, 7):
            ax[1].axvline(k * f_target, color=JINGGA, ls=":", lw=1)
    if f0 and np.isfinite(f0):
        ax[1].axvline(f0, color=MERAH, lw=1.2)
    ax[1].set_title("Spektrum 1,5 detik pertama (garis kuning = senar target & harmoniknya, merah = f0 terukur)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_profil(profil, target):
    nada = tc.pc(tc.CHORD[target])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(12), profil, color=[BIRU if k in nada else PUDAR for k in range(12)])
    for k in nada:
        ax.axvspan(k - 0.5, k + 0.5, color=JINGGA, alpha=0.12)
    ax.axhline(0.3, ls=":", color=MERAH, lw=1); ax.axhline(0.5, ls="--", color=ABU, lw=1)
    ax.set_xticks(range(12)); ax.set_xticklabels(tc.NAMA_NADA); ax.set_ylim(0, 1.05)
    ax.set_ylabel("energi relatif"); ax.set_title(f"Profil 12 nada (biru/kuning = nada chord {target})", fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================== sidebar
HALAMAN = [("1. Deskripsi project", ":material/home:", "Deskripsi project"),
           ("2. Hasil analisis: tuner", ":material/bar_chart:", "Hasil analisis tuner"),
           ("3. Hasil analisis: cek chord", ":material/bar_chart:", "Hasil analisis chord"),
           ("4. Coba tuner (rekam)", ":material/mic:", "Coba tuner"),
           ("5. Coba cek chord (rekam)", ":material/music_note:", "Coba cek chord")]
if "hal" not in st.session_state:
    st.session_state["hal"] = HALAMAN[0][0]


def pindah(kunci):
    st.session_state["hal"] = kunci


with st.sidebar:
    st.markdown('<div class="gl-logo">' + svg_img(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30"><rect width="30" height="30" rx="9" fill="#5B3FE0"/>'
        '<path d="M8 19c1.5-5 3-8 4.5-8s2 8 3.5 8 2-5 3-5 1.5 3 3 3" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        30) + '<span>DTW Guitar Lab</span></div>', unsafe_allow_html=True)
    for kunci, ikon, label in HALAMAN:
        st.button(label, icon=ikon, key=f"nav_{kunci[0]}", use_container_width=True, on_click=pindah, args=(kunci,),
                  type="primary" if st.session_state["hal"] == kunci else "secondary")
    kartu_terakhir = st.empty()          # diisi di akhir skrip, setelah rekaman diproses
    st.caption(f"Data analisis: **{R['mode']}** · dibuat {R['dibuat']}")
    if R["mode"] == "sintetis":
        st.warning("Data NSynth tidak ditemukan; hasil memakai data sintetis dan tidak untuk dilaporkan.")
    if st.button("Jalankan ulang analisis", key="ulang"):
        if os.path.exists(FILE_HASIL):
            os.remove(FILE_HASIL)
        st.cache_resource.clear()
        st.rerun()
halaman = st.session_state["hal"]


# =============================================================================== 1. deskripsi
if halaman.startswith("1"):
    kiri, kanan = st.columns([3, 1])
    with kiri:
        judul("Guitar Tuner &amp; Cek Chord", "Dynamic Time Warping untuk mengenali chord gitar · Project ADL")
    with kanan:
        st.markdown(f'<div style="text-align:right;margin-top:18px"><span class="gl-badge">Data: '
                    f'{"NSynth" if R["mode"] == "nsynth" else "sintetis"} · {T1["n_gitar_uji"]} gitar uji</span></div>',
                    unsafe_allow_html=True)
    a1, a2 = T1["akurasi"], T2["akurasi"]
    st.markdown(kpi_html([
        ("Akurasi tuner (semiton + DTW)", pct(a1.loc["semiton | DTW", "akurasi"]), ci_teks(a1, "semiton | DTW")),
        ("Akurasi chord (chroma + DTW)", pct(a2.loc["chroma | DTW", "akurasi"]), ci_teks(a2, "chroma | DTW")),
        ("Bunyi uji tuner", len(T1["hasil"]), f"{T1['n_rekaman']} rekaman × 4 variasi"),
        ("uji chord", len(T2["hasil"]), f"+ {T2['n_salah']} chord sengaja salah")]), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Guitar Tuner")
            st.markdown("Petik **satu senar**, sistem menebak senarnya dan menghitung berapa *cent* lagi nadanya harus "
                        "dinaikkan atau diturunkan.")
            st.markdown(svg_img(svg_gauge(0.7), 200, "Ilustrasi jarum tuner"), unsafe_allow_html=True)
            st.button("Buka tuner", icon=":material/mic:", type="primary", use_container_width=True,
                      on_click=pindah, args=(HALAMAN[3][0],), key="ke_tuner")
    with c2:
        with st.container(border=True):
            st.subheader("Cek Chord Gitar")
            st.markdown("Strumming **satu chord**, sistem menebak chord-nya, menyatakan benar atau salah, dan memberi "
                        "petunjuk nada yang bermasalah.")
            st.markdown(svg_img(svg_chord("G"), 150, "Diagram chord G"), unsafe_allow_html=True)
            st.button("Buka cek chord", icon=":material/music_note:", type="primary", use_container_width=True,
                      on_click=pindah, args=(HALAMAN[4][0],), key="ke_chord")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Akurasi tuner per metode")
            st.markdown(bars_html([("YIN (baseline tuner)", a1.loc["YIN (baseline tuner)", "akurasi"], ABU),
                                   ("Semiton + DTW", a1.loc["semiton | DTW", "akurasi"], BIRU),
                                   ("Semiton + Euclid resampling", a1.loc["semiton | Euclid resampling", "akurasi"], UNGU),
                                   ("Chroma + DTW", a1.loc["chroma | DTW", "akurasi"], BIRU),
                                   ("MFCC + DTW", a1.loc["MFCC | DTW", "akurasi"], BIRU)]), unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.subheader("Akurasi chord per metode")
            st.markdown(bars_html([("Template biner (baseline)", a2.loc["template biner (baseline)", "akurasi"], ABU),
                                   ("Chroma + DTW", a2.loc["chroma | DTW", "akurasi"], BIRU),
                                   ("Chroma + Euclid resampling", a2.loc["chroma | Euclid resampling", "akurasi"], UNGU),
                                   ("Semiton + DTW", a2.loc["semiton | DTW", "akurasi"], BIRU),
                                   ("MFCC + DTW", a2.loc["MFCC | DTW", "akurasi"], BIRU)]), unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Tentang project")
        st.markdown("""
Project ini menerapkan **Dynamic Time Warping (DTW)** untuk dua skenario, sebagai pengembangan studi kasus
DTW Minggu 11 (pengenalan ucapan angka). Menggunakan metode yang sama: **audio → deret vektor fitur → jarak (Euclid vs DTW) → 1-NN ke template**.
terdapat perbedaan yaitu fiturnya. MFCC menangkap warna suara, sedangkan untuk musik yang penting adalah **tinggi nada**.
""")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Bagian 1: Tuner**
- **Kelas**: 6 senar gitar standar (E2, A2, D3, G3, B3, E4)
- **Fitur utama**: semiton (energi per semitone, membedakan oktaf)
- **Baseline**: YIN (estimasi frekuensi dasar)
- **Variasi uji**: asli, derau 15 dB, pendek 0,8 detik, fals ±35 cent
- **Evaluasi**: akurasi + CI 95%, F1 per senar, McNemar, ketepatan cent
""")
        with c2:
            st.markdown("""
**Bagian 2: Cek chord**
- **Kelas**: 8 chord dasar (C, D, E, G, A, Am, Em, Dm)
- **Fitur utama**: chroma (12 nama nada, tanpa oktaf)
- **Baseline**: template chord biner
- **Variasi uji**: Strumming turun, naik, lambat, derau; chord salah (nada salah, senar mati)
- **Evaluasi**: akurasi + CI 95%, McNemar, verifikasi (ROC-AUC, sensitivitas, spesifisitas)
""")
    st.subheader("Alur metode")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#EEEAFF", color="#5B3FE0", fontname="Helvetica"]; edge [color="#6B7280"];
  a [label="Audio\\n(petikan / strumming)"]; b [label="Pra-proses\\npangkas sunyi, filter"];
  c [label="Fitur per frame\\nsemiton / chroma / MFCC"]; d [label="Jarak ke template\\nEuclid atau DTW"];
  e [label="1-NN\\ntemplate terdekat"]; f [label="Senar + cent\\n(tuner)"]; g [label="Chord + benar/salah\\n(cek chord)"];
  a -> b -> c -> d -> e; e -> f; e -> g;
}""")
    st.subheader("DTW dalam satu rumus")
    st.latex(r"D(i,j)=c(i,j)+\min\{D(i-1,j-1),\ D(i-1,j),\ D(i,j-1)\},\qquad c(i,j)=\lVert \mathbf{x}_i-\mathbf{y}_j\rVert")
    st.markdown("Jarak DTW dihitung dengan membaginya menggunakan $n+m$. Berbeda dengan Euclidean yang membandingkan setiap frame secara berurutan, "
                "DTW memungkinkan pencocokan frame yang lebih fleksibel. Dengan begitu, perbedaan tempo atau durasi suara tidak terlalu memengaruhi hasil perbandingan.")
    st.subheader("Data")
    st.markdown("""
**NSynth** (Google Magenta, CC BY 4.0): nada tunggal berdurasi 4 detik dari ribuan instrumen. Template diambil dari satu gitar di split
*valid*, data uji dari gitar lain di split *test* (instrumen tidak tumpang tindih). Strumming chord dirakit dari nada-nada gitar
NSynth dengan jeda 10–100 ms antarsenar.
""")
    with st.expander("Cara memakai dashboard"):
        st.markdown("""
- **Halaman 2 & 3**: hasil evaluasi (akurasi, selang kepercayaan, matriks konfusi, uji statistik).
- **Halaman 4**: tekan tombol mikrofon, petik satu senar ±2 detik, lalu hentikan. Hasil senar dan cent muncul otomatis.
- **Halaman 5**: mode *Otomatis* menebak chord yang kamu mainkan; mode *Manual* memeriksa chord tujuan yang dipilih.
- Tombol **Rekam lagi** di halaman 4 dan 5 menghapus rekaman sebelumnya agar bisa merekam ulang.
- Bila mikrofon tidak bisa dipakai, unggah file **WAV** sebagai gantinya.
""")


# =============================================================================== 2. hasil tuner
elif halaman.startswith("2"):
    judul("Hasil analisis tuner")
    hasil, acc = T1["hasil"], T1["akurasi"]
    tab_var = akurasi_per_variasi(hasil, T1["metode"], "pitch", tc.VARIASI)
    med_asli = np.nanmedian(np.abs(hasil.loc[hasil.variasi == "asli", "cent"]))
    med_fals = np.nanmedian(np.abs(hasil.loc[hasil.variasi == "fals", "cent"]))
    _, _, p_dtw = tc.mcnemar(hasil.pitch.to_numpy(), hasil["semiton | DTW"].to_numpy(), hasil["semiton | Euclid resampling"].to_numpy())

    st.markdown(kpi_html([
        ("Semiton + DTW", pct(acc.loc["semiton | DTW", "akurasi"]), ci_teks(acc, "semiton | DTW")),
        ("Semiton + Euclid resampling", pct(acc.loc["semiton | Euclid resampling", "akurasi"]), ci_teks(acc, "semiton | Euclid resampling")),
        ("YIN (baseline)", pct(acc.loc["YIN (baseline tuner)", "akurasi"]), ci_teks(acc, "YIN (baseline tuner)")),
        ("McNemar DTW vs Euclid", f"p = {p_dtw:.3f}", "berbeda nyata" if p_dtw < 0.05 else "tidak berbeda nyata"),
        ("Median |cent| asli / fals", f"{med_asli:.1f} / {med_fals:.1f}", "target 0 / 35")]), unsafe_allow_html=True)

    tabs = st.tabs(["Akurasi per variasi", "Akurasi & CI 95%", "Matriks konfusi", "Per senar", "Uji McNemar",
                    "DTW & Sakoe–Chiba", "Contoh kesalahan"])
    with tabs[0]:
        tampil(fig_bar_variasi(tab_var, tc.VARIASI, "Akurasi 1-NN per variasi (garis titik = tebakan acak 1/6)", 1 / 6))
        st.dataframe(fmt_tabel(tab_var, persen=tab_var.columns), use_container_width=True)
        st.info("Semiton paling stabil di semua variasi. Chroma sedikit lebih rendah karena tidak membedakan oktaf (E2 ↔ E4). "
                "MFCC mendekati tebakan acak karena merepresentasikan warna suara, bukan tinggi nada.")
    with tabs[1]:
        tampil(fig_ci(acc))
        st.dataframe(fmt_tabel(acc, persen=["akurasi", "CI 95% bawah", "CI 95% atas"], desimal=["F1 makro"]),
                     use_container_width=True)
    with tabs[2]:
        m = st.selectbox("Metode", T1["metode"], index=T1["metode"].index("semiton | DTW"))
        c1, c2 = st.columns([1, 1])
        with c1:
            tampil(fig_konfusi(hasil.label, hasil[m].map(tc.SENAR), tc.LABEL, m))
        with c2:
            e2e4 = int(((hasil.label == "E2") & (hasil[m] == 64)).sum() + ((hasil.label == "E4") & (hasil[m] == 40)).sum())
            st.metric("E2 ↔ E4 tertukar", e2e4)
            st.caption("Pasangan E2 dan E4 bernama nada sama (E) dan berjarak dua oktaf. Fitur chroma tidak dapat membedakannya.")
    with tabs[3]:
        m = st.selectbox("Metode ", T1["metode"], index=T1["metode"].index("semiton | DTW"))
        st.dataframe(fmt_tabel(T1["per_kelas"][m], desimal=["presisi", "recall", "F1"]), use_container_width=True)
        st.dataframe(T1["jumlah_uji"], use_container_width=True)
        st.caption("Tabel kedua: jumlah rekaman uji per senar dan jenis gitar (sebelum diberi 4 variasi).")
    with tabs[4]:
        st.markdown("Uji McNemar eksak: hanya menghitung bunyi yang ditebak benar oleh satu metode tetapi salah oleh metode lain.")
        st.dataframe(fmt_tabel(T1["mcnemar"], p=["p-value"]), use_container_width=True)
        st.markdown("**Hanya variasi `pendek`** (panjang uji jauh berbeda dari template):")
        st.dataframe(fmt_tabel(T1["mcnemar_pendek"], p=["p-value"]), use_container_width=True)
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            T = A["F_templ"]["semiton"][tc.PITCH.index(45)]; F = T1["contoh_pendek"]["F"]
            jarak, _, jalur = tc.dtw(T, F)
            tampil(fig_warping(T, F, jalur, f"Template A2 vs uji A2 (pendek 0,8 detik)\nDTW ternormalisasi = {jarak / (len(T) + len(F)):.3f}"))
        with c2:
            st.markdown("**Jendela Sakoe–Chiba** (fitur semiton)")
            st.dataframe(fmt_tabel(T1["sakoe"], persen=["akurasi", "akurasi 'pendek'"], desimal=["waktu (detik)"]),
                         use_container_width=True)
            st.caption("Jendela sempit mempercepat komputasi. Jika panjang dua deret sangat berbeda, jendela yang terlalu sempit "
                       "memaksa jalur mengikuti diagonal sehingga DTW kembali mirip Euclid resampling.")
    with tabs[6]:
        m = st.selectbox("Metode  ", T1["metode"], index=T1["metode"].index("semiton | DTW"))
        salah = hasil[hasil[m] != hasil.pitch][["instrumen", "sumber", "velocity", "variasi", "label", m]].copy()
        salah[m] = salah[m].map(tc.SENAR)
        st.write(f"{len(salah)} dari {len(hasil)} bunyi salah ditebak.")
        st.dataframe(salah.rename(columns={m: "prediksi", "label": "sebenarnya"}), use_container_width=True)
        if len(salah):
            st.caption("Bila kesalahan menumpuk di satu gitar dan muncul di semua variasi, penyebabnya adalah rekaman/gitar itu "
                       "sendiri (warna suara jauh dari template), bukan variasinya.")


# =============================================================================== 3. hasil chord
elif halaman.startswith("3"):
    judul("Hasil analisis cek chord", "Bagian 2 · 8 chord terbuka · chord strumming disusun dari nada gitar NSynth")
    hasil2, acc2, verif = T2["hasil"], T2["akurasi"], T2["verif"]
    tab_var2 = akurasi_per_variasi(hasil2, T2["metode"], "chord", tc.VARIASI2)
    hilang = [c for c in tc.NAMA_CHORD if c not in set(hasil2.chord)]
    _, _, p_dtw = tc.mcnemar(hasil2.chord.to_numpy(), hasil2["chroma | DTW"].to_numpy(), hasil2["chroma | Euclid resampling"].to_numpy())
    terbaik_ver = verif["ROC-AUC"].idxmax()

    st.markdown(kpi_html([
        ("Chroma + DTW", pct(acc2.loc["chroma | DTW", "akurasi"]), ci_teks(acc2, "chroma | DTW")),
        ("Chroma + Euclid resampling", pct(acc2.loc["chroma | Euclid resampling", "akurasi"]), ci_teks(acc2, "chroma | Euclid resampling")),
        ("Template biner (baseline)", pct(acc2.loc["template biner (baseline)", "akurasi"]), ci_teks(acc2, "template biner (baseline)")),
        ("McNemar DTW vs Euclid", f"p = {p_dtw:.3f}", "berbeda nyata" if p_dtw < 0.05 else "tidak berbeda nyata"),
        ("Verifikasi: AUC terbaik", f"{verif.loc[terbaik_ver, 'ROC-AUC']:.3f}", terbaik_ver)]), unsafe_allow_html=True)
    if hilang:
        st.warning(f"Chord {', '.join(hilang)} tidak ada di data uji: tidak ada gitar uji yang memiliki semua nadanya di NSynth. "
                   "Akurasi dan F1 makro dihitung dari chord yang tersedia saja.")

    tabs = st.tabs(["Akurasi per variasi", "Akurasi & CI 95%", "Matriks konfusi", "Per chord", "Uji McNemar",
                    "Jalur warping & Sakoe–Chiba", "Verifikasi benar/salah"])
    with tabs[0]:
        tampil(fig_bar_variasi(tab_var2, tc.VARIASI2, "Akurasi klasifikasi chord per variasi (garis titik = tebakan acak 1/8)",
                               1 / 8, baseline="template biner (baseline)"))
        st.dataframe(fmt_tabel(tab_var2, persen=tab_var2.columns), use_container_width=True)
    with tabs[1]:
        tampil(fig_ci(acc2))
        st.dataframe(fmt_tabel(acc2, persen=["akurasi", "CI 95% bawah", "CI 95% atas"], desimal=["F1 makro"]),
                     use_container_width=True)
    with tabs[2]:
        m = st.selectbox("Metode", T2["metode"], index=T2["metode"].index("chroma | DTW"))
        c1, c2 = st.columns([1, 1])
        with c1:
            tampil(fig_konfusi(hasil2.chord, hasil2[m], tc.NAMA_CHORD, m))
        with c2:
            for a, b in [("A", "Am"), ("E", "Em"), ("D", "Dm")]:
                n = int(((hasil2.chord == a) & (hasil2[m] == b)).sum() + ((hasil2.chord == b) & (hasil2[m] == a)).sum())
                st.metric(f"{a} ↔ {b} tertukar", n)
            st.caption("Pasangan mayor/minor hanya berbeda satu nada (setengah nada), sehingga paling sulit dibedakan.")
    with tabs[3]:
        m = st.selectbox("Metode ", T2["metode"], index=T2["metode"].index("chroma | DTW"))
        st.dataframe(fmt_tabel(T2["per_kelas"][m], desimal=["presisi", "recall", "F1"]), use_container_width=True)
    with tabs[4]:
        st.dataframe(fmt_tabel(T2["mcnemar"], p=["p-value"]), use_container_width=True)
        st.markdown("**Hanya variasi `lambat`** (urutan dan waktu munculnya nada paling berbeda dari template):")
        st.dataframe(fmt_tabel(T2["mcnemar_lambat"], p=["p-value"]), use_container_width=True)
    with tabs[5]:
        c1, c2 = st.columns(2)
        cl = T2["contoh_lambat"]; T = A["F_templ2"]["chroma"][tc.NAMA_CHORD.index(cl["chord"])]; F = cl["F"]
        jarak, _, jalur = tc.dtw(T, F)
        with c1:
            tampil(fig_fitur(F, f"Chroma uji {cl['chord']} (strumming lambat)", "chroma"))
        with c2:
            tampil(fig_warping(T, F, jalur, f"Template {cl['chord']} vs uji {cl['chord']} lambat\n"
                                            f"DTW ternormalisasi = {jarak / (len(T) + len(F)):.3f}"))
        st.markdown("**Jendela Sakoe–Chiba** (fitur chroma)")
        st.dataframe(fmt_tabel(T2["sakoe"], persen=["akurasi", "akurasi 'lambat'"], desimal=["waktu (detik)"]),
                     use_container_width=True)
    with tabs[6]:
        st.markdown("""
Pertanyaan: *"Saya mau main chord X, sudah benar belum?"* Skor = jarak ke template chord tujuan; di atas ambang dinyatakan **salah**.
Ambang ditentukan lintas gitar dengan indeks Youden. Baris **aturan terdekat = target** adalah aturan yang dipakai aplikasi di halaman 5:
benar jika chord paling mirip sama dengan chord tujuan.
""")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            for (s, (fpr, tpr)), w in zip(T2["roc"].items(), [BIRU, JINGGA, HIJAU, MERAH]):
                ax.plot(fpr, tpr, color=w, lw=2, label=f"{s} (AUC {verif.loc[s, 'ROC-AUC']:.3f})")
            r = verif.loc["aturan terdekat = target"]
            ax.scatter([1 - r.spesifisitas], [r.sensitivitas], color="black", marker="*", s=120, zorder=5,
                       label="aturan terdekat = target")
            ax.plot([0, 1], [0, 1], ":", color="gray")
            ax.set_xlabel("1 - spesifisitas (alarm palsu)"); ax.set_ylabel("sensitivitas (kesalahan terdeteksi)")
            ax.set_title("Kurva ROC verifikasi chord", fontsize=10); ax.legend(fontsize=7)
            fig.tight_layout(); tampil(fig)
        with c2:
            sk = T2["skor_ver"]; kel = tc.VARIASI2 + ["nada salah", "senar mati"]
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.boxplot([sk.loc[sk.variasi == k_, "chroma | DTW"] for k_ in kel])
            ax.set_xticks(range(1, len(kel) + 1)); ax.set_xticklabels(kel, fontsize=8)
            ax.axvspan(len(tc.VARIASI2) + 0.5, len(kel) + 0.5, color=MERAH, alpha=0.07)
            ax.set_ylabel("jarak DTW ke template chord tujuan")
            ax.set_title("Sebaran skor chroma | DTW (area merah = chord salah)", fontsize=10)
            fig.tight_layout(); tampil(fig)
        st.dataframe(fmt_tabel(verif, persen=["sensitivitas", "CI sens. bawah", "CI sens. atas", "spesifisitas",
                                              "CI spes. bawah", "CI spes. atas", "akurasi verifikasi"], desimal=["ROC-AUC"]),
                     use_container_width=True)
        st.markdown("**Proporsi dinyatakan SALAH per kelompok bunyi** (chord benar: idealnya 0%; chord salah: idealnya 100%)")
        st.dataframe(fmt_tabel(T2["rinci"], persen=T2["rinci"].columns), use_container_width=True)
        st.info("Nada salah lebih mudah terdeteksi daripada senar mati: nada dari senar yang mati sering masih terdengar dari "
                "senar lain di oktaf berbeda, dan chroma tidak membedakan oktaf.")


# =============================================================================== input audio
def rekam_ulang(kunci):
    """Naikkan penghitung agar widget rekaman/unggah dibuat baru (kosong) dan siap merekam lagi."""
    st.session_state[f"{kunci}_n"] = st.session_state.get(f"{kunci}_n", 0) + 1


def input_audio(kunci):
    n = st.session_state.get(f"{kunci}_n", 0)
    sumber = None
    if hasattr(st, "audio_input"):
        sumber = st.audio_input("Tekan mikrofon untuk merekam, tekan lagi untuk berhenti", key=f"{kunci}_rekam_{n}")
    else:
        st.warning("Versi Streamlit belum mendukung perekaman (perlu ≥ 1.40). Gunakan unggah file.")
    with st.expander("atau unggah file WAV"):
        unggah = st.file_uploader("File WAV", type=["wav"], key=f"{kunci}_unggah_{n}")
    sumber = sumber or unggah
    if sumber is None:
        return None
    st.button("Rekam lagi", icon=":material/restart_alt:", key=f"{kunci}_ulang", on_click=rekam_ulang, args=(kunci,))
    try:
        x = tc.muat(sumber.getvalue())
    except Exception as e:
        st.error(f"File tidak bisa dibaca sebagai WAV ({e}). Ubah dulu ke WAV, misalnya dengan ffmpeg.")
        return None
    if len(x) < 0.4 * tc.SR:
        st.warning("Rekaman terlalu pendek. Biarkan bunyi berlangsung ±2 detik, lalu tekan **Rekam lagi**.")
        return None
    return x


# =============================================================================== 4. tuner live
if halaman.startswith("4"):
    judul("Coba tuner", "Rekam satu petikan senar, lalu lihat senar dan selisih nadanya")
    c_mode, c_info = st.columns([1, 2])
    with c_mode:
        mode = st.radio("Mode", ["Otomatis (DTW menebak senar)", "Manual (pilih senar)"])
        manual = st.selectbox("Senar", tc.LABEL, format_func=lambda s: f"Senar {tc.NOMOR_SENAR[s]}: {s}") \
            if mode.startswith("Manual") else None
    with c_info:
        st.markdown("""
**Cara pakai:** redam senar lain dengan telapak tangan, tekan mikrofon, petik **satu** senar, biarkan berbunyi ±2 detik, lalu hentikan.

**Cara kerja:** DTW (fitur semiton) mencocokkan bunyi dengan 6 template senar. Setelah senarnya diketahui, frekuensi dasar
diukur dengan menjumlahkan energi di harmoniknya (±1 semitone dari senar), lalu dihitung selisihnya dalam *cent*
(100 cent = 1 semitone; ±5 cent = pas).
""")
    x = input_audio("tuner")
    if x is not None:
        r = tc.tuner_live(x, A, manual)
        s = tc.SENAR[r["pitch"]]
        jenis = "ok" if r["status"] == "PAS" else "warn"
        st.session_state["terakhir"] = {"besar": s, "kecil": f"senar {tc.NOMOR_SENAR[s]}",
                                        "detail": f"{r['f0']:.1f} Hz · {r['cent']:+.1f} cent",
                                        "pill": pill("PAS" if jenis == "ok" else "BELUM PAS", jenis)}
        c1, c2 = st.columns([1.25, 1])
        with c1:
            with st.container(border=True):
                g, h = st.columns([1.1, 1])
                with g:
                    st.markdown(svg_img(svg_gauge(r["cent"]), 260, f"Jarum tuner {r['cent']:+.1f} cent"), unsafe_allow_html=True)
                with h:
                    st.markdown(f'<div class="gl-res"><span class="m">Senar {tc.NOMOR_SENAR[s]} · target '
                                f'{tc.midi_ke_hz(r["pitch"]):.1f} Hz</span><span class="gl-big">{s}</span>'
                                f'<span class="f">{r["f0"]:.1f} Hz · <strong>{r["cent"]:+.1f} cent</strong></span>'
                                f'{pill(r["status"].upper(), jenis)}</div>', unsafe_allow_html=True)
                if manual and tc.SENAR[r["pitch_dtw"]] != manual:
                    st.info(f"DTW menilai bunyi ini paling mirip senar **{tc.SENAR[r['pitch_dtw']]}**. Pastikan senar yang dipetik benar.")
                if abs(r["cent"]) > 90:
                    st.caption("Selisih mendekati batas pencarian; kemungkinan senar sangat fals atau senar lain ikut berbunyi.")
        with c2:
            with st.container(border=True):
                tampil(fig_jarak(r["jarak"], tc.LABEL, tc.SENAR[r["pitch_dtw"]], "Jarak DTW ke 6 template senar"))
        st.audio(np.clip(r["x"] / (np.abs(r["x"]).max() + 1e-9), -1, 1), sample_rate=tc.SR)
        with st.expander("Lihat di balik layar", expanded=False):
            tampil(fig_gelombang_spektrum(r["x"], tc.midi_ke_hz(r["pitch"]), r["f0"]))
            c1, c2 = st.columns(2)
            with c1:
                tampil(fig_fitur(r["F"], "Fitur semiton rekaman"))
            with c2:
                T = A["F_templ"]["semiton"][tc.PITCH.index(r["pitch"])]
                tampil(fig_warping(T, r["F"], r["jalur"], f"Jalur warping: template {s} vs rekaman", label_x="frame rekaman"))
            st.caption(f"Sebagai pembanding, YIN rentang lebar mendeteksi {r['yin_lebar']:.1f} Hz. Bila angka ini jauh dari "
                       "senar target (misalnya ±82 Hz pada senar lain), kemungkinan senar E rendah ikut berdengung.")


# =============================================================================== 5. cek chord live
if halaman.startswith("5"):
    judul("Coba cek chord", "Strumming sekali, lalu lihat chord yang terdeteksi atau apakah chord pilihanmu sudah benar")
    c1, c2 = st.columns([1, 2])
    with c1:
        mode_c = st.radio("Mode", ["Otomatis (DTW menebak chord)", "Manual (pilih chord tujuan)"], key="mode_chord")
        auto = mode_c.startswith("Otomatis")
        target = None
        if not auto:
            target = st.selectbox("Chord yang ingin dimainkan", tc.NAMA_CHORD, index=tc.NAMA_CHORD.index("G"))
            st.markdown(svg_img(svg_chord(target), 170, f"Diagram chord {target}"), unsafe_allow_html=True)
            st.markdown(f"**Posisi jari** (senar 6 → 1): `{tc.POSISI_JARI[target]}`")
            st.markdown("**Nada**: " + ", ".join(tc.nama_nada(p) for p in tc.CHORD[target]))
            st.markdown("**Kelas nada**: " + ", ".join(tc.NAMA_NADA[k] for k in sorted(tc.pc(tc.CHORD[target]))))
        with st.expander("Pengaturan diagnosis"):
            b_kurang = st.slider("Ambang 'nada kurang'", 0.1, 0.6, 0.3, 0.05)
            b_lebih = st.slider("Ambang 'nada berlebih'", 0.3, 0.9, 0.5, 0.05)
    with c2:
        if auto:
            st.markdown("""
**Cara pakai:** tekan mikrofon, bunyikan chord sekali (tidak perlu memilih chord lebih dulu), biarkan berbunyi ±2 detik, lalu hentikan.
Untuk mencoba lagi, tekan **Rekam lagi**.

**Cara kerja:** DTW (fitur chroma) mencocokkan strumming dengan 8 template chord dan menampilkan chord yang paling mirip.
Petunjuk *nada kurang/berlebih* membandingkan profil 12 nada rekaman dengan nada chord yang terdeteksi; ini heuristik,
jadi gunakan sebagai petunjuk, bukan vonis. Untuk memeriksa chord tertentu, pakai mode **Manual**.
""")
        else:
            st.markdown("""
**Cara pakai:** pilih chord, tekan mikrofon, bunyikan chord sekali, biarkan berbunyi ±2 detik, lalu hentikan.
Untuk mencoba lagi, tekan **Rekam lagi**.

**Cara kerja:** DTW (fitur chroma) mencocokkan strumming dengan 8 template chord. Dinyatakan **BENAR** bila chord paling mirip
sama dengan chord tujuan. Petunjuk *nada kurang/berlebih* membandingkan profil 12 nada rekaman dengan nada chord; ini heuristik,
jadi gunakan sebagai petunjuk, bukan vonis.
""")
    x = input_audio("chord")
    if x is not None:
        if auto:
            # tebak dulu chord terdekat, lalu diagnosis dihitung terhadap chord yang terdeteksi itu
            r = tc.cek_chord_live(x, tc.NAMA_CHORD[0], A, b_kurang, b_lebih)
            if r["terdekat"] != tc.NAMA_CHORD[0]:
                r = tc.cek_chord_live(x, r["terdekat"], A, b_kurang, b_lebih)
            target = r["terdekat"]
            jenis, label_pill = "ok", "TERDETEKSI"
        else:
            r = tc.cek_chord_live(x, target, A, b_kurang, b_lebih)
            jenis = "ok" if r["benar"] else "bad"
            label_pill = "BENAR" if r["benar"] else "SALAH"
        st.session_state["terakhir"] = {"besar": r["terdekat"], "kecil": "otomatis" if auto else f"target {target}",
                                        "detail": "cek chord", "pill": pill(label_pill, jenis)}
        c1, c2, c3 = st.columns([1, 1.3, 1.3])
        with c1:
            with st.container(border=True):
                st.markdown(f'<div class="gl-res"><span class="m">{"Chord terdeteksi" if auto else f"Target: {target} · paling mirip"}</span>'
                            f'<span class="gl-big">{r["terdekat"]}</span>{pill(label_pill, jenis)}</div>',
                            unsafe_allow_html=True)
                if auto:
                    st.markdown(svg_img(svg_chord(target), 150, f"Diagram chord {target}"), unsafe_allow_html=True)
                    st.caption(f"Posisi jari (senar 6 → 1): {tc.POSISI_JARI[target]}")
                elif not r["benar"]:
                    st.error(f"Bunyi lebih mirip chord {r['terdekat']}")
                if r["kurang"]:
                    st.warning("Cek nada kurang: " + ", ".join(r["kurang"]))
                if r["lebih"]:
                    st.warning("Cek nada berlebih: " + ", ".join(r["lebih"]))
                if not r["kurang"] and not r["lebih"]:
                    st.caption("Tidak ada nada yang mencurigakan.")
        with c2:
            with st.container(border=True):
                tampil(fig_jarak(r["jarak"], tc.NAMA_CHORD, r["terdekat"], "Jarak DTW ke 8 template chord"))
        with c3:
            with st.container(border=True):
                tampil(fig_profil(r["profil"], target))
        st.audio(np.clip(r["x"] / (np.abs(r["x"]).max() + 1e-9), -1, 1), sample_rate=tc.SR)
        with st.expander("Lihat di balik layar", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                tampil(fig_fitur(r["F"], "Fitur chroma rekaman", "chroma"))
            with c2:
                T = A["F_templ2"]["chroma"][tc.NAMA_CHORD.index(target)]
                tampil(fig_warping(T, r["F"], r["jalur"], f"Jalur warping: template {target} vs rekaman", label_x="frame rekaman"))
            tampil(fig_gelombang_spektrum(r["x"]))


# =============================================================================== kartu "Rekaman terakhir" di sidebar
t = st.session_state.get("terakhir")
if t:
    kartu_terakhir.markdown(f'<div class="gl-last"><span class="k">Rekaman terakhir</span><div><span class="b">{t["besar"]}</span> '
                            f'<span class="k">{t["kecil"]}</span></div><span class="d">{t["detail"]}</span>{t["pill"]}</div>',
                            unsafe_allow_html=True)
else:
    kartu_terakhir.markdown('<div class="gl-last"><span class="k">Rekaman terakhir</span>'
                            '<span class="d">Belum ada rekaman. Buka halaman Coba tuner atau Coba cek chord.</span></div>',
                            unsafe_allow_html=True)
