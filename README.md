# Dashboard DTW Guitar Tuner & Cek Chord

## Isi
- `app.py`: dashboard Streamlit (5 halaman)
- `tuner_core.py`: fungsi inti dari notebook (DTW, fitur, YIN, f0 harmonik, rakit chord, statistik, pipeline analisis)
- `requirements.txt`: library yang dibutuhkan
- `.streamlit/config.toml`: tema warna (desain A, Studio Terang)

## Cara menjalankan
1. Salin `app.py`, `tuner_core.py`, `requirements.txt`, dan folder `.streamlit/` ke folder project, sejajar dengan
   `nsynth-valid/` dan `nsynth-test/` (atau kedua file `.tar.gz`-nya).
2. Pasang library (sekali): `pip install -r requirements.txt`
3. Jalankan: `streamlit run app.py`
4. Browser terbuka otomatis (http://localhost:8501).

Saat pertama kali dibuka, dashboard menjalankan seluruh analisis (± 1 menit) lalu menyimpannya ke
`hasil_analisis.pkl`. Pembukaan berikutnya langsung memakai file itu. Tombol "Jalankan ulang analisis"
di sidebar menghapus file tersebut dan menghitung ulang.

## Rekam langsung
Halaman 4 dan 5 memakai `st.audio_input` (Streamlit >= 1.40). Izinkan akses mikrofon saat browser
meminta. Bila mikrofon tidak bisa dipakai, unggah file WAV.

## Deploy online (opsional)
Unggah `app.py`, `tuner_core.py`, `requirements.txt`, `.streamlit/config.toml`, dan `hasil_analisis.pkl` ke GitHub, lalu deploy di
Streamlit Community Cloud. Data NSynth tidak perlu ikut diunggah karena hasil analisis dan template
sudah tersimpan di `hasil_analisis.pkl`.
