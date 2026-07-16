# TO BE ADDED AS A CELL UNDER rf-classification.ipynb

# ── Export sample spectrograms (RF: 16-QAM / QPSK / BPSK, Audio: mambo / bebop / bg) ──
# NOTE: "Q-Sprk" -> QPSK (S2 profile), "B-Sprk" -> BPSK (S1 profile), "16QAM" -> 16-QAM (S3 profile)
import os, random
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

OUTPUT_DIR = "./exported_spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- RF: build a "pure" (single-modulation) waveform using the same hop logic
#         generate_mixed_chunk() uses, just without mixing profiles ---
def generate_pure_rf_waveform(profile_name, duration=None):
    profile   = sys_state.profiles[profile_name]
    freq_list = profile["freqs"]
    mod_type  = profile["mod"]
    fs        = sys_state.fs
    total_dur = duration if duration is not None else sys_state.chunk_duration
    hop_dur   = univ_hop_dur
    num_syms  = univ_sph

    full_wave, current_t, hop_idx = [], 0.0, 0
    while current_t < total_dur:
        this_hop_dur = min(hop_dur, total_dur - current_t)
        if this_hop_dur <= 0:
            break
        f_c = freq_list[hop_idx % len(freq_list)]
        hop_idx += 1
        tt = np.linspace(0, this_hop_dur, int(fs * this_hop_dur), endpoint=False)
        if len(tt) == 0:
            break

        if mod_type == "16-QAM":
            scale = 1.0 / np.sqrt(10)
            levs = [-3, -1, 1, 3]
            syms = [complex(x, y) * scale for x, y in zip(
                np.random.choice(levs, num_syms), np.random.choice(levs, num_syms))]
        elif mod_type == "QPSK":
            scale = 1.0 / np.sqrt(2)
            syms = [(np.random.choice([-1, 1]) + 1j * np.random.choice([-1, 1])) * scale
                    for _ in range(num_syms)]
        else:  # BPSK
            syms = [(2 * np.random.randint(0, 2) - 1) + 0j for _ in range(num_syms)]

        s_up = np.repeat(syms, int(len(tt) / num_syms) + 1)[:len(tt)]
        wave = s_up.real * np.cos(2 * np.pi * f_c * tt) - s_up.imag * np.sin(2 * np.pi * f_c * tt)
        full_wave.append(wave)
        current_t += this_hop_dur

    raw = np.concatenate(full_wave)
    raw = raw[:int(fs * total_dur)]
    return raw + np.random.randn(len(raw)) * 0.1  # match noise_level=0.1 used elsewhere


def save_rf_spectrogram(label, profile_name, out_path, duration=4.0, dpi=300):
    wave = generate_pure_rf_waveform(profile_name, duration=duration)
    fs = sys_state.fs

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 5), facecolor='#0d1117')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d1117')

    nfft = min(512, max(64, len(wave) // 8))
    noverlap = nfft // 2
    _, _, t_bins, im = ax.specgram(
        wave, NFFT=nfft, Fs=fs, noverlap=noverlap,
        cmap='inferno', vmin=-80, vmax=0,
    )
    ax.set_ylim(0, min(2000, fs / 2))
    ax.set_xlim(0, max(float(t_bins[-1]), 0.01))

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('dB', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white', labelsize=9)

    ax.set_title(f"RF Spectrogram — {label}", color='white', fontsize=14, pad=10)
    ax.set_xlabel("Time (s)", color='white', fontsize=12)
    ax.set_ylabel("Frequency (Hz)", color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2f3a')

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, format='png', facecolor='#0d1117', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


def save_audio_spectrogram(label, file_dir, out_path, sr=22050, duration=1.0, dpi=300):
    files = [f for f in os.listdir(file_dir) if f.endswith('.wav')]
    fpath = os.path.join(file_dir, random.choice(files))
    y, _ = librosa.load(fpath, sr=sr, duration=duration)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                    ax=ax, cmap='inferno', fmax=sr // 2)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('dB', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white', labelsize=9)

    ax.set_title(f"Audio Spectrogram — {label}", color='white', fontsize=14, pad=10)
    ax.set_xlabel("Time (s)", color='white', fontsize=12)
    ax.set_ylabel("Frequency (Hz)", color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2f3a')

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, format='png', facecolor='#0d1117', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


# --- RF exports ---
rf_targets = [
    ("16QAM", "S3 (THREAT)"),
    ("QPSK",  "S2 (Standard)"),
    ("BPSK",  "S1 (Long Range)"),
]
for label, profile_name in rf_targets:
    out_path = os.path.join(OUTPUT_DIR, f"rf_spectrogram_{label}.png")
    save_rf_spectrogram(label, profile_name, out_path)

# --- Audio exports ---
audio_targets = [
    ("mambo", AUDIO_DATA_PATHS["mambo"]),
    ("bebop", AUDIO_DATA_PATHS["bebop"]),
    ("bg",    AUDIO_DATA_PATHS["background"]),
]
for label, folder in audio_targets:
    out_path = os.path.join(OUTPUT_DIR, f"audio_spectrogram_{label}.png")
    save_audio_spectrogram(label, folder, out_path)

print("\nAll 6 spectrograms saved to:", os.path.abspath(OUTPUT_DIR))
