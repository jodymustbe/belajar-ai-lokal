"""Hari ke-3: Memahami Inference, Prefill vs Decode, dan Sampling pada LLM.

Dependensi:
    pip install numpy requests

Cara menjalankan:
    python 01-fundamental/day03_inference_sampling.py
"""

import numpy as np
import requests

HOST = "http://localhost:11434"
MODEL = "qwen3.5:2b-q4_K_M"


# ==============================================================================
# BAGIAN 1: SIMULASI SAMPLER MANUAL DENGAN NUMPY (LOGITS -> TOKEN TERPILIH)
# ==============================================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    """Mengubah skor mentah (logits) menjadi distribusi probabilitas (total = 1.0)."""
    shifted = logits - np.max(logits)  # stabilisasi numerik agar tidak overflow
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def jalankan_simulasi_sampler(
    konteks: str,
    kandidat_token: list[str],
    logits_mentah: list[float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42,
) -> str:
    rng = np.random.default_rng(seed)
    logits = np.array(logits_mentah, dtype=float)

    print(f"\n--- Konfigurasi: temperature={temperature}, top_k={top_k}, top_p={top_p}, seed={seed} ---")
    print(f"[1] Konteks Input  : {konteks!r}")
    print(f"[2] Logits Mentah  : {dict(zip(kandidat_token, logits))}")

    # Langkah A: Temperature Scaling
    if temperature <= 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        print("[3] Temperature=0  : Mode Greedy (otomatis pilih skor tertinggi 100%).")
    else:
        scaled_logits = logits / temperature
        probs = softmax(scaled_logits)
        prob_str = {tok: f"{p*100:.1f}%" for tok, p in zip(kandidat_token, probs)}
        print(f"[3] Setelah Softmax (T={temperature}): {prob_str}")

        # Langkah B: Top-K Filtering (ambil K kandidat teratas saja)
        if 0 < top_k < len(probs):
            top_k_idx = np.argsort(probs)[::-1][:top_k]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_k_idx] = True
            probs = np.where(mask, probs, 0.0)
            probs /= np.sum(probs)  # normalisasi ulang agar total kembali 100%
            prob_str = {tok: f"{p*100:.1f}%" for tok, p in zip(kandidat_token, probs)}
            print(f"[4] Setelah Top-K (k={top_k})  : {prob_str}")

        # Langkah C: Top-P / Nucleus Filtering (ambil kandidat hingga akumulasi >= top_p)
        if 0.0 < top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cum_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cum_probs, top_p)
            keep_idx = sorted_idx[: cutoff_idx + 1]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep_idx] = True
            probs = np.where(mask, probs, 0.0)
            probs /= np.sum(probs)
            prob_str = {tok: f"{p*100:.1f}%" for tok, p in zip(kandidat_token, probs)}
            print(f"[5] Setelah Top-P (p={top_p}): {prob_str}")

    # Langkah D: Sampling (melempar dadu berbobot) & Autoregressive Update
    idx_terpilih = int(rng.choice(len(kandidat_token), p=probs))
    token_terpilih = kandidat_token[idx_terpilih]
    konteks_baru = konteks + token_terpilih

    print(f"[6] Token Terpilih : {token_terpilih!r}")
    print(f"[7] Loop Autoregresif -> Input berikutnya menjadi: {konteks_baru!r}")
    return token_terpilih


def demo_bagian_1() -> None:
    print("=" * 72)
    print("BAGIAN 1: SIMULASI CARA KERJA SAMPLER LLM (NUMPY)")
    print("=" * 72)
    print(
        "Penjelasan: Neural network tidak langsung mengeluarkan kata, melainkan angka skor\n"
        "(logits) untuk setiap token di kamus. Sampler bertugas mengubah skor tersebut\n"
        "menjadi probabilitas dan memilih 1 token berikutnya."
    )

    konteks = "Status kelembapan tanah saat ini adalah"
    kandidat = [" normal", " kering", " basah", " 45%", " error"]
    logits_palsu = [3.2, 2.4, 2.0, 1.1, -0.5]

    # Skenario 1: Temperature rendah (fokus/yakin pada jawaban teratas)
    jalankan_simulasi_sampler(konteks, kandidat, logits_palsu, temperature=0.2, seed=42)

    # Skenario 2: Temperature normal (probabilitas menyebar sesuai skor asli)
    jalankan_simulasi_sampler(konteks, kandidat, logits_palsu, temperature=1.0, seed=42)

    # Skenario 3: Temperature tinggi + filter Top-K & Top-P (kreatif tapi tetap dibatasi)
    jalankan_simulasi_sampler(
        konteks, kandidat, logits_palsu, temperature=1.2, top_k=3, top_p=0.85, seed=42
    )


# ==============================================================================
# BAGIAN 2: EKSPERIMEN INFERENCE NYATA PADA QWEN3.5:2B-Q4_K_M DI OLLAMA
# ==============================================================================
def panggil_ollama(prompt: str, options: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": 4096, **options},
    }
    res = requests.post(f"{HOST}/api/chat", json=payload, timeout=120)
    res.raise_for_status()
    return res.json()


def demo_bagian_2() -> None:
    print("\n" + "=" * 72)
    print(f"BAGIAN 2: EKSPERIMEN NYATA PADA {MODEL} (OLLAMA)")
    print("=" * 72)

    # 2A. Membedakan Prefill vs Decode
    print("\n[A] Mengukur Fase PREFILL vs DECODE:")
    print(
        "    - Prefill: Model membaca seluruh token prompt awal sekaligus.\n"
        "    - Decode : Model menghasilkan token jawaban satu per satu secara berulang."
    )
    prompt_panjang = (
        "Data sensor lahan A menunjukkan suhu udara 31 derajat Celcius, "
        "kelembapan udara 68%, dan kadar air tanah 22%. "
        "Jelaskan dalam 2 kalimat singkat apakah tanaman cabai perlu segera disiram."
    )
    r = panggil_ollama(prompt_panjang, {"temperature": 0, "seed": 1, "num_predict": 80})

    p_tokens = r.get("prompt_eval_count", 0)
    p_sec = r.get("prompt_eval_duration", 1) / 1e9
    d_tokens = r.get("eval_count", 0)
    d_sec = r.get("eval_duration", 1) / 1e9

    print(f"    -> Prefill : {p_tokens} token diproses dalam {p_sec:.3f} detik ({p_tokens/p_sec:.1f} tok/s)")
    print(f"    -> Decode  : {d_tokens} token ditulis dalam {d_sec:.3f} detik ({d_tokens/d_sec:.1f} tok/s)")
    print(f"    -> Jawaban : {r['message']['content'].strip()}")

    # 2B. Membandingkan Deterministik vs Kreatif (3x percobaan per setelan)
    prompt_uji = "Buat 1 kalimat slogan singkat (maksimal 8 kata) untuk sistem irigasi otomatis."

    print("\n[B] Uji Konsistensi: Deterministik (temperature=0, seed=42) - 3x Run:")
    for i in range(1, 4):
        out = panggil_ollama(prompt_uji, {"temperature": 0, "seed": 42, "num_predict": 40})
        print(f"    Run {i}: {out['message']['content'].strip()}")

    print("\n[C] Uji Variasi: Kreatif/Acak (temperature=1.2, top_k=40, top_p=0.9) - 3x Run:")
    for i in range(1, 4):
        out = panggil_ollama(
            prompt_uji, {"temperature": 1.2, "top_k": 40, "top_p": 0.9, "num_predict": 40}
        )
        print(f"    Run {i}: {out['message']['content'].strip()}")


if __name__ == "__main__":
    demo_bagian_1()
    try:
        demo_bagian_2()
    except requests.RequestException as err:
        print(f"\n[ERROR] Gagal terhubung ke Ollama di {HOST}: {err}")
        print("Pastikan aplikasi Ollama aktif dan model qwen3.5:2b-q4_K_M sudah terpasang.")