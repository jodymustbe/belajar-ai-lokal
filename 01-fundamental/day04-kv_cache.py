"""Hari ke-4: Analisis Arsitektur Model & Pengukuran Memori KV Cache di Ollama.

Rumus Teoritis KV Cache:
    Bytes = 2 (Key & Value) * layers * kv_heads * head_dim * ctx * bytes_per_elem

Dependensi:
    pip install requests

Cara menjalankan:
    python 01-fundamental/day04_kv_cache.py
"""

import subprocess
import time
import requests

HOST = "http://localhost:11434"
MODEL = "qwen3.5:2b-q4_K_M"
CTX_LIST = [2048, 4096, 8192]
MIB = 1024 * 1024


# ==============================================================================
# 1. AMBIL METADATA ARSITEKTUR DARI /api/show (SETARA `ollama show`)
# ==============================================================================
def ambil_arsitektur_model(model: str) -> dict:
    res = requests.post(f"{HOST}/api/show", json={"model": model}, timeout=30)
    res.raise_for_status()
    data = res.json()
    info = data.get("model_info", {})

    arch = info.get("general.architecture", "unknown")

    def cari_nilai(akhiran: str):
        kunci_utama = f"{arch}.{akhiran}"
        if kunci_utama in info:
            return info[kunci_utama]
        for k, v in info.items():
            if k.endswith(f".{akhiran}"):
                return v
        return None

    layers = cari_nilai("block_count")
    emb_dim = cari_nilai("embedding_length")
    q_heads = cari_nilai("attention.head_count")
    kv_heads = cari_nilai("attention.head_count_kv")
    head_dim = cari_nilai("attention.key_length")

    # Fallback jika key_length tidak ditulis eksplisit di metadata GGUF
    if head_dim is None and emb_dim and q_heads:
        head_dim = emb_dim // q_heads

    return {
        "architecture": arch,
        "layers": layers,
        "embedding_length": emb_dim,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }


def hitung_kv_cache_bytes(
    layers: int, kv_heads: int | list[int], head_dim: int, ctx: int, bytes_per_elem: int
) -> tuple[int, int, int]:
    """Menghitung ukuran KV cache teoritis dalam byte.

    Mendukung arsitektur standar (kv_heads berupa integer) maupun arsitektur hibrida
    (kv_heads berupa list per layer).
    """
    if isinstance(kv_heads, list):
        active_layers = sum(1 for h in kv_heads if h > 0)
        total_kv_heads = sum(kv_heads)
    else:
        active_layers = layers
        total_kv_heads = layers * kv_heads

    total_bytes = 2 * total_kv_heads * head_dim * ctx * bytes_per_elem
    return total_bytes, active_layers, total_kv_heads


# ==============================================================================
# 2. PENGUKURAN VRAM NYATA PADA BERBAGAI UKURAN num_ctx
# ==============================================================================
def baca_nvidia_smi_mb() -> int | None:
    """Membaca total VRAM terpakai di GPU via nvidia-smi (opsional, Windows/Linux)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def unload_model(model: str) -> None:
    """Mengosongkan model dari memori agar pengukuran tiap num_ctx dimulai bersih."""
    try:
        requests.post(
            f"{HOST}/api/chat",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        time.sleep(1.0)
    except requests.RequestException:
        pass


def ukur_vram_per_ctx(model: str, ctx: int) -> dict:
    unload_model(model)

    # Muat model dengan num_ctx target (num_predict=1 agar cepat selesai)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Halo"}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": ctx, "num_predict": 1},
    }
    res = requests.post(f"{HOST}/api/chat", json=payload, timeout=180)
    res.raise_for_status()

    # Baca alokasi memori aktif dari /api/ps (setara `ollama ps`)
    ps_res = requests.get(f"{HOST}/api/ps", timeout=30)
    ps_res.raise_for_status()
    models_running = ps_res.json().get("models", [])

    target = next((m for m in models_running if m.get("name") == model), None)
    if not target:
        raise RuntimeError(f"Model {model} tidak ditemukan di /api/ps setelah dimuat.")

    size_total = target.get("size", 0)
    size_vram = target.get("size_vram", 0)
    gpu_pct = (size_vram / size_total * 100.0) if size_total > 0 else 0.0

    return {
        "ctx": ctx,
        "active_ctx": target.get("context_length", ctx),
        "total_mib": size_total / MIB,
        "vram_mib": size_vram / MIB,
        "gpu_pct": gpu_pct,
        "nvidia_smi_mb": baca_nvidia_smi_mb(),
    }


def main() -> None:
    print("=" * 76)
    print(f"HARI KE-4: ANALISIS KV CACHE & UKURAN KONTEKS ({MODEL})")
    print("=" * 76)

    # Langkah 1: Ambil parameter arsitektur
    arch = ambil_arsitektur_model(MODEL)
    print("\n[1] Parameter Arsitektur dari /api/show:")
    for k, v in arch.items():
        print(f"    - {k:<18}: {v}")

    if not all([arch["layers"], arch["kv_heads"], arch["head_dim"]]):
        print("\n[PERINGATAN] Metadata layer/kv_heads/head_dim tidak lengkap di GGUF.")
        return

    # Langkah 2: Hitung KV Cache teoritis (f16 = 2 byte vs q8_0 = 1 byte)
    print("\n[2] Perhitungan Teoritis KV Cache (2 * layers * kv_heads * head_dim * ctx * bytes):")
    print(f"    {'num_ctx':>8} | {'KV Cache f16 (2B)':>18} | {'KV Cache q8_0 (1B)':>19}")
    print("    " + "-" * 52)
    teori_f16 = {}
    for ctx in CTX_LIST:
        b_f16, _, _ = hitung_kv_cache_bytes(
            arch["layers"], arch["kv_heads"], arch["head_dim"], ctx, bytes_per_elem=2
        )
        b_q8, _, _ = hitung_kv_cache_bytes(
            arch["layers"], arch["kv_heads"], arch["head_dim"], ctx, bytes_per_elem=1
        )
        teori_f16[ctx] = b_f16 / MIB
        print(f"    {ctx:>8} | {b_f16 / MIB:>14.2f} MiB | {b_q8 / MIB:>15.2f} MiB")

    # Langkah 3: Ukur pemakaian memori aktual di Ollama (/api/ps)
    print("\n[3] Pengukuran Aktual di Ollama (/api/ps):")
    hasil = []
    for ctx in CTX_LIST:
        print(f"    Menguji num_ctx = {ctx} ...")
        hasil.append(ukur_vram_per_ctx(MODEL, ctx))

    print("\n    Ringkasan Hasil Aktual:")
    header = (
        f"    {'num_ctx':>8} | {'Total Mem':>11} | {'VRAM Ollama':>12} | "
        f"{'% GPU':>7} | {'Delta Total':>12} | {'GPU Total (SMI)':>15}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))

    base_mem = hasil[0]["total_mib"]
    for row in hasil:
        delta = row["total_mib"] - base_mem
        smi_str = f"{row['nvidia_smi_mb']} MB" if row["nvidia_smi_mb"] is not None else "N/A"
        print(
            f"    {row['ctx']:>8} | {row['total_mib']:>8.1f} MiB | "
            f"{row['vram_mib']:>9.1f} MiB | {row['gpu_pct']:>6.1f}% | "
            f"{delta:>+9.1f} MiB | {smi_str:>15}"
        )

    # Langkah 4: Catatan analisis otomatis
    print("\n[4] Hal yang Perlu Diamati untuk Logbook Hari ke-4:")
    print(
        "    - Perhatikan kenaikan 'Total Mem' dari 2048 -> 4096 -> 8192 dan bandingkan\n"
        "      dengan selisih angka teoritis f16 vs q8_0 pada tabel [2] di atas."
    )
    if any(r["gpu_pct"] < 100.0 for r in hasil):
        print(
            "    - TERDETEKSI SPILL KE CPU/RAM (% GPU < 100%) pada konteks besar!\n"
            "      Ini membuktikan batas aman VRAM 4 GB tidak sanggup menampung bobot +\n"
            "      KV cache penuh pada num_ctx tersebut."
        )
    else:
        print(
            "    - Seluruh pengujian bertahan di 100% GPU. Catat margin sisa VRAM sebelum\n"
            "      menyentuh batas ~3.200 MiB."
        )


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as err:
        print(f"\n[ERROR] Gagal terhubung ke Ollama di {HOST}: {err}")
        print("Pastikan aplikasi Ollama aktif sebelum menjalankan skrip ini.")