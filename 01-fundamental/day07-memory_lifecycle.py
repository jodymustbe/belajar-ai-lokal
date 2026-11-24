"""Uji Manajemen Memori Ollama: Rasio GPU/CPU, Batas Aman RAM, & Siklus `keep_alive`.

Fitur pengujian:
1. Memverifikasi lingkungan `OLLAMA_MAX_LOADED_MODELS=1` dan `OLLAMA_NUM_PARALLEL=1`.
2. Menguji matriks model (`qwen3.5:2b-q4_K_M`, `phi4-mini:latest`, `qwen3:4b`, `qwen3.5:4b`)
   pada `num_ctx` = `[2048, 4096, 8192]` dengan 1 prompt per kombinasi.
3. Mencatat rasio GPU/CPU (`/api/ps`) serta `psutil.virtual_memory().available` pada
   3 titik: SEBELUM muat, SAAT termuat, dan SESUDAH unload.
4. Menandai status BATAS AMAN: 100% GPU offload AND available RAM >= ambang batas.
5. Menguji parameter `keep_alive` (`"0"`, `"5m"`, `"-1"`) untuk memantau apakah RAM
   kembali pulih setelah model di-unload.

Dependensi:
    pip install psutil requests

Cara menjalankan:
    py 01-fundamental/day07_memory_lifecycle.py
    # Opsional: ubah ambang aman RAM tersedia (default: 1536 MiB = 1.5 GiB)
    py 01-fundamental/day07_memory_lifecycle.py --min-ram-mib 1536
"""

import argparse
import os
import time
import psutil
import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODELS = [
    "qwen3.5:2b-q4_K_M",
    "phi4-mini:latest",
    "qwen3:4b",
    "qwen3.5:4b",
]
CTX_LIST = [2048, 4096, 8192]
KEEP_ALIVE_MODES = [0, "5m", -1]

MIB = 1024 * 1024
PROMPT_UJI = "Sebutkan 3 fungsi utama sistem manajemen memori pada runtime LLM lokal dalam 2 kalimat."


# ==============================================================================
# 1. UTILITAS MEMORI & STATUS OLLAMA (/api/ps)
# ==============================================================================
def ram_available_mib() -> float:
    """Mengembalikan sisa RAM sistem yang tersedia dalam satuan MiB."""
    return psutil.virtual_memory().available / MIB


def cek_env_server() -> None:
    """Memeriksa apakah env var sudah terbaca di sesi shell saat ini."""
    print("=" * 94)
    print("LANGKAH 1: VERIFIKASI KONFIGURASI LINGKUNGAN OLLAMA")
    print("=" * 94)
    for var_name, expected in [("OLLAMA_MAX_LOADED_MODELS", "1"), ("OLLAMA_NUM_PARALLEL", "1")]:
        val = os.environ.get(var_name)
        status = "OK" if val == expected else f"BELUM TERBACA DI SHELL INI (nilai={val!r})"
        print(f"  - {var_name:<26} = {expected} -> {status}")
    print(
        "  Catatan: Pastikan variabel di atas sudah di-set via `setx` dan server Ollama\n"
        "           sudah di-restart sebelum skrip ini dijalankan."
    )


def ambil_status_ps(model: str) -> dict:
    """Membaca rasio GPU/CPU dan ukuran memori model aktif dari endpoint /api/ps."""
    res = requests.get(f"{HOST}/api/ps", timeout=15)
    res.raise_for_status()
    for m in res.json().get("models", []):
        if m.get("name") == model:
            total_bytes = m.get("size", 0)
            vram_bytes = m.get("size_vram", 0)
            gpu_pct = (vram_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
            cpu_pct = max(0.0, 100.0 - gpu_pct)
            return {
                "loaded": True,
                "total_mib": total_bytes / MIB,
                "vram_mib": vram_bytes / MIB,
                "gpu_pct": gpu_pct,
                "cpu_pct": cpu_pct,
                "expires_at": m.get("expires_at", "-"),
            }
    return {
        "loaded": False,
        "total_mib": 0.0,
        "vram_mib": 0.0,
        "gpu_pct": 0.0,
        "cpu_pct": 0.0,
        "expires_at": "-",
    }


def paksa_unload_semua(jeda_detik: float = 2.0) -> None:
    """Mengeluarkan semua model yang sedang dimuat di Ollama agar baseline RAM bersih."""
    try:
        res = requests.get(f"{HOST}/api/ps", timeout=15)
        if res.status_code == 200:
            for m in res.json().get("models", []):
                requests.post(
                    f"{HOST}/api/chat",
                    json={"model": m["name"], "messages": [], "keep_alive": "0"},
                    timeout=30,
                )
        time.sleep(jeda_detik)
    except requests.RequestException:
        pass


def kirim_prompt(model: str, num_ctx: int, keep_alive: int) -> dict:
    """Mengirim satu prompt tetap ke /api/chat dengan konfigurasi konteks & keep_alive."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_UJI}],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": 32,
            "temperature": 0,
            "seed": 42,
        },
    }
    if "qwen3" in model.lower():
        payload["think"] = False

    t0 = time.perf_counter()
    res = requests.post(f"{HOST}/api/chat", json=payload, timeout=300)
    res.raise_for_status()
    elapsed = time.perf_counter() - t0
    data = res.json()

    d_tok = data.get("eval_count", 0)
    d_sec = data.get("eval_duration", 1) / 1e9
    return {
        "decode_tps": d_tok / d_sec if d_sec > 0 else 0.0,
        "elapsed_sec": elapsed,
    }


# ==============================================================================
# 2. MATRIKS UJI MODEL x NUM_CTX & PENANDAAN BATAS AMAN (LANGKAH 2-4)
# ==============================================================================
def evaluasi_batas_aman(gpu_pct: float, ram_saat_mib: float, ambang_mib: float) -> str:
    """Menentukan apakah kombinasi model + num_ctx masuk kategori BATAS AMAN."""
    full_gpu = gpu_pct >= 99.9
    ram_aman = ram_saat_mib >= ambang_mib
    if full_gpu and ram_aman:
        return "AMAN (100% GPU)"

    alasan = []
    if not full_gpu:
        alasan.append(f"Spill CPU {100.0 - gpu_pct:.1f}%")
    if not ram_aman:
        alasan.append(f"RAM < {ambang_mib:.0f} MiB")
    return "TIDAK AMAN (" + ", ".join(alasan) + ")"


def jalankan_matriks_konteks(installed: set[str], ambang_ram_mib: float) -> list[dict]:
    print("\n" + "=" * 94)
    print(f"LANGKAH 2-4: MATRIKS MODEL x NUM_CTX (Ambang Aman RAM Tersedia >= {ambang_ram_mib:.0f} MiB)")
    print("=" * 94)

    semua_hasil = []

    for model in MODELS:
        if model not in installed:
            print(f"\n[LEWATI] Model `{model}` belum terpasang di Ollama.")
            continue

        print(f"\n--- Menguji Model: {model} ---")
        for ctx in CTX_LIST:
            paksa_unload_semua(jeda_detik=2.0)
            ram_sebelum = ram_available_mib()

            try:
                # Gunakan keep_alive="2m" sementara agar status /api/ps & RAM SAAT termuat terbaca
                perf = kirim_prompt(model=model, num_ctx=ctx, keep_alive="2m")
                ps_info = ambil_status_ps(model)
                ram_saat = ram_available_mib()

                # Unload model setelah diukur untuk mencatat RAM SESUDAH unload
                paksa_unload_semua(jeda_detik=2.0)
                ram_sesudah = ram_available_mib()

                status_aman = evaluasi_batas_aman(ps_info["gpu_pct"], ram_saat, ambang_ram_mib)
                baris = {
                    "model": model,
                    "num_ctx": ctx,
                    "gpu_pct": ps_info["gpu_pct"],
                    "cpu_pct": ps_info["cpu_pct"],
                    "vram_mib": ps_info["vram_mib"],
                    "ram_sebelum": ram_sebelum,
                    "ram_saat": ram_saat,
                    "ram_sesudah": ram_sesudah,
                    "decode_tps": perf["decode_tps"],
                    "status": status_aman,
                }
                semua_hasil.append(baris)

                print(
                    f"  ctx={ctx:<4} | GPU/CPU: {ps_info['gpu_pct']:5.1f}%/{ps_info['cpu_pct']:4.1f}% | "
                    f"RAM Avail (Sblm/Saat/Ssdh): {ram_sebelum:5.0f} / {ram_saat:5.0f} / {ram_sesudah:5.0f} MiB | "
                    f"Decode: {perf['decode_tps']:5.1f} t/s | {status_aman}"
                )
            except requests.RequestException as err:
                print(f"  ctx={ctx:<4} | [GAGAL / OOM] {err}")

    # Cetak ringkasan tabel lengkap
    if semua_hasil:
        print("\n" + "=" * 116)
        print("RINGKASAN TABEL MATRIKS & BATAS AMAN")
        print("=" * 116)
        header = (
            f"{'Model':<20} | {'ctx':>4} | {'GPU/CPU':>11} | {'VRAM':>8} | "
            f"{'RAM Sblm':>9} | {'RAM Saat':>9} | {'RAM Ssdh':>9} | {'Decode':>8} | {'Status Batas Aman'}"
        )
        print(header)
        print("-" * len(header))
        for r in semua_hasil:
            rasio = f"{r['gpu_pct']:.0f}%/{r['cpu_pct']:.0f}%"
            print(
                f"{r['model']:<20} | {r['num_ctx']:>4} | {rasio:>11} | {r['vram_mib']:>5.0f} MiB | "
                f"{r['ram_sebelum']:>6.0f} MiB | {r['ram_saat']:>6.0f} MiB | {r['ram_sesudah']:>6.0f} MiB | "
                f"{r['decode_tps']:>5.1f} t/s | {r['status']}"
            )

    return semua_hasil


# ==============================================================================
# 3. UJI SIKLUS KEEP_ALIVE ("0", "5m", "-1") & PEMULIHAN RAM (LANGKAH 5)
# ==============================================================================
def uji_keep_alive(model: str) -> None:
    print("\n" + "=" * 94)
    print(f"LANGKAH 5: UJI PARAMETER `keep_alive` (`0`, `5m`, `-1`) PADA `{model}`")
    print("=" * 94)

    for mode in KEEP_ALIVE_MODES:
        paksa_unload_semua(jeda_detik=2.0)
        ram_sblm = ram_available_mib()

        print(f"\n>>> Menguji keep_alive = {mode!r} (num_ctx=2048)...")
        kirim_prompt(model=model, num_ctx=2048, keep_alive=mode)

        # Cek kondisi tepat setelah request selesai (t = 0s)
        ps_t0 = ambil_status_ps(model)
        ram_t0 = ram_available_mib()

        # Tunggu 5 detik untuk mengamati apakah model bertahan atau langsung dilepas
        time.sleep(5.0)
        ps_t5 = ambil_status_ps(model)
        ram_t5 = ram_available_mib()

        print(f"  - RAM Sebelum Muat           : {ram_sblm:7.1f} MiB")
        print(
            f"  - Tepat Setelah Prompt (t=0s): {ram_t0:7.1f} MiB "
            f"(Selisih: {ram_t0 - ram_sblm:+6.1f} MiB | Loaded: {ps_t0['loaded']})"
        )
        print(
            f"  - Setelah Jeda 5 Detik (t=5s): {ram_t5:7.1f} MiB "
            f"(Selisih: {ram_t5 - ram_sblm:+6.1f} MiB | Loaded: {ps_t5['loaded']})"
        )

        # Jika model masih tertahan di memori ("5m" atau "-1"), uji manual unload & pemulihan RAM
        if ps_t5["loaded"]:
            paksa_unload_semua(jeda_detik=2.0)
            ps_clean = ambil_status_ps(model)
            ram_pulih = ram_available_mib()
            print(
                f"  - Setelah Manual Unload      : {ram_pulih:7.1f} MiB "
                f"(Selisih vs Awal: {ram_pulih - ram_sblm:+6.1f} MiB | Loaded: {ps_clean['loaded']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Uji Memori Ollama, Batas Aman, & keep_alive")
    parser.add_argument(
        "--min-ram-mib",
        type=float,
        default=1536.0,
        help="Ambang minimal RAM tersedia (MiB) agar dianggap aman (default: 1536 MiB = 1.5 GiB)",
    )
    args = parser.parse_args()

    cek_env_server()

    tags_res = requests.get(f"{HOST}/api/tags", timeout=15)
    tags_res.raise_for_status()
    installed = {m["name"] for m in tags_res.json().get("models", [])}

    # Langkah 2-4: Jalankan matriks semua model x num_ctx
    jalankan_matriks_konteks(installed, ambang_ram_mib=args.min_ram_mib)

    # Langkah 5: Uji keep_alive pada model utama yang tersedia
    kandidat_ka = [m for m in MODELS if m in installed]
    if kandidat_ka:
        uji_keep_alive(kandidat_ka[0])

    paksa_unload_semua(jeda_detik=1.0)
    print("\n[SELESAI] Pengujian selesai dan memori telah dibersihkan.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as err:
        print(f"[ERROR] Gagal terhubung ke server Ollama di {HOST}: {err}")