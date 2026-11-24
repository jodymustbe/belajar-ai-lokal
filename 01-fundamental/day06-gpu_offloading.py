"""Hari ke-6: Eksperimen GPU Layer Offloading, CPU Spillover, & Mitigasinya.

Menguji dua model kelas ~4B (`qwen3:4b` dan `phi4-mini:latest`) pada GPU 4 GB VRAM:
1. Mengukur dampak CPU spillover (offload < 100% GPU) terhadap kecepatan Prefill & Decode.
2. Menguji metode mitigasi:
   - Pemangkasan panjang konteks (`num_ctx`)
   - Tuning manual jumlah layer GPU (`num_gpu`)
   - Penghematan memori KV Cache (`OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0`)

Dependensi:
    pip install requests

Cara menjalankan:
    # Tahap 1: Jalankan pengujian baseline, pemangkasan konteks, dan tuning layer
    python 01-fundamental/day06_gpu_offloading.py

    # Tahap 2 (opsional): Setelah mengaktifkan OLLAMA_FLASH_ATTENTION=1 dan
    # OLLAMA_KV_CACHE_TYPE=q8_0 lalu me-restart Ollama, jalankan mode perbandingan KV-Q8:
    python 01-fundamental/day06_gpu_offloading.py --mode-kv-q8
"""

import argparse
import ctypes
import os
import subprocess
import time
import requests

HOST = "http://localhost:11434"
TARGET_MODELS = ["qwen3.5:4b"]
MIB = 1024 * 1024

# Prompt tetap berukuran menengah (~90 token) agar fase Prefill dan Decode terukur jelas
PROMPT_UJI = (
    "Jelaskan dalam 4 poin singkat bagaimana mekanisme pembagian layer model bahasa "
    "antara memori VRAM GPU dan RAM sistem memengaruhi kecepatan generasi teks "
    "pada perangkat dengan kapasitas VRAM terbatas."
)


# ==============================================================================
# 1. UTILITAS PEMBACAAN MEMORI & METADATA MODEL
# ==============================================================================
def baca_nvidia_smi_mb() -> int | None:
    """Membaca total VRAM terpakai di GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def baca_ram_sistem_mb() -> tuple[int, int] | None:
    """Membaca (RAM Terpakai MB, RAM Total MB) menggunakan API bawaan OS."""
    try:
        if os.name == "nt":

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_mb = int(stat.ullTotalPhys / MIB)
            used_mb = int((stat.ullTotalPhys - stat.ullAvailPhys) / MIB)
            return used_mb, total_mb
        if os.path.exists("/proc/meminfo"):
            mem = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = int(parts[1])
            total_mb = mem["MemTotal"] // 1024
            avail_mb = mem.get("MemAvailable", mem.get("MemFree", 0)) // 1024
            return total_mb - avail_mb, total_mb
    except Exception:
        pass
    return None


def ambil_info_layer(model: str) -> dict:
    """Mengambil jumlah layer (block_count) dan arsitektur model dari /api/show."""
    res = requests.post(f"{HOST}/api/show", json={"model": model}, timeout=30)
    res.raise_for_status()
    info = res.json().get("model_info", {})
    arch = info.get("general.architecture", "unknown")

    layers = info.get(f"{arch}.block_count")
    if layers is None:
        for k, v in info.items():
            if k.endswith(".block_count"):
                layers = v
                break

    return {"architecture": arch, "total_layers": int(layers) if layers else 32}


def unload_semua_model() -> None:
    """Mengeluarkan semua model aktif dari VRAM/RAM agar pengukuran tiap skenario bersih."""
    try:
        ps = requests.get(f"{HOST}/api/ps", timeout=15).json()
        for m in ps.get("models", []):
            requests.post(
                f"{HOST}/api/chat",
                json={"model": m["name"], "keep_alive": 0},
                timeout=30,
            )
        time.sleep(1.5)
    except requests.RequestException:
        pass


# ==============================================================================
# 2. EKSEKUSI SATU SKENARIO OFFLOADING
# ==============================================================================
def jalankan_skenario(model: str, nama_skenario: str, num_ctx: int, num_gpu: int) -> dict:
    unload_semua_model()

    options = {
        "num_ctx": num_ctx,
        "num_gpu": num_gpu,
        "num_predict": 120,
        "temperature": 0,
        "seed": 42,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_UJI}],
        "stream": False,
        "options": options,
    }
    # Nonaktifkan mode berpikir panjang khusus keluarga Qwen3 agar jumlah token sebanding
    if "qwen3" in model.lower():
        payload["think"] = False

    t0 = time.perf_counter()
    res = requests.post(f"{HOST}/api/chat", json=payload, timeout=300)
    res.raise_for_status()
    wall_sec = time.perf_counter() - t0
    data = res.json()

    p_tok = data.get("prompt_eval_count", 0)
    p_sec = data.get("prompt_eval_duration", 1) / 1e9
    d_tok = data.get("eval_count", 0)
    d_sec = data.get("eval_duration", 1) / 1e9

    prefill_tps = p_tok / p_sec if p_sec > 0 else 0.0
    decode_tps = d_tok / d_sec if d_sec > 0 else 0.0

    # Baca rasio offload GPU vs CPU dari /api/ps
    size_total = 0
    size_vram = 0
    ps_res = requests.get(f"{HOST}/api/ps", timeout=15)
    if ps_res.status_code == 200:
        for m in ps_res.json().get("models", []):
            if m.get("name") == model:
                size_total = m.get("size", 0)
                size_vram = m.get("size_vram", 0)

    gpu_pct = (size_vram / size_total * 100.0) if size_total > 0 else 0.0
    ram_info = baca_ram_sistem_mb()

    return {
        "skenario": nama_skenario,
        "num_ctx": num_ctx,
        "num_gpu": num_gpu,
        "total_mib": size_total / MIB,
        "vram_mib": size_vram / MIB,
        "gpu_pct": gpu_pct,
        "cpu_pct": max(0.0, 100.0 - gpu_pct),
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "wall_sec": wall_sec,
        "smi_mb": baca_nvidia_smi_mb(),
        "ram_used_mb": ram_info[0] if ram_info else None,
        "ram_total_mb": ram_info[1] if ram_info else None,
    }


def cetak_tabel_hasil(model: str, total_layers: int, hasil_list: list[dict]) -> None:
    print(f"\nRingkasan Hasil `{model}` (Total Layers = {total_layers}):")
    header = (
        f"  {'Skenario':<20} | {'ctx':>4} | {'num_gpu':>7} | {'% GPU':>6} | "
        f"{'VRAM':>9} | {'Prefill':>10} | {'Decode':>9} | {'RAM Sys':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in hasil_list:
        gpu_arg = "auto" if r["num_gpu"] == -1 else str(r["num_gpu"])
        ram_str = f"{r['ram_used_mb']} MB" if r["ram_used_mb"] is not None else "N/A"
        print(
            f"  {r['skenario']:<20} | {r['num_ctx']:>4} | {gpu_arg:>7} | "
            f"{r['gpu_pct']:>5.1f}% | {r['vram_mib']:>6.0f} MiB | "
            f"{r['prefill_tps']:>6.1f} t/s | {r['decode_tps']:>5.1f} t/s | {ram_str:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hari ke-6: Uji GPU Layer Offloading & Mitigasi")
    parser.add_argument(
        "--mode-kv-q8",
        action="store_true",
        help="Uji skenario S5 (num_ctx=4096 dengan KV cache q8_0 & Flash Attention aktif)",
    )
    args = parser.parse_args()

    print("=" * 84)
    print("HARI KE-6: EKSPERIMEN GPU LAYER OFFLOADING & CPU SPILLOVER")
    print("=" * 84)

    tags_res = requests.get(f"{HOST}/api/tags", timeout=15)
    tags_res.raise_for_status()
    installed = {m["name"] for m in tags_res.json().get("models", [])}

    for model in TARGET_MODELS:
        if model not in installed:
            print(f"\n[LEWATI] Model `{model}` belum terpasang di Ollama.")
            continue

        meta = ambil_info_layer(model)
        total_layers = meta["total_layers"]
        layer_50 = max(1, total_layers // 2)
        layer_85 = max(1, int(total_layers * 0.85))

        print("\n" + "-" * 84)
        print(f"MODEL: {model} (Arsitektur: {meta['architecture']}, Total Layers: {total_layers})")
        print("-" * 84)

        if args.mode_kv_q8:
            skenario_list = [
                ("S5_KV_Q8_ctx4096", 4096, -1),
                ("S5_KV_Q8_ctx2048", 2048, -1),
            ]
        else:
            skenario_list = [
                ("S1_Auto_ctx4096", 4096, -1),
                ("S2_Full_CPU_0GPU", 2048, 0),
                ("S3_Trim_ctx1024", 1024, -1),
                (f"S4_Manual_{layer_50}L", 2048, layer_50),
                (f"S4_Manual_{layer_85}L", 2048, layer_85),
                (f"S4_Force_All_{total_layers}L", 2048, total_layers),
            ]

        hasil_model = []
        for nama, ctx, ngpu in skenario_list:
            print(f"  Menjalankan {nama} (num_ctx={ctx}, num_gpu={ngpu}) ...")
            try:
                res = jalankan_skenario(model, nama, ctx, ngpu)
                hasil_model.append(res)
                print(
                    f"    -> Offload: {res['gpu_pct']:.1f}% GPU | "
                    f"Prefill: {res['prefill_tps']:.1f} tok/s | Decode: {res['decode_tps']:.1f} tok/s"
                )
            except requests.RequestException as err:
                print(f"    -> [GAGAL / OOM] {nama}: {err}")

        if hasil_model:
            cetak_tabel_hasil(model, total_layers, hasil_model)

    unload_semua_model()
    print("\n[SELESAI] Semua model telah dikosongkan dari VRAM.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as err:
        print(f"[ERROR] Gagal terhubung ke Ollama di {HOST}: {err}")