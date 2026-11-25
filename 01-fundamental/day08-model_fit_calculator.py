"""Mini-Proyek: Kalkulator CLI Estimasi VRAM & Kelayakan 100% GPU (`model_fit_calc.py`).

Dikalibrasi berdasarkan hasil eksperimen Hari ke-4 hingga Hari ke-7 pada GPU 4 GB VRAM:
1. Membaca metadata arsitektur & bobot dari file GGUF (`gguf-py`) atau API Ollama (`/api/show`).
2. Menghitung rincian VRAM: Bobot Model + KV Cache (`f16`/`q8_0`/`q4_0`) + Compute Overhead.
3. Menilai kelayakan 100% GPU, estimasi rasio GPU/CPU otomatis, serta rekomendasi `num_gpu` & `num_ctx`.
4. Memvalidasi akurasi kalkulator secara langsung terhadap 3 model berbeda via `/api/ps`.

Dependensi:
    pip install gguf requests

Contoh penggunaan:
    # 1. Cek kelayakan satu model pada konteks tertentu:
    py 01-fundamental/model_fit_calc.py --model qwen3.5:2b-q4_K_M --ctx 8192 --kv-type q8_0

    # 2. Cek langsung dari file .gguf lokal:
    py 01-fundamental/model_fit_calc.py --gguf path/to/model.gguf --ctx 4096

    # 3. Validasi akurasi prediksi vs pengukuran nyata pada 3 model berbeda:
    py 01-fundamental/model_fit_calc.py --validate
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from gguf import GGUFReader, GGUFValueType
import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MIB = 1024 * 1024
GIB = 1024**3

DEFAULT_VALIDATION_MODELS = [
    "qwen3.5:2b-q4_K_M",
    "qwen3:4b",
    "phi4-mini:latest",
]

KV_BYTES_PER_ELEM = {
    "f16": 2.0,
    "fp16": 2.0,
    "q8_0": 34.0 / 32.0,  # 32 elemen int8 + 1 skala fp16 (2 byte)
    "q4_0": 18.0 / 32.0,  # 32 elemen 4-bit + 1 skala fp16 (2 byte)
}


@dataclass
class ModelMetadata:
    name: str
    arch: str
    total_params: int
    weights_mib: float
    bpw: float
    n_layers: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    source: str


@dataclass
class FitPrediction:
    weights_mib: float
    kv_mib: float
    overhead_mib: float
    total_est_mib: float
    usable_vram_mib: float
    margin_mib: float
    fit_100_gpu: bool
    est_auto_gpu_pct: float
    rec_num_gpu: int
    rec_safe_ctx: int


# ==============================================================================
# 1. DETEKSI HARDWARE & PEMBACAAN METADATA (GGUF / OLLAMA)
# ==============================================================================
def deteksi_vram_gpu_mib(default_total_mib: float = 4096.0) -> tuple[float, str]:
    """Mendeteksi kapasitas VRAM GPU melalui nvidia-smi jika tersedia."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        baris = out.strip().splitlines()[0].split(",")
        gpu_name = baris[0].strip()
        total_mib = float(baris[1].strip())
        return total_mib, gpu_name
    except Exception:
        return default_total_mib, "GPU Default (4 GB)"


def ekstrak_field_gguf(field) -> object:
    if not field.types:
        return None
    tipe = field.types[0]
    if tipe == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    if tipe == GGUFValueType.ARRAY:
        return len(field.data)
    try:
        val = field.parts[field.data[0]][0]
        return val.item() if hasattr(val, "item") else val
    except Exception:
        return None


def baca_metadata_dari_gguf(gguf_path: Path, nama_tampil: str | None = None) -> ModelMetadata:
    """Membaca parameter arsitektur dan ukuran bobot langsung dari header file GGUF."""
    reader = GGUFReader(str(gguf_path), mode="r")
    arch_f = reader.fields.get("general.architecture")
    arch = str(ekstrak_field_gguf(arch_f)) if arch_f else "unknown"

    total_params = sum(int(t.n_elements) for t in reader.tensors)
    tensor_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    file_bytes = gguf_path.stat().st_size
    bpw = (tensor_bytes * 8.0 / total_params) if total_params > 0 else 4.85

    def get_int(suffix: str, fallback: int) -> int:
        f = reader.fields.get(f"{arch}.{suffix}")
        val = ekstrak_field_gguf(f) if f else None
        return int(val) if isinstance(val, (int, float)) else fallback

    n_layers = get_int("block_count", 32)
    n_embd = get_int("embedding_length", 2048)
    n_head = get_int("attention.head_count", 16)
    n_head_kv = get_int("attention.head_count_kv", n_head)
    head_dim = get_int("attention.key_length", max(1, n_embd // max(1, n_head)))

    return ModelMetadata(
        name=nama_tampil or gguf_path.stem,
        arch=arch,
        total_params=total_params,
        weights_mib=file_bytes / MIB,
        bpw=bpw,
        n_layers=n_layers,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        source=f"GGUF ({gguf_path.name[:20]}...)",
    )


def baca_metadata_dari_ollama(model_name: str) -> ModelMetadata:
    """Mengambil metadata model dari Ollama (/api/show + file blob GGUF jika lokal)."""
    res = requests.post(f"{HOST}/api/show", json={"model": model_name}, timeout=20)
    res.raise_for_status()
    payload = res.json()

    # Jika file blob GGUF bisa diakses langsung di disk, baca header GGUF-nya secara akurat
    modelfile = payload.get("modelfile", "")
    for baris in modelfile.splitlines():
        if baris.startswith("FROM "):
            kandidat = Path(baris[5:].strip().strip('"'))
            if kandidat.is_file():
                return baca_metadata_dari_gguf(kandidat, nama_tampil=model_name)

    # Fallback jika Ollama berjalan remote dan blob tidak diakses langsung
    info = payload.get("model_info", {})
    arch = str(info.get("general.architecture", "unknown"))
    total_params = int(info.get("general.parameter_count", 0))

    # Estimasi ukuran bobot dari tags atau rumus (params * bpw) / 8
    weights_mib = 0.0
    tags = requests.get(f"{HOST}/api/tags", timeout=15).json().get("models", [])
    for m in tags:
        if m.get("name") == model_name:
            weights_mib = m.get("size", 0) / MIB
            break
    if weights_mib <= 0 and total_params > 0:
        weights_mib = (total_params * 4.85 / 8.0) / MIB

    bpw = (weights_mib * MIB * 8.0 / total_params) if total_params > 0 else 4.85
    n_layers = int(info.get(f"{arch}.block_count", 32))
    n_embd = int(info.get(f"{arch}.embedding_length", 2048))
    n_head = int(info.get(f"{arch}.attention.head_count", 16))
    n_head_kv = int(info.get(f"{arch}.attention.head_count_kv", n_head))
    head_dim = int(info.get(f"{arch}.attention.key_length", max(1, n_embd // max(1, n_head))))

    return ModelMetadata(
        name=model_name,
        arch=arch,
        total_params=total_params,
        weights_mib=weights_mib,
        bpw=bpw,
        n_layers=n_layers,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        source="Ollama /api/show",
    )


# ==============================================================================
# 2. KALKULATOR ESTIMASI VRAM & KELAYAKAN 100% GPU
# ==============================================================================
def hitung_estimasi_vram(
    meta: ModelMetadata,
    num_ctx: int,
    kv_type: str = "q8_0",
    total_vram_mib: float = 4096.0,
    os_reserve_mib: float = 1200.0,
) -> FitPrediction:
    """Menghitung kebutuhan VRAM total, status 100% GPU, dan rekomendasi layer/konteks."""
    kv_key = kv_type.lower()
    if kv_key not in KV_BYTES_PER_ELEM:
        raise ValueError(f"Tipe KV cache tidak dikenal: {kv_type}. Pilih f16, q8_0, atau q4_0.")

    bytes_per_elem = KV_BYTES_PER_ELEM[kv_key]

    # Arsitektur hybrid Gated DeltaNet (qwen35) hanya memakai full GQA KV cache pada 1/4 layer
    is_hybrid_qwen35 = "qwen35" in meta.arch.lower()
    n_kv_layers = max(1, meta.n_layers // 4) if is_hybrid_qwen35 else meta.n_layers

    kv_mib = (2.0 * n_kv_layers * meta.n_head_kv * meta.head_dim * num_ctx * bytes_per_elem) / MIB
    if is_hybrid_qwen35:
        kv_mib += 16.0  # Buffer state rekuren DeltaNet

    # Kalibrasi empiris overhead CUDA context + compute buffer (Hasil Hari ke-5 s.d. ke-7)
    overhead_mib = 240.0 + (meta.weights_mib * 0.04) + (num_ctx / 1024.0) * 10.0
    total_est_mib = meta.weights_mib + kv_mib + overhead_mib

    usable_vram_mib = max(0.0, total_vram_mib - os_reserve_mib)
    margin_mib = usable_vram_mib - total_est_mib
    fit_100_gpu = margin_mib >= 0.0

    # Penjadwal otomatis Ollama menyisakan ~550 MiB safety margin sebelum offload sebagian ke CPU
    auto_budget_mib = max(0.0, usable_vram_mib - 550.0)
    if total_est_mib <= auto_budget_mib:
        est_auto_gpu_pct = 100.0
    else:
        avail_for_weights = max(0.0, auto_budget_mib - kv_mib - overhead_mib)
        est_auto_gpu_pct = round(min(99.0, (avail_for_weights / meta.weights_mib) * 100.0), 1)

    # Hitung rekomendasi maksimal `num_gpu` layer agar seluruh blok transformer masuk ke VRAM
    if fit_100_gpu:
        rec_num_gpu = meta.n_layers
    else:
        avail_for_blocks = max(0.0, usable_vram_mib - kv_mib - overhead_mib)
        rec_num_gpu = max(1, min(meta.n_layers, int(meta.n_layers * (avail_for_blocks / meta.weights_mib))))

    # Cari konteks aman maksimal (dari kandidat standar)
    rec_safe_ctx = 512
    for kandidat_ctx in [8192, 4096, 2048, 1024, 512]:
        kv_k = (2.0 * n_kv_layers * meta.n_head_kv * meta.head_dim * kandidat_ctx * bytes_per_elem) / MIB
        if is_hybrid_qwen35:
            kv_k += 16.0
        ovh_k = 240.0 + (meta.weights_mib * 0.04) + (kandidat_ctx / 1024.0) * 10.0
        if (meta.weights_mib + kv_k + ovh_k) <= usable_vram_mib:
            rec_safe_ctx = kandidat_ctx
            break

    return FitPrediction(
        weights_mib=round(meta.weights_mib, 1),
        kv_mib=round(kv_mib, 1),
        overhead_mib=round(overhead_mib, 1),
        total_est_mib=round(total_est_mib, 1),
        usable_vram_mib=round(usable_vram_mib, 1),
        margin_mib=round(margin_mib, 1),
        fit_100_gpu=fit_100_gpu,
        est_auto_gpu_pct=est_auto_gpu_pct,
        rec_num_gpu=rec_num_gpu,
        rec_safe_ctx=rec_safe_ctx,
    )


def cetak_laporan_tunggal(
    meta: ModelMetadata,
    pred: FitPrediction,
    num_ctx: int,
    kv_type: str,
    gpu_name: str,
    total_vram_mib: float,
) -> None:
    print("\n" + "=" * 76)
    print(f"KALKULATOR KELAYAKAN VRAM & 100% GPU: {meta.name}")
    print("=" * 76)
    print(f"  Sumber Metadata   : {meta.source}")
    print(f"  Arsitektur        : {meta.arch} ({meta.n_layers} layers, KV heads={meta.n_head_kv}, dim={meta.head_dim})")
    print(f"  Parameter & BPW   : {meta.total_params / 1e9:.3f} B | {meta.bpw:.2f} bits/weight")
    print(f"  Target GPU        : {gpu_name} (Total={total_vram_mib:.0f} MiB | Usable={pred.usable_vram_mib:.0f} MiB)")
    print(f"  Konfigurasi Uji   : num_ctx={num_ctx} | KV Cache={kv_type.upper()}")

    print("\n[Rincian Estimasi Kebutuhan VRAM]")
    print(f"  1. Bobot Model (GGUF)      : {pred.weights_mib:8.1f} MiB")
    print(f"  2. Alokasi KV Cache        : {pred.kv_mib:8.1f} MiB")
    print(f"  3. CUDA & Compute Overhead : {pred.overhead_mib:8.1f} MiB")
    print("  " + "-" * 42)
    print(f"  Total Estimasi VRAM        : {pred.total_est_mib:8.1f} MiB")
    print(f"  Sisa Margin VRAM           : {pred.margin_mib:+8.1f} MiB")

    status_str = "LAYAK 100% GPU (AMAN)" if pred.fit_100_gpu else "TIDAK MUAT 100% GPU (CPU SPILLOVER)"
    print("\n[Keputusan & Rekomendasi]")
    print(f"  Status Kelayakan  : {status_str}")
    print(f"  Prediksi Auto GPU : {pred.est_auto_gpu_pct:.1f}% GPU (mode default Ollama)")
    if not pred.fit_100_gpu:
        print(
            f"  Saran Mitigasi    : Gunakan `num_ctx={pred.rec_safe_ctx}` atau paksa layer "
            f"`num_gpu={pred.rec_num_gpu}` dengan `OLLAMA_KV_CACHE_TYPE=q8_0`."
        )


# ==============================================================================
# 3. VALIDASI AKURASI TERHADAP 3 MODEL BERBEDA VIA OLLAMA
# ==============================================================================
def ukur_aktual_ollama(model: str, num_ctx: int) -> dict:
    """Memuat model ke Ollama dan membaca rasio GPU/CPU serta VRAM aktual dari /api/ps."""
    # Unload model lain agar pembacaan /api/ps bersih
    ps_old = requests.get(f"{HOST}/api/ps", timeout=15).json()
    for m in ps_old.get("models", []):
        requests.post(f"{HOST}/api/chat", json={"model": m["name"], "messages": [], "keep_alive": 0}, timeout=30)
    time.sleep(1.0)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Jawab singkat: apa fungsi KV cache?"}],
        "stream": False,
        "keep_alive": "2m",
        "options": {"num_ctx": num_ctx, "num_predict": 24, "temperature": 0},
    }
    if "qwen3" in model.lower():
        payload["think"] = False

    res = requests.post(f"{HOST}/api/chat", json=payload, timeout=300)
    res.raise_for_status()
    data = res.json()
    d_tok = data.get("eval_count", 0)
    d_sec = data.get("eval_duration", 1) / 1e9
    decode_tps = d_tok / d_sec if d_sec > 0 else 0.0

    ps_now = requests.get(f"{HOST}/api/ps", timeout=15).json()
    vram_mib = 0.0
    gpu_pct = 0.0
    for m in ps_now.get("models", []):
        if m.get("name") == model:
            tot = m.get("size", 0)
            vram = m.get("size_vram", 0)
            vram_mib = vram / MIB
            gpu_pct = (vram / tot * 100.0) if tot > 0 else 0.0

    # Bersihkan kembali
    requests.post(f"{HOST}/api/chat", json={"model": model, "messages": [], "keep_alive": 0}, timeout=30)

    return {
        "actual_vram_mib": round(vram_mib, 1),
        "actual_gpu_pct": round(gpu_pct, 1),
        "actual_fit_100": gpu_pct >= 99.9,
        "decode_tps": round(decode_tps, 1),
    }


def jalankan_validasi_3_model(
    models: list[str],
    num_ctx: int,
    kv_type: str,
    total_vram_mib: float,
    os_reserve_mib: float,
    gpu_name: str,
) -> None:
    print("\n" + "=" * 106)
    print(f"VALIDASI AKURASI KALKULATOR PADA {len(models)} MODEL BERBEDA ({gpu_name}, ctx={num_ctx}, KV={kv_type})")
    print("=" * 106)

    tags = {m["name"] for m in requests.get(f"{HOST}/api/tags", timeout=15).json().get("models", [])}
    hasil_validasi = []

    for model in models:
        if model not in tags:
            print(f"[LEWATI] Model `{model}` belum terpasang di Ollama.")
            continue

        print(f"  Memvalidasi `{model}` ...")
        meta = baca_metadata_dari_ollama(model)
        pred = hitung_estimasi_vram(meta, num_ctx, kv_type, total_vram_mib, os_reserve_mib)
        aktual = ukur_aktual_ollama(model, num_ctx)

        cocok_keputusan = pred.fit_100_gpu == aktual["actual_fit_100"]
        selisih_gpu_pct = pred.est_auto_gpu_pct - aktual["actual_gpu_pct"]

        hasil_validasi.append(
            {
                "model": model,
                "pred_vram": pred.total_est_mib,
                "act_vram": aktual["actual_vram_mib"],
                "pred_fit": "100% GPU" if pred.fit_100_gpu else "SPILLOVER",
                "act_fit": "100% GPU" if aktual["actual_fit_100"] else "SPILLOVER",
                "pred_gpu_pct": pred.est_auto_gpu_pct,
                "act_gpu_pct": aktual["actual_gpu_pct"],
                "selisih_pct": selisih_gpu_pct,
                "decode_tps": aktual["decode_tps"],
                "akurasi_status": "COCOK" if cocok_keputusan else "MELESET",
            }
        )

    if not hasil_validasi:
        print("[PERINGATAN] Tidak ada model target yang ditemukan untuk divalidasi.")
        return

    print("\n" + "-" * 106)
    header = (
        f"{'Model':<20} | {'Est VRAM':>9} | {'Act VRAM':>9} | {'Pred %GPU':>9} | "
        f"{'Act %GPU':>8} | {'Status Pred vs Act':<21} | {'Decode':>8} | {'Validasi'}"
    )
    print(header)
    print("-" * 106)

    cocok_total = 0
    for r in hasil_validasi:
        if r["akurasi_status"] == "COCOK":
            cocok_total += 1
        status_pair = f"{r['pred_fit']} vs {r['act_fit']}"
        print(
            f"{r['model']:<20} | {r['pred_vram']:>6.0f} MiB | {r['act_vram']:>6.0f} MiB | "
            f"{r['pred_gpu_pct']:>8.1f}% | {r['act_gpu_pct']:>7.1f}% | {status_pair:<21} | "
            f"{r['decode_tps']:>5.1f} t/s | {r['akurasi_status']}"
        )

    print("-" * 106)
    print(
        f"Akurasi Klasifikasi Kelayakan 100% GPU: {cocok_total}/{len(hasil_validasi)} model "
        f"({cocok_total / len(hasil_validasi) * 100:.0f}% tepat)."
    )


def main() -> None:
    vram_terdeteksi, nama_gpu = deteksi_vram_gpu_mib(4096.0)
    default_kv = os.environ.get("OLLAMA_KV_CACHE_TYPE", "q8_0")

    parser = argparse.ArgumentParser(description="Kalkulator CLI Estimasi VRAM & Kelayakan 100% GPU")
    parser.add_argument("--model", type=str, default="qwen3.5:2b-q4_K_M", help="Nama model Ollama")
    parser.add_argument("--gguf", type=Path, default=None, help="Path langsung ke file .gguf lokal")
    parser.add_argument("--ctx", type=int, default=4096, help="Panjang konteks num_ctx (default: 4096)")
    parser.add_argument("--kv-type", type=str, default=default_kv, choices=["f16", "q8_0", "q4_0"], help="Tipe KV cache")
    parser.add_argument("--vram-mb", type=float, default=vram_terdeteksi, help="Total VRAM GPU dalam MiB")
    parser.add_argument("--os-reserve-mb", type=float, default=1200.0, help="Reservasi VRAM OS & driver (default: 1200 MiB)")
    parser.add_argument("--validate", action="store_true", help="Validasikan kalkulator pada 3 model berbeda via Ollama")
    parser.add_argument("--models", nargs="+", default=DEFAULT_VALIDATION_MODELS, help="Daftar model untuk mode --validate")
    args = parser.parse_args()

    if args.validate:
        jalankan_validasi_3_model(
            models=args.models,
            num_ctx=args.ctx,
            kv_type=args.kv_type,
            total_vram_mib=args.vram_mb,
            os_reserve_mib=args.os_reserve_mb,
            gpu_name=nama_gpu,
        )
        return

    meta = baca_metadata_dari_gguf(args.gguf) if args.gguf else baca_metadata_dari_ollama(args.model)
    pred = hitung_estimasi_vram(
        meta=meta,
        num_ctx=args.ctx,
        kv_type=args.kv_type,
        total_vram_mib=args.vram_mb,
        os_reserve_mib=args.os_reserve_mb,
    )
    cetak_laporan_tunggal(meta, pred, args.ctx, args.kv_type, nama_gpu, args.vram_mb)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as err:
        print(f"[ERROR] Gagal terhubung ke server Ollama di {HOST}: {err}")