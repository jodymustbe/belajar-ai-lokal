"""Hari ke-5: Simulasi Kuantisasi Blok (ggml), Inspeksi GGUF, & Uji 10 Prompt Tetap.

Struktur Skrip:
1. Simulasi kuantisasi blok ala ggml (`Q8_0`, `Q4_0`, vs `int4` per-tensor) dengan NumPy
   untuk melihat pengaruh overhead skala `fp16` (BPW) dan ketahanan terhadap outlier.
2. Inspeksi file GGUF asli (`qwen3.5:2b-q4_K_M`) menggunakan `gguf-py`: membaca
   header, metadata, sebaran tipe tensor, dan estimasi ukuran `(params * bpw) / 8`.
3. Benchmark 10 prompt tetap via Ollama API (`Q4_K_M`, serta `Q8_0` jika terpasang).

Dependensi:
    pip install numpy gguf requests

Cara menjalankan:
    python 01-fundamental/day05_gguf_quantization.py
    # Jika ingin menjalankan Bagian 1 & 2 saja tanpa memanggil Ollama:
    python 01-fundamental/day05_gguf_quantization.py --skip-ollama
"""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import time
from gguf import GGUFReader, GGUFValueType
import numpy as np
import requests

HOST = "http://localhost:11434"
MODEL_Q4 = "qwen3.5:2b-q4_K_M"
MODEL_Q8 = "qwen3.5:2b-q8_0"

BLOK = 32
MIB = 1024 * 1024
GIB = 1024**3

# 10 Prompt Tetap untuk menguji dampak kuantisasi lintas aspek
PROMPT_BENCHMARK_10 = [
    {
        "id": "P01_Logika",
        "prompt": (
            "Semua sensor kelembapan tanah tipe kapasitif di lahan A butuh kalibrasi ulang. "
            "Sensor S-04 adalah sensor kapasitif di lahan A. "
            "Apakah S-04 perlu dikalibrasi ulang? Jawab Ya atau Tidak beserta 1 kalimat alasan."
        ),
    },
    {
        "id": "P02_Aritmatika",
        "prompt": (
            "Sebuah pompa irigasi mengalirkan air dengan debit 2,5 liter per detik. "
            "Berapa liter air yang terkumpul jika pompa menyala selama 12 menit? "
            "Tulis langkah hitung singkat dan hasil akhirnya."
        ),
    },
    {
        "id": "P03_Coding",
        "prompt": (
            "Tulis fungsi Python bernama `adc_ke_persen(nilai_adc, kering=3200, basah=1400)` "
            "untuk mengubah pembacaan sensor tanah 12-bit menjadi persentase 0-100% (termasuk clamping). "
            "Berikan kodenya saja tanpa penjelasan panjang."
        ),
    },
    {
        "id": "P04_JSON_Strict",
        "prompt": (
            "Ekstrak data berikut ke JSON valid dengan kunci `suhu_c`, `kelembapan_pct`, dan `status`: "
            "'Pembacaan node 2 mencatat suhu 31.5 C dan kelembapan udara 64% dalam kondisi normal.' "
            "Keluarkan JSON murni saja tanpa blok markdown."
        ),
    },
    {
        "id": "P05_Domain_Pertanian",
        "prompt": (
            "Jelaskan dalam maksimal 3 kalimat perbedaan antara kapasitas lapang (field capacity) "
            "dan titik layu permanen (permanent wilting point) pada tanah pertanian."
        ),
    },
    {
        "id": "P06_Ringkasan",
        "prompt": (
            "Ringkas paragraf ini menjadi tepat 2 poin singkat: "
            "'Kuantisasi GGUF mengecilkan ukuran bobot model bahasa dari presisi 16-bit menjadi 4-bit "
            "sehingga konsumsi VRAM GPU berkurang drastis dan kecepatan baca memori meningkat. "
            "Namun, pemotongan presisi yang terlalu ekstrem pada lapisan sensitif dapat menurunkan "
            "akurasi penalaran dan format keluaran terstruktur.'"
        ),
    },
    {
        "id": "P07_Instruksi_Negatif",
        "prompt": (
            "Sebutkan 3 komponen utama sistem irigasi tetes dalam 1 kalimat, "
            "tetapi JANGAN gunakan kata 'air' maupun 'tanaman' sama sekali."
        ),
    },
    {
        "id": "P08_Terjemahan",
        "prompt": (
            "Terjemahkan kalimat teknis ini ke bahasa Indonesia yang alami: "
            "'Mixed-precision quantization keeps sensitive attention layers at higher bit-widths "
            "to preserve reasoning accuracy.'"
        ),
    },
    {
        "id": "P09_Kepatuhan_Fakta",
        "prompt": (
            "Konteks: Kadar air tanah terukur 18% (batas minimum aman tanaman cabai adalah 35%). "
            "Berdasarkan konteks tersebut saja, apakah lahan perlu segera disiram? Jawab singkat."
        ),
    },
    {
        "id": "P10_Konsep_GGUF",
        "prompt": (
            "Jelaskan dalam 2 kalimat mengapa kuantisasi Q4_K_M memiliki ukuran rata-rata "
            "lebih dari 4 bit per parameter."
        ),
    },
]


# ==============================================================================
# BAGIAN 1: SIMULASI KUANTISASI BLOK ALA GGML (NUMPY)
# ==============================================================================
def q8_0(x: np.ndarray) -> np.ndarray:
    """Simulasi Q8_0: 32 bobot int8 (32 B) + 1 skala fp16 (2 B) = 34 B/blok (8.5 bpw)."""
    b = x.reshape(-1, BLOK)
    d = np.abs(b).max(axis=1, keepdims=True) / 127.0
    d16 = d.astype(np.float16).astype(np.float32)
    d16[d16 == 0] = 1.0
    q = np.round(b / d16).clip(-127, 127).astype(np.int8)
    return (q * d16).reshape(-1)


def q4_0(x: np.ndarray) -> np.ndarray:
    """Simulasi Q4_0: 32 bobot 4-bit (16 B) + 1 skala fp16 (2 B) = 18 B/blok (4.5 bpw)."""
    b = x.reshape(-1, BLOK)
    idx = np.abs(b).argmax(axis=1)
    mx = b[np.arange(len(b)), idx][:, None]
    d = (mx / -8.0).astype(np.float16).astype(np.float32)
    d[d == 0] = 1.0
    q = np.floor(b / d + 8.5).clip(0, 15).astype(np.uint8)
    return ((q.astype(np.float32) - 8) * d).reshape(-1)


def int4_per_tensor(x: np.ndarray) -> np.ndarray:
    """Kuantisasi 4-bit naif tanpa blok: satu skala untuk seluruh tensor."""
    d = np.abs(x).max() / 7.0
    if d == 0:
        return np.zeros_like(x)
    return np.round(x / d).clip(-8, 7) * d


def rel_err(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y) / np.linalg.norm(x))


def demo_kuantisasi_blok() -> None:
    print("=" * 78)
    print("BAGIAN 1: SIMULASI KUANTISASI BLOK ALA GGML (Q8_0 & Q4_0 DENGAN NUMPY)")
    print("=" * 78)

    bpw_q8 = (2 + 32) * 8 / BLOK
    bpw_q4 = (2 + 16) * 8 / BLOK
    print(f"BPW Teoritis Blok {BLOK}: Q8_0 = {bpw_q8:.2f} bpw | Q4_0 = {bpw_q4:.2f} bpw")
    print(
        "Alasan: Setiap blok berisi 32 bobot + 1 faktor skala fp16 (2 byte = 16 bit),\n"
        "        sehingga menambah +0.5 bit overhead di atas 8-bit maupun 4-bit murni."
    )

    rng = np.random.default_rng(0)
    n = BLOK * 4096
    normal = rng.normal(0, 0.02, n).astype(np.float32)
    outlier = normal.copy()
    outlier[rng.choice(n, 20, replace=False)] *= 40  # 20 outlier ekstrem

    for nama, x in [("Distribusi Normal (Tanpa Outlier)", normal), ("Dengan 20 Outlier Ekstrem (40x)", outlier)]:
        print(f"\n--- {nama} ---")
        for fn, lbl in [
            (q8_0, "Q8_0 (blok 32)"),
            (q4_0, "Q4_0 (blok 32)"),
            (int4_per_tensor, "4-bit 1 skala/tensor"),
        ]:
            print(f"  {lbl:<22} | galat relatif = {rel_err(x, fn(x)):.4f}")


# ==============================================================================
# BAGIAN 2: INSPEKSI FILE GGUF ASLI (HEADER, METADATA, TENSOR, ESTIMASI BOBOT)
# ==============================================================================
def ekstrak_nilai_field(field) -> object:
    if not field.types:
        return None
    tipe_utama = field.types[0]

    if tipe_utama == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    if tipe_utama == GGUFValueType.ARRAY:
        return f"<Array len={len(field.data)}>"
    try:
        val = field.parts[field.data[0]][0]
        return val.item() if hasattr(val, "item") else val
    except Exception:
        return str(field.parts[-1])


def cari_blob_dari_ollama(model_name: str) -> Path | None:
    try:
        res = requests.post(f"{HOST}/api/show", json={"model": model_name}, timeout=15)
        if res.status_code != 200:
            return None
        modelfile = res.json().get("modelfile", "")
        for baris in modelfile.splitlines():
            if baris.startswith("FROM "):
                kandidat = Path(baris[5:].strip().strip('"'))
                if kandidat.is_file():
                    return kandidat
    except requests.RequestException:
        pass
    return None


def cari_blob_di_folder(models_dir: Path, kata_kunci_arch: str = "qwen35") -> Path | None:
    blobs_dir = models_dir / "blobs"
    if not blobs_dir.is_dir():
        return None

    kandidat_files = sorted(
        blobs_dir.iterdir(), key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True
    )
    for file_path in kandidat_files:
        if not file_path.is_file() or file_path.stat().st_size < 100 * MIB:
            continue
        try:
            reader = GGUFReader(str(file_path), mode="r")
            arch_field = reader.fields.get("general.architecture")
            arch = ekstrak_nilai_field(arch_field) if arch_field else ""
            if kata_kunci_arch.lower() in str(arch).lower():
                return file_path
        except Exception:
            continue
    return None


def inspeksi_gguf(blob_path: Path, label_model: str) -> None:
    print("\n" + "=" * 78)
    print(f"BAGIAN 2: INSPEKSI FILE GGUF ASLI ({label_model})")
    print("=" * 78)
    print(f"File Blob     : {blob_path.name[:24]}... ({blob_path.stat().st_size / GIB:.2f} GiB)")

    reader = GGUFReader(str(blob_path), mode="r")

    ukuran_file_bytes = blob_path.stat().st_size
    total_tensor_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    overhead_bytes = ukuran_file_bytes - total_tensor_bytes
    total_params = sum(int(t.n_elements) for t in reader.tensors)

    print("\n[1] Header & Ringkasan Struktur File:")
    print(f"    - Byte Order / Endian  : {reader.byte_order}")
    print(f"    - Jumlah Metadata KV   : {len(reader.fields)} entri")
    print(f"    - Jumlah Tensor        : {len(reader.tensors)} tensor")
    print(f"    - Total Parameter      : {total_params:,} ({total_params / 1e9:.3f} B)")
    print(f"    - Payload Bobot Murni  : {total_tensor_bytes / MIB:,.2f} MiB")
    print(f"    - Overhead Header/Vocab: {overhead_bytes / MIB:,.2f} MiB")
    print(f"    - Ukuran File Total    : {ukuran_file_bytes / MIB:,.2f} MiB ({ukuran_file_bytes / GIB:.2f} GiB)")

    arch_field = reader.fields.get("general.architecture")
    arch = str(ekstrak_nilai_field(arch_field)) if arch_field else "unknown"

    kunci_penting = [
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        f"{arch}.block_count",
        f"{arch}.embedding_length",
        f"{arch}.attention.head_count",
        f"{arch}.attention.head_count_kv",
        f"{arch}.attention.key_length",
        "tokenizer.ggml.model",
    ]
    print("\n[2] Metadata Utama dari Header GGUF:")
    for k in kunci_penting:
        if k in reader.fields:
            print(f"    - {k:<32}: {ekstrak_nilai_field(reader.fields[k])}")

    hitung_tensor = Counter()
    hitung_elemen = defaultdict(int)
    hitung_bytes = defaultdict(int)

    for t in reader.tensors:
        tipe = t.tensor_type.name
        hitung_tensor[tipe] += 1
        hitung_elemen[tipe] += int(t.n_elements)
        hitung_bytes[tipe] += int(t.n_bytes)

    print("\n[3] Daftar Tipe Kuantisasi Tensor (Bukti Mixed-Precision):")
    print(
        f"    {'Tipe Kuantisasi':<16} | {'Jml Tensor':>10} | {'Jml Parameter':>15} | "
        f"{'% Params':>8} | {'Ukuran (MiB)':>12}"
    )
    print("    " + "-" * 72)
    for tipe, jml in hitung_tensor.most_common():
        pct = (hitung_elemen[tipe] / total_params * 100.0) if total_params else 0.0
        print(
            f"    {tipe:<16} | {jml:>10} | {hitung_elemen[tipe]:>15,} | "
            f"{pct:>7.1f}% | {hitung_bytes[tipe] / MIB:>12.2f}"
        )

    # Perbandingan rumus `(params * bpw) / 8` terhadap Payload Bobot Murni
    bpw_aktual_bobot = (total_tensor_bytes * 8) / total_params
    print("\n[4] Estimasi Ukuran Bobot `(params * bpw) / 8` vs Payload Bobot Aktual:")
    print(f"    BPW Aktual Model Ini : {bpw_aktual_bobot:.3f} bits/weight\n")

    skenario_bpw = [
        ("Teori 4-bit murni", 4.00),
        ("Teori Q4_0 (blok 32)", 4.50),
        ("Rule-of-thumb Q4_K_M (7B)", 4.85),
        ("BPW Aktual Model Ini", bpw_aktual_bobot),
        ("Teori Q8_0 (blok 32)", 8.50),
    ]
    print(
        f"    {'Skenario Perhitungan':<26} | {'BPW':>6} | {'Estimasi (MiB)':>14} | "
        f"{'Selisih vs Bobot':>17}"
    )
    print("    " + "-" * 72)
    for nama_skenario, bpw in skenario_bpw:
        est_bytes = (total_params * bpw) / 8.0
        selisih_mib = (total_tensor_bytes - est_bytes) / MIB
        print(
            f"    {nama_skenario:<26} | {bpw:>6.2f} | {est_bytes / MIB:>14.2f} | "
            f"{selisih_mib:>+14.2f} MiB"
        )


# ==============================================================================
# BAGIAN 3: BENCHMARK 10 PROMPT TETAP VIA OLLAMA API
# ==============================================================================
def daftar_model_ollama() -> set[str]:
    res = requests.get(f"{HOST}/api/tags", timeout=15)
    res.raise_for_status()
    return {m["name"] for m in res.json().get("models", [])}


def jalankan_10_prompt(model: str) -> dict:
    print(f"\n>>> Menjalankan 10 prompt tetap pada `{model}` (temperature=0, seed=42)...")
    hasil_prompt = []
    total_prefill_tok = 0
    total_prefill_sec = 0.0
    total_decode_tok = 0
    total_decode_sec = 0.0

    for item in PROMPT_BENCHMARK_10:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": 4096,
                "num_predict": 160,
            },
        }
        res = requests.post(f"{HOST}/api/chat", json=payload, timeout=180)
        res.raise_for_status()
        data = res.json()

        p_tok = data.get("prompt_eval_count", 0)
        p_sec = data.get("prompt_eval_duration", 1) / 1e9
        d_tok = data.get("eval_count", 0)
        d_sec = data.get("eval_duration", 1) / 1e9
        teks = data.get("message", {}).get("content", "").strip()

        total_prefill_tok += p_tok
        total_prefill_sec += p_sec
        total_decode_tok += d_tok
        total_decode_sec += d_sec

        hasil_prompt.append(
            {
                "id": item["id"],
                "jawaban": teks,
                "decode_tok_s": d_tok / d_sec if d_sec > 0 else 0.0,
            }
        )
        print(f"    [{item['id']}] selesai ({d_tok / d_sec:.1f} tok/s)")

    vram_mib = 0.0
    gpu_pct = 0.0
    ps_res = requests.get(f"{HOST}/api/ps", timeout=15)
    if ps_res.status_code == 200:
        for m in ps_res.json().get("models", []):
            if m.get("name") == model:
                size_tot = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                vram_mib = size_vram / MIB
                gpu_pct = (size_vram / size_tot * 100.0) if size_tot else 0.0

    return {
        "model": model,
        "avg_prefill_tok_s": total_prefill_tok / total_prefill_sec if total_prefill_sec else 0.0,
        "avg_decode_tok_s": total_decode_tok / total_decode_sec if total_decode_sec else 0.0,
        "vram_mib": vram_mib,
        "gpu_pct": gpu_pct,
        "responses": hasil_prompt,
    }


def bandingkan_kualitas_output(installed_models: set[str]) -> None:
    print("\n" + "=" * 78)
    print("BAGIAN 3: PERBANDINGAN KUALITAS OUTPUT PADA 10 PROMPT TETAP")
    print("=" * 78)

    if MODEL_Q4 not in installed_models:
        print(f"[PERINGATAN] Model `{MODEL_Q4}` belum terpasang di Ollama.")
        return

    hasil_q4 = jalankan_10_prompt(MODEL_Q4)

    if MODEL_Q8 not in installed_models:
        print(f"\n[INFO] `{MODEL_Q8}` belum terpasang (hanya `{MODEL_Q4}` yang diuji).")
        print(
            f"Ringkasan `{MODEL_Q4}` -> Prefill: {hasil_q4['avg_prefill_tok_s']:.1f} tok/s | "
            f"Decode: {hasil_q4['avg_decode_tok_s']:.1f} tok/s | "
            f"VRAM: {hasil_q4['vram_mib']:.1f} MiB ({hasil_q4['gpu_pct']:.0f}% GPU)"
        )
        for r in hasil_q4["responses"]:
            ringkas = " ".join(r["jawaban"].split())
            print(f"  - {r['id']:<21}: {ringkas[:110]}")
        return

    hasil_q8 = jalankan_10_prompt(MODEL_Q8)

    print("\n[A] Ringkasan Performa & Memori:")
    print(f"    {'Metrik':<24} | {MODEL_Q4:<22} | {MODEL_Q8:<22}")
    print("    " + "-" * 74)
    print(f"    {'VRAM Terpakai':<24} | {hasil_q4['vram_mib']:>18.1f} MiB | {hasil_q8['vram_mib']:>18.1f} MiB")
    print(f"    {'Offload GPU':<24} | {hasil_q4['gpu_pct']:>21.1f}% | {hasil_q8['gpu_pct']:>21.1f}%")
    print(f"    {'Rata-rata Prefill':<24} | {hasil_q4['avg_prefill_tok_s']:>16.1f} tok/s | {hasil_q8['avg_prefill_tok_s']:>16.1f} tok/s")
    print(f"    {'Rata-rata Decode':<24} | {hasil_q4['avg_decode_tok_s']:>16.1f} tok/s | {hasil_q8['avg_decode_tok_s']:>16.1f} tok/s")

    print("\n[B] Perbandingan Jawaban per Prompt (Q4_K_M vs Q8_0):")
    identik = 0
    for r4, r8 in zip(hasil_q4["responses"], hasil_q8["responses"]):
        sama = r4["jawaban"] == r8["jawaban"]
        if sama:
            identik += 1
        status = "IDENTIK" if sama else "BERBEDA"
        print(f"\n--- {r4['id']} [{status}] ---")
        print(f"  Q4_K_M: {' '.join(r4['jawaban'].split())}")
        print(f"  Q8_0  : {' '.join(r8['jawaban'].split())}")

    print(f"\nTotal jawaban 100% identik: {identik}/10 prompt.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hari ke-5: Simulasi Kuantisasi Blok & Inspeksi GGUF")
    default_models_dir = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models_dir,
        help="Direktori root models Ollama (default: env OLLAMA_MODELS atau ~/.ollama/models)",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Lewati pengujian 10 prompt ke Ollama API (hanya jalankan Bagian 1 & 2)",
    )
    args = parser.parse_args()

    # Bagian 1: Simulasi matematis kuantisasi blok (NumPy)
    demo_kuantisasi_blok()

    # Bagian 2: Inspeksi file GGUF asli
    blob_q4 = cari_blob_dari_ollama(MODEL_Q4) or cari_blob_di_folder(args.models_dir, "qwen35")
    if blob_q4 is None:
        print(
            "\n[PERINGATAN] File blob GGUF tidak ditemukan. Pastikan Ollama aktif atau arahkan "
            "`--models-dir` ke folder models Ollama kamu."
        )
        return

    inspeksi_gguf(blob_q4, MODEL_Q4)

    blob_q8 = cari_blob_dari_ollama(MODEL_Q8)
    if blob_q8 and blob_q8 != blob_q4:
        inspeksi_gguf(blob_q8, MODEL_Q8)

    # Bagian 3: Benchmark 10 prompt tetap
    if args.skip_ollama:
        print("\n[INFO] Flag `--skip-ollama` aktif; melewati pengujian 10 prompt.")
        return

    try:
        installed = daftar_model_ollama()
        bandingkan_kualitas_output(installed)
    except requests.RequestException as err:
        print(f"\n[INFO] Ollama API tidak terjangkau ({err}); melewati uji 10 prompt.")


if __name__ == "__main__":
    main()