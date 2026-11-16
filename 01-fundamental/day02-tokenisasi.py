from transformers import AutoTokenizer

# Hanya mengunduh file tokenizer (beberapa MB), bukan bobot model penuh
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

teks_en = (
    "Soil moisture sensors measure volumetric water content in agricultural fields "
    "to optimize irrigation scheduling and prevent crop stress."
)

teks_id = (
    "Sensor kelembapan tanah mengukur kadar air volumetrik di lahan pertanian "
    "untuk mengoptimalkan penjadwalan irigasi dan mencegah stres pada tanaman."
)

def analisis_teks(label, teks):
    daftar_kata = teks.split()
    token_ids = tokenizer.encode(teks, add_special_tokens=False)
    
    # Mengubah tiap ID token kembali ke teks agar terlihat cara pemotongannya
    potongan_token = [tokenizer.decode([tid]) for tid in token_ids]
    rasio = len(token_ids) / len(daftar_kata)

    print(f"=== {label} ===")
    print(f"Jumlah kata      : {len(daftar_kata)}")
    print(f"Jumlah token     : {len(token_ids)}")
    print(f"Rasio token/kata : {rasio:.2f}")
    print(f"Potongan token   : {potongan_token}\n")

analisis_teks("Bahasa Inggris", teks_en)
analisis_teks("Bahasa Indonesia", teks_id)