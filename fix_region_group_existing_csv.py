from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "graduate_salary_rf_ready.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "graduate_salary_rf_ready_fixed.csv"

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text

def classify_region(location):
    location_norm = normalize_text(location)

    jabodetabek_keywords = [
        "jakarta", "bogor", "depok", "tangerang", "bekasi",
        "kelapa gading", "cempaka putih", "cilandak", "pancoran",
        "kalideres", "penjaringan", "cakung", "cengkareng",
        "kebon jeruk", "kemayoran", "grogol petamburan",
        "kebayoran lama", "ciputat", "serpong", "karawaci",
        "cileungsi", "gunung putri", "cibitung", "cikarang"
    ]

    jawa_barat_keywords = [
        "bandung", "karawang", "purwakarta", "cirebon",
        "majalengka", "sukabumi", "padalarang", "margaasih",
        "cileunyi"
    ]

    jawa_tengah_diy_keywords = [
        "semarang", "yogyakarta", "di yogyakarta", "sleman",
        "bantul", "surakarta", "kendal", "jepara", "batang",
        "sukoharjo"
    ]

    jawa_timur_keywords = [
        "surabaya", "sidoarjo", "mojokerto", "jombang",
        "gresik", "malang", "kediri", "bojonegoro",
        "ngoro", "gedangan", "tenggilis mejoyo", "sambikerep",
        "waru", "jawa timur"
    ]

    sumatera_keywords = [
        "medan", "batam", "pekanbaru", "palembang",
        "lampung", "jambi", "bengkulu", "prabumulih",
        "deli serdang", "sumatera selatan"
    ]

    kalimantan_keywords = [
        "kalimantan",
        "balikpapan", "samarinda", "bontang", "tarakan",
        "banjarmasin", "banjarbaru", "martapura",
        "palangkaraya", "palangka raya",
        "pontianak", "pontianak kota", "singkawang",
        "pangkalan bun", "sampit"
    ]

    sulawesi_keywords = [
        "makassar", "sulawesi", "morowali", "tomohon"
    ]

    if any(keyword in location_norm for keyword in jabodetabek_keywords):
        return "jabodetabek"

    if any(keyword in location_norm for keyword in jawa_barat_keywords):
        return "jawa barat"

    if any(keyword in location_norm for keyword in jawa_tengah_diy_keywords):
        return "jawa tengah diy"

    if any(keyword in location_norm for keyword in jawa_timur_keywords):
        return "jawa timur"

    if any(keyword in location_norm for keyword in sumatera_keywords):
        return "sumatera"

    if any(keyword in location_norm for keyword in kalimantan_keywords):
        return "kalimantan"

    if any(keyword in location_norm for keyword in sulawesi_keywords):
        return "sulawesi"

    return "lainnya"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "location" not in df.columns:
        raise ValueError("Kolom 'location' tidak ditemukan di dataset.")

    print("Jumlah region sebelum perbaikan:")
    print(df["region_group"].value_counts() if "region_group" in df.columns else "Kolom region_group belum ada")

    df["region_group"] = df["location"].apply(classify_region)

    print("\nJumlah region setelah perbaikan:")
    print(df["region_group"].value_counts())

    print("\nData Kalimantan setelah perbaikan:")
    kalimantan_df = df[df["region_group"] == "kalimantan"]

    kolom_tampil = [
        col for col in ["job_title", "company", "location", "region_group"]
        if col in df.columns
    ]

    print(kalimantan_df[kolom_tampil].to_string(index=False))

    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\nFile berhasil disimpan sebagai: {OUTPUT_FILE}")
    except PermissionError:
        print("\nFile gagal disimpan karena sedang terbuka.")
        print("Tutup Excel atau preview CSV, lalu jalankan ulang script ini.")

if __name__ == "__main__":
    main()