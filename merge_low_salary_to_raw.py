"""
Merge data dari jobstreet_raw_batch_09_low_salary_only.csv ke jobstreet_raw.csv

Script ini menggabungkan data low salary batch yang sudah dikumpulkan ke master file raw.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

import config


def safe_write_csv(df: pd.DataFrame, output_path: Path):
    """Simpan CSV dengan fallback jika file terkunci."""
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✓ Berhasil simpan: {output_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"⚠ File utama terkunci, disimpan sebagai: {fallback_path}")


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Baca CSV dengan penanganan error."""
    if not path.exists():
        print(f"⚠ File tidak ditemukan: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        print(f"✓ Dibaca: {path.name} | {len(df)} baris")
        return df
    except Exception as e:
        print(f"✗ Gagal baca {path.name}: {e}")
        return pd.DataFrame()


def merge_low_salary_to_raw():
    """Merge low salary batch ke jobstreet_raw.csv."""
    
    print("=" * 80)
    print("Merge Low Salary Data ke Master Raw File")
    print("=" * 80)

    low_batch_path = config.BATCH_OUTPUT_DIR / "jobstreet_raw_batch_09_low_salary_only.csv"
    raw_output_path = config.RAW_OUTPUT_FILE

    # Baca kedua file
    print("\n📂 Membaca file...")
    low_batch_df = safe_read_csv(low_batch_path)
    raw_df = safe_read_csv(raw_output_path)

    if low_batch_df.empty:
        print("✗ File low salary batch kosong atau gagal dibaca. Merge dibatalkan.")
        return

    # Gabungkan
    print("\n🔗 Menggabungkan data...")
    pieces = []
    
    if not raw_df.empty:
        pieces.append(raw_df)
    pieces.append(low_batch_df)
    
    merged_df = pd.concat(pieces, ignore_index=True)
    print(f"Total sebelum deduplikasi: {len(merged_df)} baris")

    # Deduplikasi berdasarkan job_url
    if "job_url" in merged_df.columns:
        merged_df = merged_df.dropna(subset=["job_url"])
        merged_df = merged_df.drop_duplicates(subset=["job_url"], keep="first")
    else:
        merged_df = merged_df.drop_duplicates(keep="first")

    print(f"Total setelah deduplikasi: {len(merged_df)} baris")

    # Simpan
    print("\n💾 Menyimpan hasil merge...")
    safe_write_csv(merged_df, raw_output_path)

    print("\n📊 Statistik:")
    print(f"  Data dari low salary batch: {len(low_batch_df)}")
    print(f"  Data lama di raw file: {len(raw_df) if not raw_df.empty else 0}")
    print(f"  Total merged: {len(merged_df)}")
    print(f"  Duplikat yang dihilangkan: {len(raw_df) + len(low_batch_df) - len(merged_df) if not raw_df.empty else len(low_batch_df) - len(merged_df)}")

    print("\n✅ Merge selesai!")
    print(f"File master raw berhasil diupdate: {raw_output_path}")

    return merged_df


if __name__ == "__main__":
    merge_low_salary_to_raw()
