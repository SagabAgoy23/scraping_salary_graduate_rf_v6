import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import config
from preprocess_dataset import build_proxy_dataset, build_rf_ready_dataset
from scrapper import JobStreetScraper


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "job_url" in df.columns:
        df = df.dropna(subset=["job_url"])
        df = df.drop_duplicates(subset=["job_url"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")
    return df.reset_index(drop=True)


def _batch_key_from_arg(value: str) -> str:
    batches = list(config.SEARCH_KEYWORD_BATCHES.keys())
    value = str(value).strip()

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(batches):
            return batches[idx - 1]
        raise ValueError(f"Nomor batch {idx} tidak tersedia. Pilih 1 sampai {len(batches)}.")

    if value in config.SEARCH_KEYWORD_BATCHES:
        return value

    # Boleh pakai bentuk pendek, misalnya: it, data, finance, admin.
    matches = [name for name in batches if value.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Batch '{value}' ambigu. Kandidat: {', '.join(matches)}")

    raise ValueError(f"Batch '{value}' tidak ditemukan.")


def _sanitize_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _set_active_batch(batch_name: str):
    config.ACTIVE_BATCH_NAME = batch_name
    config.SEARCH_KEYWORDS = config.SEARCH_KEYWORD_BATCHES[batch_name]
    safe = _sanitize_name(batch_name)
    config.SESSION_CHECKPOINT_FILE = config.CHECKPOINT_DIR / f"jobstreet_raw_checkpoint_{safe}.csv"
    return safe


def _batch_paths(batch_name: str):
    safe = _sanitize_name(batch_name)
    raw_path = config.BATCH_OUTPUT_DIR / f"jobstreet_raw_batch_{safe}.csv"
    proxy_path = config.BATCH_OUTPUT_DIR / f"graduate_salary_proxy_batch_{safe}.csv"
    rf_path = config.BATCH_OUTPUT_DIR / f"graduate_salary_rf_ready_batch_{safe}.csv"
    return raw_path, proxy_path, rf_path


def list_batches():
    print("Daftar batch keyword:")
    for idx, (name, keywords) in enumerate(config.SEARCH_KEYWORD_BATCHES.items(), start=1):
        print(f"{idx}. {name} = {len(keywords)} keyword")
        for keyword in keywords:
            print(f"   - {keyword}")
    print("\nContoh menjalankan satu batch:")
    print("python main.py --batch 1")
    print("python main.py --batch it")
    print("python main.py --batch finance")


def _merge_batch_result(batch_name: str, session_records: list[dict]) -> pd.DataFrame:
    batch_raw_path, _, _ = _batch_paths(batch_name)

    pieces = []
    old_batch_df = _safe_read_csv(batch_raw_path)
    checkpoint_df = _safe_read_csv(config.SESSION_CHECKPOINT_FILE)
    session_df = pd.DataFrame(session_records)

    for df in [old_batch_df, checkpoint_df, session_df]:
        if not df.empty:
            pieces.append(df)

    if not pieces:
        batch_df = pd.DataFrame()
    else:
        batch_df = _deduplicate(pd.concat(pieces, ignore_index=True))

    if not batch_df.empty:
        batch_raw_path.parent.mkdir(exist_ok=True)
        batch_df.to_csv(batch_raw_path, index=False, encoding="utf-8-sig")

    return batch_df


def _build_batch_datasets(batch_name: str):
    batch_raw_path, batch_proxy_path, batch_rf_path = _batch_paths(batch_name)
    if not batch_raw_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    proxy_df = build_proxy_dataset(batch_raw_path, batch_proxy_path)
    rf_df = build_rf_ready_dataset(batch_proxy_path, batch_rf_path)
    return proxy_df, rf_df


def _build_combined_outputs():
    pieces = []

    # Sertakan combined raw lama agar data dari versi sebelumnya tidak hilang jika user menyalin folder outputs.
    if config.APPEND_EXISTING_RAW and config.RAW_OUTPUT_FILE.exists():
        old_combined = _safe_read_csv(config.RAW_OUTPUT_FILE)
        if not old_combined.empty:
            pieces.append(old_combined)

    for path in sorted(config.BATCH_OUTPUT_DIR.glob("jobstreet_raw_batch_*.csv")):
        df = _safe_read_csv(path)
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw_df = _deduplicate(pd.concat(pieces, ignore_index=True))
    raw_df.to_csv(config.RAW_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    proxy_df = build_proxy_dataset(config.RAW_OUTPUT_FILE, config.PROXY_DATASET_FILE)
    rf_df = build_rf_ready_dataset(config.PROXY_DATASET_FILE, config.RF_READY_DATASET_FILE)
    return raw_df, proxy_df, rf_df


def run_batch(batch_name: str):
    safe = _set_active_batch(batch_name)
    batch_raw_path, batch_proxy_path, batch_rf_path = _batch_paths(batch_name)

    print("=" * 80)
    print(f"Menjalankan batch: {batch_name}")
    print(f"Jumlah keyword dalam batch: {len(config.SEARCH_KEYWORDS)}")
    print(f"Maks halaman per keyword: {config.MAX_PAGE_PER_KEYWORD}")
    print(f"Maks lowongan per keyword: {config.MAX_JOBS_PER_KEYWORD}")
    print(f"Checkpoint batch: {config.SESSION_CHECKPOINT_FILE}")
    print(f"CSV raw batch: {batch_raw_path}")
    print("=" * 80)

    records = []
    crashed = False
    try:
        with JobStreetScraper(headless=config.HEADLESS) as scraper:
            records = scraper.scrape()
    except KeyboardInterrupt:
        crashed = True
        print("\nScraping dihentikan manual. Data yang sudah masuk checkpoint tetap akan disimpan.")
    except Exception as exc:
        crashed = True
        print(f"\nScraping batch berhenti karena error: {type(exc).__name__}: {exc}")
        print("Data yang sudah masuk checkpoint tetap akan disimpan ke CSV batch.")

    batch_df = _merge_batch_result(batch_name, records)
    print(f"Raw batch tersimpan: {batch_raw_path}")
    print(f"Jumlah raw batch setelah deduplikasi: {len(batch_df)}")

    if not batch_df.empty:
        proxy_df, rf_df = _build_batch_datasets(batch_name)
        print(f"Proxy batch tersimpan: {batch_proxy_path}")
        print(f"RF-ready batch tersimpan: {batch_rf_path}")
        print(f"Jumlah proxy batch: {len(proxy_df)}")
        print(f"Jumlah RF-ready batch: {len(rf_df)}")
    else:
        print("Batch ini belum menghasilkan data.")

    raw_df, proxy_df, rf_df = _build_combined_outputs()
    print("-" * 80)
    print(f"Raw gabungan tersimpan: {config.RAW_OUTPUT_FILE}")
    print(f"Proxy gabungan tersimpan: {config.PROXY_DATASET_FILE}")
    print(f"RF-ready gabungan tersimpan: {config.RF_READY_DATASET_FILE}")
    print(f"Jumlah raw gabungan: {len(raw_df)}")
    print(f"Jumlah proxy gabungan: {len(proxy_df)}")
    print(f"Jumlah RF-ready gabungan: {len(rf_df)}")

    if len(rf_df) > 0 and "salary_range_label" in rf_df.columns:
        print("Distribusi target gabungan:")
        print(rf_df["salary_range_label"].value_counts())

    if crashed:
        print("\nBatch belum selesai penuh, tetapi data sementara sudah disimpan. Jalankan ulang batch yang sama untuk lanjut tanpa mulai dari nol.")
    else:
        batches = list(config.SEARCH_KEYWORD_BATCHES.keys())
        current_idx = batches.index(batch_name) + 1
        if current_idx < len(batches):
            print(f"\nLanjutkan batch berikutnya dengan: python main.py --batch {current_idx + 1}")
        else:
            print("\nSemua batch bernomor sudah sampai batch terakhir. Jalankan training dengan: python train_random_forest.py")


def main():
    parser = argparse.ArgumentParser(description="Scraper JobStreet berbasis batch untuk dataset klasifikasi range gaji awal lulusan.")
    parser.add_argument("--list-batches", action="store_true", help="Tampilkan daftar batch keyword.")
    parser.add_argument("--batch", default="1", help="Nomor atau nama batch. Contoh: 1, it, finance, admin.")
    parser.add_argument("--all-batches", action="store_true", help="Jalankan semua batch berurutan. Tetap menyimpan CSV per batch.")
    args = parser.parse_args()

    if args.list_batches:
        list_batches()
        return

    if args.all_batches:
        for name in config.SEARCH_KEYWORD_BATCHES.keys():
            run_batch(name)
        print("\nSemua batch selesai. Jalankan: python train_random_forest.py")
        return

    try:
        batch_name = _batch_key_from_arg(args.batch)
    except ValueError as exc:
        print(str(exc))
        print("Gunakan python main.py --list-batches untuk melihat pilihan batch.")
        sys.exit(1)

    run_batch(batch_name)


if __name__ == "__main__":
    main()
