import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import config
import salary_parser
from feature_extractor import normalize_text
from preprocess_dataset import build_proxy_dataset, build_rf_ready_dataset
from scrapper import JobRecord, JobStreetScraper


TARGET_PER_CLASS = 771

TARGET_LABELS = [
    "0 sampai < 1 juta",
    "1 sampai < 2 juta",
]

SALARY_BINS_NEW = [
    0,
    1_000_000,
    2_000_000,
    4_000_000,
    6_000_000,
    8_000_000,
    10_000_000,
    float("inf"),
]

SALARY_LABELS_NEW = [
    "0 sampai < 1 juta",
    "1 sampai < 2 juta",
    "2 sampai < 4 juta",
    "4 sampai < 6 juta",
    "6 sampai < 8 juta",
    "8 sampai < 10 juta",
    ">= 10 juta",
]

BATCH_NAME = "09_low_salary_only"

LOW_BATCH_RAW_PATH = config.BATCH_OUTPUT_DIR / "jobstreet_raw_batch_09_low_salary_only.csv"
LOW_BATCH_PROXY_PATH = config.BATCH_OUTPUT_DIR / "graduate_salary_proxy_batch_09_low_salary_only.csv"
LOW_BATCH_RF_READY_PATH = config.BATCH_OUTPUT_DIR / "graduate_salary_rf_ready_batch_09_low_salary_only.csv"

LOW_CHECKPOINT_PATH = config.CHECKPOINT_DIR / "jobstreet_raw_checkpoint_09_low_salary_only.csv"

LOW_ADDED_RF_READY_PATH = config.OUTPUT_DIR / "graduate_salary_rf_ready_low_added.csv"

MAX_PAGE_PER_KEYWORD_LOW = 40
MAX_EMPTY_PAGE_STREAK = 5
SAVE_EVERY_ACCEPTED_RECORD = 1

LOW_SALARY_KEYWORDS = [
    "magang berbayar",
    "magang uang saku",
    "paid internship",
    "internship allowance",
    "internship mahasiswa",
    "magang mahasiswa",
    "internship fresh graduate",
    "part time mahasiswa",
    "part time admin",
    "part time data entry",
    "data entry part time",
    "freelance data entry",
    "freelance admin",
    "freelance mahasiswa",
    "admin freelance",
    "admin part time",
    "customer service part time",
    "sales part time",
    "marketing part time",
    "content creator part time",
    "social media intern",
    "admin intern",
    "finance intern",
    "accounting intern",
    "hr intern",
    "marketing intern",
    "data analyst intern",
    "it support intern",
    "web developer intern",
    "programmer intern",
    "junior admin fresh graduate",
    "admin entry level",
    "staff administrasi entry level",
    "operator part time",
    "cashier part time",
    "kasir part time",
    "barista part time",
    "waiter part time",
    "crew part time",
    "crew outlet part time",
    "spg part time",
    "sales promotor part time",
    "gaji 500 ribu",
    "gaji 700 ribu",
    "gaji 800 ribu",
    "gaji 900 ribu",
    "gaji 1 juta",
    "gaji 1.5 juta",
    "gaji 1500000",
    "gaji 1800000",
    "uang saku 500",
    "uang saku 700",
    "uang saku 1000000",
    "honor 1 juta",
]


def force_salary_range_config():
    config.SALARY_BINS = SALARY_BINS_NEW
    config.SALARY_LABELS = SALARY_LABELS_NEW

    salary_parser.SALARY_BINS = SALARY_BINS_NEW
    salary_parser.SALARY_LABELS = SALARY_LABELS_NEW


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        return df
    except Exception as exc:
        print(f"Gagal membaca CSV: {path}")
        print(f"Error: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def deduplicate_by_url(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "job_url" in df.columns:
        df = df.dropna(subset=["job_url"])
        df = df.drop_duplicates(subset=["job_url"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    return df.reset_index(drop=True)


def load_seen_urls() -> set:
    seen_urls = set()

    paths = [
        config.RAW_OUTPUT_FILE,
        LOW_BATCH_RAW_PATH,
        LOW_CHECKPOINT_PATH,
    ]

    for batch_file in config.BATCH_OUTPUT_DIR.glob("jobstreet_raw_batch_*.csv"):
        paths.append(batch_file)

    for path in paths:
        df = safe_read_csv(path)
        if not df.empty and "job_url" in df.columns:
            seen_urls.update(df["job_url"].dropna().astype(str).tolist())

    return seen_urls


def save_session_checkpoint(records: list[dict]):
    if not records:
        return

    checkpoint_df = pd.DataFrame(records)
    checkpoint_df = deduplicate_by_url(checkpoint_df)

    LOW_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_df.to_csv(LOW_CHECKPOINT_PATH, index=False, encoding="utf-8-sig")

    print(f"Checkpoint tersimpan: {LOW_CHECKPOINT_PATH}")
    print(f"Jumlah data sesi di checkpoint: {len(checkpoint_df)}")


def merge_low_batch_raw(records: list[dict]) -> pd.DataFrame:
    pieces = []

    old_batch_df = safe_read_csv(LOW_BATCH_RAW_PATH)
    checkpoint_df = safe_read_csv(LOW_CHECKPOINT_PATH)
    session_df = pd.DataFrame(records)

    for df in [old_batch_df, checkpoint_df, session_df]:
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    low_batch_df = deduplicate_by_url(pd.concat(pieces, ignore_index=True))

    LOW_BATCH_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    low_batch_df.to_csv(LOW_BATCH_RAW_PATH, index=False, encoding="utf-8-sig")

    return low_batch_df


def build_low_batch_datasets():
    if not LOW_BATCH_RAW_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    low_proxy_df = build_proxy_dataset(LOW_BATCH_RAW_PATH, LOW_BATCH_PROXY_PATH)
    low_rf_df = build_rf_ready_dataset(LOW_BATCH_PROXY_PATH, LOW_BATCH_RF_READY_PATH)

    if not low_rf_df.empty:
        low_rf_df = low_rf_df[low_rf_df["salary_range_label"].isin(TARGET_LABELS)].copy()
        low_rf_df.to_csv(LOW_BATCH_RF_READY_PATH, index=False, encoding="utf-8-sig")

    return low_proxy_df, low_rf_df


def load_base_and_low_rf_ready() -> pd.DataFrame:
    base_path = config.RF_READY_DATASET_FILE
    print(f"\nDataset dasar yang dibaca script: {base_path}")
    print(f"Apakah file ada: {base_path.exists()}")

    base_df = safe_read_csv(base_path)

    if not base_df.empty:
        print(f"Ukuran dataset dasar: {base_df.shape}")
        if "salary_range_label" in base_df.columns:
            print("Distribusi label dataset dasar:")
            print(base_df["salary_range_label"].value_counts())
            print("Unique label:")
            print(base_df["salary_range_label"].unique())
    else:
        print("Dataset dasar kosong atau gagal dibaca.")

    if LOW_BATCH_RAW_PATH.exists():
        build_low_batch_datasets()

    low_df = safe_read_csv(LOW_BATCH_RF_READY_PATH)

    pieces = []
    for df in [base_df, low_df]:
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    combined_df = pd.concat(pieces, ignore_index=True)
    combined_df = deduplicate_by_url(combined_df)

    return combined_df


def get_current_counts(df: pd.DataFrame) -> dict:
    counts = {label: 0 for label in TARGET_LABELS}

    if df.empty or "salary_range_label" not in df.columns:
        return counts

    value_counts = df["salary_range_label"].value_counts().to_dict()

    for label in TARGET_LABELS:
        counts[label] = int(value_counts.get(label, 0))

    return counts


def target_done(counts: dict) -> bool:
    return all(counts[label] >= TARGET_PER_CLASS for label in TARGET_LABELS)


def print_counts(counts: dict):
    print("\nJumlah dua kelas target:")
    for label in TARGET_LABELS:
        kurang = max(0, TARGET_PER_CLASS - counts.get(label, 0))
        print(f"{label}: {counts.get(label, 0)}/{TARGET_PER_CLASS}, kurang {kurang}")


def extract_candidate_salary_label(card: dict, detail_text: str):
    salary_source_text = normalize_text(
        card.get("salary_text", ""),
        card.get("job_title", ""),
        card.get("card_text", ""),
        detail_text,
    )

    salary_min, salary_max, salary_avg = salary_parser.parse_salary(salary_source_text)
    salary_label = salary_parser.salary_to_range_label(salary_avg)

    return salary_label, salary_avg


def save_low_added_rf_ready():
    combined_df = load_base_and_low_rf_ready()

    if combined_df.empty:
        print("Dataset gabungan kosong. Belum ada file final yang dibuat.")
        return pd.DataFrame()

    if "salary_range_label" not in combined_df.columns:
        print("Kolom salary_range_label tidak ada.")
        return pd.DataFrame()

    final_parts = []

    for label in SALARY_LABELS_NEW:
        label_df = combined_df[combined_df["salary_range_label"] == label].copy()

        if label in TARGET_LABELS and len(label_df) > TARGET_PER_CLASS:
            label_df = label_df.head(TARGET_PER_CLASS)

        final_parts.append(label_df)

    final_df = pd.concat(final_parts, ignore_index=True)
    final_df = deduplicate_by_url(final_df)

    final_df.to_csv(LOW_ADDED_RF_READY_PATH, index=False, encoding="utf-8-sig")

    print("\nFile dataset lama tidak ditimpa:")
    print(config.RF_READY_DATASET_FILE)

    print("\nFile dataset baru hasil tambahan dua range tersimpan di:")
    print(LOW_ADDED_RF_READY_PATH)

    print("\nDistribusi file baru:")
    print(final_df["salary_range_label"].value_counts())

    return final_df


def setup_runtime_config():
    force_salary_range_config()

    config.ACTIVE_BATCH_NAME = BATCH_NAME
    config.SEARCH_KEYWORDS = LOW_SALARY_KEYWORDS
    config.SESSION_CHECKPOINT_FILE = LOW_CHECKPOINT_PATH

    config.MAX_PAGE_PER_KEYWORD = max(int(config.MAX_PAGE_PER_KEYWORD), MAX_PAGE_PER_KEYWORD_LOW)
    config.MAX_JOBS_PER_KEYWORD = max(int(config.MAX_JOBS_PER_KEYWORD), 500)

    config.FETCH_JOB_DETAIL = True


def run():
    setup_runtime_config()

    print("=" * 80)
    print("Scraping khusus dua range gaji rendah")
    print("Target kelas:")
    for label in TARGET_LABELS:
        print(f"{label}: {TARGET_PER_CLASS}")
    print(f"Raw batch: {LOW_BATCH_RAW_PATH}")
    print(f"Checkpoint: {LOW_CHECKPOINT_PATH}")
    print("=" * 80)

    current_df = load_base_and_low_rf_ready()
    counts = get_current_counts(current_df)
    print_counts(counts)

    if target_done(counts):
        print("\nDua kelas target sudah mencapai 771. Scraping tidak dijalankan.")
        save_low_added_rf_ready()
        return

    seen_urls = load_seen_urls()
    print(f"\nJumlah URL lama yang akan dilewati: {len(seen_urls)}")

    accepted_records = []

    try:
        with JobStreetScraper(headless=config.HEADLESS) as scraper:
            for keyword_index, keyword in enumerate(LOW_SALARY_KEYWORDS, start=1):
                if target_done(counts):
                    print("\nDua kelas target sudah penuh. Scraping dihentikan.")
                    break

                print("\n" + "=" * 80)
                print(f"[{keyword_index}/{len(LOW_SALARY_KEYWORDS)}] Keyword: {keyword}")
                print("=" * 80)

                empty_page_streak = 0

                for page in range(1, MAX_PAGE_PER_KEYWORD_LOW + 1):
                    if target_done(counts):
                        break

                    search_url = scraper._search_url(keyword, page)
                    print(f"Open page {page}: {search_url}")

                    loaded = scraper._load_page(search_url, config.PAGE_LOAD_DELAY_SECONDS)
                    if not loaded:
                        empty_page_streak += 1
                        if empty_page_streak >= MAX_EMPTY_PAGE_STREAK:
                            print("Terlalu banyak halaman gagal. Pindah keyword.")
                            break
                        continue

                    scraper._scroll_page()
                    html = scraper._get_page_source()

                    if not html:
                        empty_page_streak += 1
                        continue

                    cards = scraper._parse_cards(html, keyword)

                    if not cards:
                        empty_page_streak += 1
                        print("Tidak ada kartu lowongan terbaca.")
                        if empty_page_streak >= MAX_EMPTY_PAGE_STREAK:
                            print("Beberapa halaman kosong berturut turut. Pindah keyword.")
                            break
                        continue

                    accepted_on_page = 0

                    for card in cards:
                        job_url = str(card.get("job_url", "")).strip()

                        if not job_url:
                            continue

                        if job_url in seen_urls:
                            continue

                        detail_text = scraper._fetch_detail_text(job_url)

                        salary_label, salary_avg = extract_candidate_salary_label(card, detail_text)

                        seen_urls.add(job_url)

                        if salary_label not in TARGET_LABELS:
                            continue

                        if counts[salary_label] >= TARGET_PER_CLASS:
                            print(f"Skip kelas sudah penuh: {salary_label}")
                            continue

                        description = scraper._clean_text(
                            f"{card.get('card_text', '')} {detail_text}"
                        )

                        record = JobRecord(
                            source_keyword=keyword,
                            job_title=card.get("job_title", ""),
                            company=card.get("company", ""),
                            location=card.get("location", ""),
                            salary_text=card.get("salary_text", ""),
                            job_description=description,
                            job_url=job_url,
                            scraped_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                        accepted_records.append(asdict(record))
                        counts[salary_label] += 1
                        accepted_on_page += 1

                        print(
                            f"Accepted | {salary_label} | "
                            f"{counts[salary_label]}/{TARGET_PER_CLASS} | "
                            f"{record.job_title[:90]}"
                        )

                        if len(accepted_records) % SAVE_EVERY_ACCEPTED_RECORD == 0:
                            save_session_checkpoint(accepted_records)
                            merge_low_batch_raw(accepted_records)
                            build_low_batch_datasets()
                            save_low_added_rf_ready()

                        if target_done(counts):
                            break

                    if accepted_on_page == 0:
                        empty_page_streak += 1
                        print("Halaman terbaca, tetapi tidak ada data baru untuk dua range target.")
                        if empty_page_streak >= MAX_EMPTY_PAGE_STREAK:
                            print("Beberapa halaman berturut turut tidak menghasilkan data target. Pindah keyword.")
                            break
                    else:
                        empty_page_streak = 0
                        print_counts(counts)

    except KeyboardInterrupt:
        print("\nScraping dihentikan manual. Data yang sudah diterima tetap disimpan.")

    except Exception as exc:
        print(f"\nScraping berhenti karena error: {type(exc).__name__}: {exc}")
        print("Data yang sudah diterima tetap disimpan.")

    finally:
        print("\nFinalisasi...")

        save_session_checkpoint(accepted_records)
        low_batch_df = merge_low_batch_raw(accepted_records)

        if not low_batch_df.empty:
            build_low_batch_datasets()

        save_low_added_rf_ready()

        print("\nSelesai.")


if __name__ == "__main__":
    run()