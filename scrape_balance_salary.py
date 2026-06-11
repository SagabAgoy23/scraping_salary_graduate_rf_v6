import time
from pathlib import Path
from dataclasses import asdict

import pandas as pd

import config
from feature_extractor import normalize_text
from preprocess_dataset import build_proxy_dataset, build_rf_ready_dataset
from salary_parser import parse_salary, salary_to_range_label
from scrapper import JobStreetScraper, JobRecord


TARGET_PER_CLASS = 771

TARGET_LABELS = [
    "0 sampai < 1 juta",
    "1 sampai < 2 juta",
    "2 sampai < 4 juta",
    "4 sampai < 6 juta",
    "6 sampai < 8 juta",
    "8 sampai < 10 juta",
    ">= 10 juta",
]

BATCH_NAME = "09_balance_salary"
SAFE_BATCH_NAME = "09_balance_salary"

BATCH_RAW_PATH = config.BATCH_OUTPUT_DIR / f"jobstreet_raw_batch_{SAFE_BATCH_NAME}.csv"
BATCH_PROXY_PATH = config.BATCH_OUTPUT_DIR / f"graduate_salary_proxy_batch_{SAFE_BATCH_NAME}.csv"
BATCH_RF_READY_PATH = config.BATCH_OUTPUT_DIR / f"graduate_salary_rf_ready_batch_{SAFE_BATCH_NAME}.csv"

CHECKPOINT_PATH = config.CHECKPOINT_DIR / f"jobstreet_raw_checkpoint_{SAFE_BATCH_NAME}.csv"

BALANCED_RF_READY_PATH = config.OUTPUT_DIR / f"graduate_salary_rf_balanced_{TARGET_PER_CLASS}.csv"
FULL_RF_READY_BACKUP_PATH = config.OUTPUT_DIR / "graduate_salary_rf_ready_full_before_balancing.csv"

SAVE_EVERY_ACCEPTED_RECORD = 1

BALANCE_KEYWORDS = [
    "magang berbayar",
    "paid internship",
    "internship allowance",
    "magang uang saku",
    "internship mahasiswa",
    "magang mahasiswa",
    "part time mahasiswa",
    "part time admin",
    "part time customer service",
    "part time sales",
    "freelance admin",
    "freelance data entry",
    "data entry part time",
    "admin entry level",
    "admin staff fresh graduate",
    "staff administrasi fresh graduate",
    "customer service fresh graduate",
    "operator produksi fresh graduate",
    "sales promotor fresh graduate",
    "sales executive fresh graduate",
    "marketing staff fresh graduate",
    "finance staff fresh graduate",
    "accounting staff fresh graduate",
    "tax staff fresh graduate",
    "audit staff fresh graduate",
    "it support fresh graduate",
    "junior programmer fresh graduate",
    "junior web developer fresh graduate",
    "junior software engineer fresh graduate",
    "frontend developer fresh graduate",
    "backend developer fresh graduate",
    "mobile developer fresh graduate",
    "qa tester fresh graduate",
    "data analyst fresh graduate",
    "junior data analyst fresh graduate",
    "business analyst fresh graduate",
    "business intelligence fresh graduate",
    "management trainee fresh graduate",
    "graduate trainee",
    "officer development program",
    "management development program",
    "banking staff fresh graduate",
    "teller fresh graduate",
    "relationship officer fresh graduate",
    "account executive fresh graduate",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def deduplicate_raw(df: pd.DataFrame) -> pd.DataFrame:
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
        BATCH_RAW_PATH,
        CHECKPOINT_PATH,
    ]

    for path in paths:
        df = safe_read_csv(path)
        if not df.empty and "job_url" in df.columns:
            seen_urls.update(df["job_url"].dropna().astype(str).tolist())

    return seen_urls


def rebuild_current_datasets() -> pd.DataFrame:
    if config.RAW_OUTPUT_FILE.exists():
        build_proxy_dataset(config.RAW_OUTPUT_FILE, config.PROXY_DATASET_FILE)
        build_rf_ready_dataset(config.PROXY_DATASET_FILE, config.RF_READY_DATASET_FILE)

    rf_df = safe_read_csv(config.RF_READY_DATASET_FILE)

    if rf_df.empty:
        return pd.DataFrame()

    if "salary_range_label" not in rf_df.columns:
        return pd.DataFrame()

    return rf_df


def get_current_counts(rf_df: pd.DataFrame) -> dict:
    counts = {label: 0 for label in TARGET_LABELS}

    if rf_df.empty:
        return counts

    value_counts = rf_df["salary_range_label"].value_counts().to_dict()

    for label in TARGET_LABELS:
        counts[label] = min(int(value_counts.get(label, 0)), TARGET_PER_CLASS)

    return counts


def target_done(counts: dict) -> bool:
    return all(counts[label] >= TARGET_PER_CLASS for label in TARGET_LABELS)


def print_counts(counts: dict):
    print("\nDistribusi target sementara:")
    for label in TARGET_LABELS:
        print(f"{label}: {counts.get(label, 0)}/{TARGET_PER_CLASS}")


def merge_and_save_raw_batch(new_records: list[dict]) -> pd.DataFrame:
    pieces = []

    old_batch_df = safe_read_csv(BATCH_RAW_PATH)
    checkpoint_df = safe_read_csv(CHECKPOINT_PATH)
    new_df = pd.DataFrame(new_records)

    for df in [old_batch_df, checkpoint_df, new_df]:
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    batch_df = deduplicate_raw(pd.concat(pieces, ignore_index=True))
    BATCH_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(BATCH_RAW_PATH, index=False, encoding="utf-8-sig")

    return batch_df


def save_checkpoint(new_records: list[dict]):
    if not new_records:
        return

    checkpoint_df = pd.DataFrame(new_records)

    if "job_url" in checkpoint_df.columns:
        checkpoint_df = checkpoint_df.drop_duplicates(subset=["job_url"], keep="last")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_df.to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8-sig")

    print(f"Checkpoint balance tersimpan: {CHECKPOINT_PATH} dengan {len(checkpoint_df)} baris sesi ini")


def build_combined_outputs():
    pieces = []

    combined_old = safe_read_csv(config.RAW_OUTPUT_FILE)
    if not combined_old.empty:
        pieces.append(combined_old)

    for path in sorted(config.BATCH_OUTPUT_DIR.glob("jobstreet_raw_batch_*.csv")):
        df = safe_read_csv(path)
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw_df = deduplicate_raw(pd.concat(pieces, ignore_index=True))
    raw_df.to_csv(config.RAW_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    proxy_df = build_proxy_dataset(config.RAW_OUTPUT_FILE, config.PROXY_DATASET_FILE)
    rf_df = build_rf_ready_dataset(config.PROXY_DATASET_FILE, config.RF_READY_DATASET_FILE)

    return raw_df, proxy_df, rf_df


def build_batch_outputs():
    if not BATCH_RAW_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    proxy_df = build_proxy_dataset(BATCH_RAW_PATH, BATCH_PROXY_PATH)
    rf_df = build_rf_ready_dataset(BATCH_PROXY_PATH, BATCH_RF_READY_PATH)

    return proxy_df, rf_df


def save_balanced_rf_ready():
    rf_df = safe_read_csv(config.RF_READY_DATASET_FILE)

    if rf_df.empty:
        print("RF-ready gabungan kosong. Balanced dataset belum bisa dibuat.")
        return pd.DataFrame()

    if "salary_range_label" not in rf_df.columns:
        print("Kolom salary_range_label tidak ditemukan. Balanced dataset belum bisa dibuat.")
        return pd.DataFrame()

    rf_df.to_csv(FULL_RF_READY_BACKUP_PATH, index=False, encoding="utf-8-sig")

    balanced_parts = []

    for label in TARGET_LABELS:
        label_df = rf_df[rf_df["salary_range_label"] == label].copy()

        if len(label_df) >= TARGET_PER_CLASS:
            label_df = label_df.head(TARGET_PER_CLASS)

        balanced_parts.append(label_df)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)

    balanced_df.to_csv(BALANCED_RF_READY_PATH, index=False, encoding="utf-8-sig")

    balanced_df.to_csv(config.RF_READY_DATASET_FILE, index=False, encoding="utf-8-sig")

    print("\nDataset full sebelum balancing tersimpan di:")
    print(FULL_RF_READY_BACKUP_PATH)

    print("\nDataset balanced tersimpan di:")
    print(BALANCED_RF_READY_PATH)

    print("\nFile RF-ready utama juga sudah ditimpa dengan versi balanced:")
    print(config.RF_READY_DATASET_FILE)

    print("\nDistribusi balanced final:")
    print(balanced_df["salary_range_label"].value_counts())

    return balanced_df


def extract_candidate_label(card: dict, detail_text: str) -> tuple[str, int | None]:
    salary_source_text = normalize_text(
        card.get("salary_text", ""),
        card.get("job_title", ""),
        card.get("card_text", ""),
        detail_text,
    )

    salary_min, salary_max, salary_avg = parse_salary(salary_source_text)
    salary_label = salary_to_range_label(salary_avg)

    return salary_label, salary_avg


def setup_balance_config():
    config.ACTIVE_BATCH_NAME = BATCH_NAME
    config.SESSION_CHECKPOINT_FILE = CHECKPOINT_PATH

    if BATCH_NAME in config.SEARCH_KEYWORD_BATCHES:
        config.SEARCH_KEYWORDS = config.SEARCH_KEYWORD_BATCHES[BATCH_NAME]
    else:
        config.SEARCH_KEYWORDS = BALANCE_KEYWORDS

    config.MAX_PAGE_PER_KEYWORD = max(int(config.MAX_PAGE_PER_KEYWORD), 20)
    config.MAX_JOBS_PER_KEYWORD = max(int(config.MAX_JOBS_PER_KEYWORD), 300)


def run_balance_scraping():
    setup_balance_config()

    print("=" * 80)
    print("Balance scraping untuk salary_range_label")
    print(f"Target per kelas: {TARGET_PER_CLASS}")
    print(f"Batch aktif: {BATCH_NAME}")
    print(f"Jumlah keyword balance: {len(config.SEARCH_KEYWORDS)}")
    print(f"CSV raw batch balance: {BATCH_RAW_PATH}")
    print(f"Checkpoint balance: {CHECKPOINT_PATH}")
    print("=" * 80)

    current_rf_df = rebuild_current_datasets()
    counts = get_current_counts(current_rf_df)

    print_counts(counts)

    if target_done(counts):
        print("\nSemua kelas sudah mencapai target. Scraping tidak dijalankan.")
        save_balanced_rf_ready()
        return

    seen_urls = load_seen_urls()
    print(f"\nURL yang sudah pernah diambil dan akan dilewati: {len(seen_urls)}")

    accepted_records = []

    try:
        with JobStreetScraper(headless=config.HEADLESS) as scraper:
            for keyword_index, keyword in enumerate(config.SEARCH_KEYWORDS, start=1):
                if target_done(counts):
                    print("\nSemua kelas sudah mencapai target. Scraping dihentikan.")
                    break

                print(f"\n[{keyword_index}/{len(config.SEARCH_KEYWORDS)}] Keyword balance: {keyword}")

                empty_pages = 0

                for page in range(1, config.MAX_PAGE_PER_KEYWORD + 1):
                    if target_done(counts):
                        print("\nSemua kelas sudah mencapai target. Pindah ke finalisasi.")
                        break

                    search_url = scraper._search_url(keyword, page)
                    print(f"Open page {page}: {search_url}")

                    loaded = scraper._load_page(search_url, config.PAGE_LOAD_DELAY_SECONDS)

                    if not loaded:
                        empty_pages += 1
                        if empty_pages >= 2:
                            print("Dua halaman gagal berturut turut. Pindah keyword berikutnya.")
                            break
                        continue

                    scraper._scroll_page()
                    html = scraper._get_page_source()

                    if not html:
                        empty_pages += 1
                        continue

                    cards = scraper._parse_cards(html, keyword)

                    if not cards:
                        empty_pages += 1
                        print("Tidak ada kartu lowongan terbaca.")
                        if empty_pages >= 2:
                            print("Dua halaman kosong berturut turut. Pindah keyword berikutnya.")
                            break
                        continue

                    accepted_on_page = 0

                    for card in cards:
                        job_url = str(card.get("job_url", "")).strip()

                        if not job_url:
                            continue

                        if job_url in seen_urls:
                            continue

                        seen_urls.add(job_url)

                        detail_text = scraper._fetch_detail_text(job_url)

                        salary_label, salary_avg = extract_candidate_label(card, detail_text)

                        if salary_label not in TARGET_LABELS:
                            continue

                        if counts[salary_label] >= TARGET_PER_CLASS:
                            print(f"Skip karena kelas sudah penuh: {salary_label}")
                            continue

                        description = scraper._clean_text(
                            f"{card.get('card_text', '')} {detail_text}"
                        )

                        record = JobRecord(
                            source_keyword=card.get("source_keyword", ""),
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
                            f"{record.job_title[:80]}"
                        )

                        if len(accepted_records) % SAVE_EVERY_ACCEPTED_RECORD == 0:
                            save_checkpoint(accepted_records)
                            merge_and_save_raw_batch(accepted_records)

                        if target_done(counts):
                            break

                    if accepted_on_page == 0:
                        empty_pages += 1
                        print("Halaman terbaca, tetapi tidak ada data baru yang masuk target.")
                        if empty_pages >= 4:
                            print("Empat halaman tanpa data target baru. Pindah keyword berikutnya.")
                            break
                    else:
                        empty_pages = 0
                        save_checkpoint(accepted_records)
                        merge_and_save_raw_batch(accepted_records)
                        print_counts(counts)

    except KeyboardInterrupt:
        print("\nScraping dihentikan manual. Data yang sudah diterima akan tetap disimpan.")

    except Exception as exc:
        print(f"\nScraping berhenti karena error: {type(exc).__name__}: {exc}")
        print("Data yang sudah diterima akan tetap disimpan.")

    finally:
        print("\nFinalisasi file batch dan gabungan...")

        save_checkpoint(accepted_records)
        batch_df = merge_and_save_raw_batch(accepted_records)

        if not batch_df.empty:
            build_batch_outputs()

        raw_df, proxy_df, rf_df = build_combined_outputs()

        print("\nRingkasan output gabungan:")
        print(f"Raw gabungan: {len(raw_df)} baris")
        print(f"Proxy gabungan: {len(proxy_df)} baris")
        print(f"RF-ready gabungan sebelum balancing: {len(rf_df)} baris")

        if not rf_df.empty and "salary_range_label" in rf_df.columns:
            print("\nDistribusi RF-ready gabungan sebelum balancing:")
            print(rf_df["salary_range_label"].value_counts())

        save_balanced_rf_ready()

        print("\nSelesai.")


if __name__ == "__main__":
    run_balance_scraping()