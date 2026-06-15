from pathlib import Path
import pandas as pd

import config
import salary_parser
from preprocess_dataset import build_proxy_dataset, build_rf_ready_dataset


TARGET_PER_CLASS = 771

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

OLD_RF_READY_PATH = config.OUTPUT_DIR / "graduate_salary_rf_ready.csv"

LOW_CHECKPOINT_PATH = config.CHECKPOINT_DIR / "jobstreet_raw_checkpoint_09_low_salary_only.csv"
LOW_BATCH_RAW_PATH = config.BATCH_OUTPUT_DIR / "jobstreet_raw_batch_09_low_salary_only.csv"
LOW_BATCH_PROXY_PATH = config.BATCH_OUTPUT_DIR / "graduate_salary_proxy_batch_09_low_salary_only.csv"
LOW_BATCH_RF_READY_PATH = config.BATCH_OUTPUT_DIR / "graduate_salary_rf_ready_batch_09_low_salary_only.csv"

FINAL_OUTPUT_PATH = config.OUTPUT_DIR / "graduate_salary_rf_ready_low_added.csv"


def force_salary_range_config():
    config.SALARY_BINS = SALARY_BINS_NEW
    config.SALARY_LABELS = SALARY_LABELS_NEW

    salary_parser.SALARY_BINS = SALARY_BINS_NEW
    salary_parser.SALARY_LABELS = SALARY_LABELS_NEW


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"File tidak ditemukan: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as exc:
        print(f"Gagal membaca: {path}")
        print(f"Error: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def dedupe_by_job_url(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "job_url" in df.columns:
        df = df.dropna(subset=["job_url"])
        df = df.drop_duplicates(subset=["job_url"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    return df.reset_index(drop=True)


def relabel_from_salary_avg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "salary_avg" not in df.columns:
        return df

    df = df.copy()
    df["salary_avg"] = pd.to_numeric(df["salary_avg"], errors="coerce")

    df["salary_range_label"] = df["salary_avg"].apply(
        lambda value: salary_parser.salary_to_range_label(value)
        if pd.notna(value)
        else "unknown"
    )

    return df


def merge_checkpoint_into_low_batch_raw():
    checkpoint_df = read_csv_auto(LOW_CHECKPOINT_PATH)
    old_low_batch_df = read_csv_auto(LOW_BATCH_RAW_PATH)

    pieces = []

    if not old_low_batch_df.empty:
        pieces.append(old_low_batch_df)

    if not checkpoint_df.empty:
        pieces.append(checkpoint_df)

    if not pieces:
        print("Tidak ada data raw low salary yang bisa digabung.")
        return pd.DataFrame()

    low_raw_df = dedupe_by_job_url(pd.concat(pieces, ignore_index=True))

    LOW_BATCH_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    low_raw_df.to_csv(LOW_BATCH_RAW_PATH, index=False, encoding="utf-8-sig")

    print(f"Raw low salary gabungan tersimpan di: {LOW_BATCH_RAW_PATH}")
    print(f"Jumlah raw low salary: {len(low_raw_df)}")

    return low_raw_df


def build_low_rf_ready():
    if not LOW_BATCH_RAW_PATH.exists():
        print("Raw batch low salary belum ada.")
        return pd.DataFrame()

    build_proxy_dataset(LOW_BATCH_RAW_PATH, LOW_BATCH_PROXY_PATH)
    build_rf_ready_dataset(LOW_BATCH_PROXY_PATH, LOW_BATCH_RF_READY_PATH)

    low_rf_df = read_csv_auto(LOW_BATCH_RF_READY_PATH)
    low_rf_df = relabel_from_salary_avg(low_rf_df)

    low_rf_df = low_rf_df[
        low_rf_df["salary_range_label"].isin([
            "0 sampai < 1 juta",
            "1 sampai < 2 juta",
        ])
    ].copy()

    low_rf_df = dedupe_by_job_url(low_rf_df)
    low_rf_df.to_csv(LOW_BATCH_RF_READY_PATH, index=False, encoding="utf-8-sig")

    print(f"RF ready low salary tersimpan di: {LOW_BATCH_RF_READY_PATH}")
    print(f"Jumlah RF ready low salary: {len(low_rf_df)}")

    if not low_rf_df.empty:
        print("Distribusi low salary:")
        print(low_rf_df["salary_range_label"].value_counts())

    return low_rf_df


def merge_with_old_rf_ready():
    old_df = read_csv_auto(OLD_RF_READY_PATH)
    low_df = read_csv_auto(LOW_BATCH_RF_READY_PATH)

    if old_df.empty:
        print("Dataset lama kosong atau tidak terbaca.")
        return pd.DataFrame()

    old_df = relabel_from_salary_avg(old_df)
    low_df = relabel_from_salary_avg(low_df)

    combined_df = pd.concat([old_df, low_df], ignore_index=True)
    combined_df = dedupe_by_job_url(combined_df)

    final_parts = []

    for label in SALARY_LABELS_NEW:
        label_df = combined_df[combined_df["salary_range_label"] == label].copy()

        if label in ["0 sampai < 1 juta", "1 sampai < 2 juta"]:
            if len(label_df) > TARGET_PER_CLASS:
                label_df = label_df.head(TARGET_PER_CLASS)

        final_parts.append(label_df)

    final_df = pd.concat(final_parts, ignore_index=True)
    final_df = dedupe_by_job_url(final_df)

    FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nDataset lama tidak ditimpa:")
    print(OLD_RF_READY_PATH)

    print("\nDataset baru hasil gabungan tersimpan di:")
    print(FINAL_OUTPUT_PATH)

    print("\nDistribusi dataset baru:")
    print(final_df["salary_range_label"].value_counts())

    return final_df


def main():
    force_salary_range_config()

    print("=" * 80)
    print("Menggabungkan checkpoint low salary dengan dataset awal")
    print("=" * 80)

    merge_checkpoint_into_low_batch_raw()
    build_low_rf_ready()
    merge_with_old_rf_ready()

    print("\nSelesai.")


if __name__ == "__main__":
    main()