from pathlib import Path
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
BATCHES_DIR = OUTPUTS_DIR / "batches"

MERGE_CONFIG = [
    {
        "pattern": "jobstreet_raw_batch_*.csv",
        "output": "jobstreet_raw.csv"
    },
    {
        "pattern": "graduate_salary_proxy_batch_*.csv",
        "output": "graduate_salary_proxy_dataset.csv"
    },
    {
        "pattern": "graduate_salary_rf_ready_batch_*.csv",
        "output": "graduate_salary_rf_ready.csv"
    }
]


def safe_write_csv(df, output_path):
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Berhasil simpan: {output_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )

        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"File utama terkunci, jadi disimpan sebagai: {fallback_path}")


def merge_batch_files(pattern, output_name):
    files = sorted(BATCHES_DIR.glob(pattern))

    if not files:
        print(f"Tidak ada file batch untuk pola: {pattern}")
        return

    print("=" * 80)
    print(f"Menggabungkan pola: {pattern}")
    print(f"Jumlah file ditemukan: {len(files)}")

    dfs = []

    for file in files:
        try:
            df_part = pd.read_csv(file)
            df_part["source_batch_file"] = file.name
            dfs.append(df_part)

            print(f"Dibaca: {file.name} | {len(df_part)} baris")

        except Exception as e:
            print(f"Gagal baca {file.name}: {e}")

    if not dfs:
        print(f"Tidak ada dataframe valid untuk pola: {pattern}")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    before_dedup = len(df_all)

    if "url" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["url"], keep="first")
    elif "job_url" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["job_url"], keep="first")
    elif "title" in df_all.columns and "company" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["title", "company"], keep="first")
    else:
        df_all = df_all.drop_duplicates(keep="first")

    after_dedup = len(df_all)

    print(f"Jumlah sebelum deduplikasi: {before_dedup}")
    print(f"Jumlah setelah deduplikasi: {after_dedup}")

    output_path = OUTPUTS_DIR / output_name
    safe_write_csv(df_all, output_path)


def main():
    if not BATCHES_DIR.exists():
        raise FileNotFoundError(f"Folder batches tidak ditemukan: {BATCHES_DIR}")

    for config in MERGE_CONFIG:
        merge_batch_files(
            pattern=config["pattern"],
            output_name=config["output"]
        )


if __name__ == "__main__":
    main()