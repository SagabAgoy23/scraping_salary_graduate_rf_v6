"""
Konfigurasi scraper dan pembentukan dataset.
Fokus penelitian:
Klasifikasi Range Gaji Awal Lulusan Perguruan Tinggi Menggunakan Algoritma Random Forest
Berdasarkan Portofolio Akademik dan Pengalaman Organisasi.

Versi v6:
1. Keyword dipisah menjadi batch/grup agar tidak scraping semua keyword dalam satu kali jalan.
2. Setiap batch langsung punya CSV sendiri.
3. Checkpoint dipisah per batch agar jika error, batch yang sudah dikumpulkan tidak hilang.
4. Output gabungan tetap dibuat untuk training Random Forest.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batches"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
BATCH_OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

RAW_OUTPUT_FILE = OUTPUT_DIR / "jobstreet_raw.csv"
PROXY_DATASET_FILE = OUTPUT_DIR / "graduate_salary_proxy_dataset.csv"
RF_READY_DATASET_FILE = OUTPUT_DIR / "graduate_salary_rf_ready.csv"

# Keyword dibagi per batch agar proses lebih aman.
# Jalankan satu batch dulu, misalnya: python main.py --batch 1
# Setelah selesai, lanjut: python main.py --batch 2, dan seterusnya.
SEARCH_KEYWORD_BATCHES = {
    "01_umum_fresh_graduate": [
        "fresh graduate s1",
        "fresh graduate",
        "lulusan baru",
        "entry level",
        "entry level s1",
        "junior fresh graduate",
        "staff fresh graduate",
        "officer fresh graduate",
        "associate fresh graduate",
    ],
    "02_trainee_program": [
        "management trainee fresh graduate",
        "graduate trainee",
        "officer development program",
        "management development program",
        "trainee program fresh graduate",
        "leader trainee fresh graduate",
        "management trainee s1",
    ],
    "03_it_software": [
        "junior programmer fresh graduate",
        "junior developer fresh graduate",
        "junior web developer fresh graduate",
        "backend developer fresh graduate",
        "frontend developer fresh graduate",
        "mobile developer fresh graduate",
        "software engineer fresh graduate",
        "it support fresh graduate",
        "system analyst fresh graduate",
        "qa tester fresh graduate",
        "programmer junior",
        "web developer junior",
        "qa tester junior",
        "it support junior",
    ],
    "04_data_business": [
        "data analyst fresh graduate",
        "junior data analyst fresh graduate",
        "business intelligence fresh graduate",
        "business analyst fresh graduate",
        "data analyst junior",
        "business analyst junior",
    ],
    "05_finance_accounting_tax_audit": [
        "finance staff fresh graduate",
        "accounting staff fresh graduate",
        "tax staff fresh graduate",
        "audit staff fresh graduate",
        "staff akuntansi fresh graduate",
        "staff finance fresh graduate",
        "staff pajak fresh graduate",
    ],
    "06_admin_operation_logistic": [
        "admin staff fresh graduate",
        "operation staff fresh graduate",
        "staff administrasi fresh graduate",
        "staff operasional fresh graduate",
        "staff gudang fresh graduate",
        "staff purchasing fresh graduate",
        "staff export import fresh graduate",
    ],
    "07_marketing_sales_hr_service": [
        "marketing staff fresh graduate",
        "sales executive fresh graduate",
        "customer service fresh graduate",
        "hr staff fresh graduate",
        "staff marketing fresh graduate",
        "staff sales fresh graduate",
        "customer service officer fresh graduate",
    ],
    "08_internship_banking": [
        "internship s1",
        "magang mahasiswa",
        "internship fresh graduate",
        "teller fresh graduate",
        "banking staff fresh graduate",
    ],
}

# Nilai ini akan diubah otomatis oleh main.py sesuai batch yang dijalankan.
SEARCH_KEYWORDS = SEARCH_KEYWORD_BATCHES["01_umum_fresh_graduate"]
ACTIVE_BATCH_NAME = "01_umum_fresh_graduate"
SESSION_CHECKPOINT_FILE = CHECKPOINT_DIR / "jobstreet_raw_checkpoint_01_umum_fresh_graduate.csv"

# Batas per keyword dalam batch. Kalau masih kurang banyak, naikkan angka ini.
MAX_PAGE_PER_KEYWORD = 10
MAX_JOBS_PER_KEYWORD = 100

# Jika True, output gabungan lama akan dipakai untuk deduplikasi URL.
APPEND_EXISTING_RAW = True

# Checkpoint agar hasil sebelum crash tidak hilang.
CHECKPOINT_EVERY_N_RECORDS = 10

# Pengaturan stabilitas Selenium. Jika Chrome/Chromedriver mati di tengah scraping, scraper akan restart.
MAX_PAGE_RETRIES = 3
MAX_DRIVER_RESTARTS = 20
RESTART_DELAY_SECONDS = 3
SELENIUM_PAGE_LOAD_TIMEOUT = 60
SKIP_EXISTING_URLS = True

# True agar browser tidak muncul. False kalau mau debugging visual.
HEADLESS = False

# Tetap True agar fitur dari deskripsi detail lowongan tidak miskin.
FETCH_JOB_DETAIL = True

# Delay. Jangan dibuat 0 karena lebih rawan gagal dimuat atau terdeteksi agresif.
PAGE_LOAD_DELAY_SECONDS = 3
DETAIL_LOAD_DELAY_SECONDS = 2
SCROLL_PAUSE_SECONDS = 1

# Lokasi default. JobStreet Indonesia akan dipakai sebagai sumber.
JOBSTREET_BASE_URL = "https://id.jobstreet.com/id"

# Label target klasifikasi berbasis gaji bulanan rupiah.
SALARY_BINS = [0, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000, float("inf")]
SALARY_LABELS = [
    "0 sampai < 1 juta",
    "1 sampai < 2 juta",
    "2 sampai < 4 juta",
    "4 sampai < 6 juta",
    "6 sampai < 8 juta",
    "8 sampai < 10 juta",
    ">= 10 juta",
]

# =========================================================
# Konfigurasi tambahan untuk balancing salary range
# Paste di bagian paling bawah config.py
# =========================================================

SALARY_BINS = [
    0,
    1_000_000,
    2_000_000,
    4_000_000,
    6_000_000,
    8_000_000,
    10_000_000,
    float("inf"),
]

SALARY_LABELS = [
    "0 sampai < 1 juta",
    "1 sampai < 2 juta",
    "2 sampai < 4 juta",
    "4 sampai < 6 juta",
    "6 sampai < 8 juta",
    "8 sampai < 10 juta",
    ">= 10 juta",
]

SEARCH_KEYWORD_BATCHES["09_balance_salary"] = [
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