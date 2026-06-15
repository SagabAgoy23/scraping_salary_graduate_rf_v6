# Scraping Salary Graduate RF v6

Versi ini memisahkan keyword scraping menjadi batch agar tidak menjalankan semua keyword sekaligus.
Tujuannya agar jika ChromeDriver error di tengah jalan, data batch yang sudah terkumpul tetap tersimpan.

## Instalasi

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Lihat daftar batch

```powershell
python main.py --list-batches
```

## Jalankan satu batch

```powershell
python main.py --batch 1
python main.py --batch 2
python main.py --batch 3
```

Bisa juga pakai nama pendek:

```powershell
python main.py --batch it
python main.py --batch finance
python main.py --batch admin
```

## Output per batch

Setiap batch membuat file sendiri di folder:

```text
outputs/batches/
```

Contoh:

```text
outputs/batches/jobstreet_raw_batch_01_umum_fresh_graduate.csv
outputs/batches/graduate_salary_proxy_batch_01_umum_fresh_graduate.csv
outputs/batches/graduate_salary_rf_ready_batch_01_umum_fresh_graduate.csv
```

## Output gabungan

Setelah setiap batch selesai atau error, script tetap membangun output gabungan:

```text
outputs/jobstreet_raw.csv
outputs/graduate_salary_proxy_dataset.csv
outputs/graduate_salary_rf_ready.csv
```

## Checkpoint per batch

Checkpoint disimpan di:

```text
outputs/checkpoints/
```

Kalau error, jalankan ulang batch yang sama. URL yang sudah masuk checkpoint atau output gabungan akan dilewati.

## Training Random Forest

Setelah beberapa batch terkumpul:

```powershell
python train_random_forest.py
```

Hasil training:

```text
outputs/random_forest_results.txt
outputs/confusion_matrix.csv
outputs/feature_importance.csv
models/random_forest_salary_classifier.joblib
```
