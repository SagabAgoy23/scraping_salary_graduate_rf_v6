# Skema Dataset Yang Dipakai

Judul penelitian:
Klasifikasi Range Gaji Awal Lulusan Perguruan Tinggi Menggunakan Algoritma Random Forest Berdasarkan Portofolio Akademik dan Pengalaman Organisasi.

## Masalah utama

Dataset ideal untuk judul ini bukan sekadar dataset lowongan kerja. Dataset ideal adalah data individu lulusan yang berisi profil akademik, portofolio, pengalaman organisasi, pengalaman kerja awal, dan gaji awal aktual.

Karena scraper mengambil data dari JobStreet, dataset yang dihasilkan adalah dataset proxy. Artinya fitur seperti IPK, portofolio, organisasi, leadership, sertifikat, dan pengalaman diekstrak dari requirement lowongan, bukan dari profil alumni asli.

## Output project

### outputs/jobstreet_raw.csv

Data mentah hasil scraping JobStreet.

Kolom utama:

| Kolom | Makna |
| --- | --- |
| source_keyword | Keyword pencarian |
| job_title | Judul lowongan |
| company | Nama perusahaan |
| location | Lokasi kerja |
| salary_text | Teks gaji dari lowongan |
| job_description | Deskripsi lengkap lowongan |
| job_url | URL lowongan |
| scraped_at | Waktu scraping |

### outputs/graduate_salary_proxy_dataset.csv

Dataset proxy yang sudah disesuaikan dengan judul penelitian.

Kolom target:

| Kolom | Makna |
| --- | --- |
| salary_min | Gaji minimum bulanan hasil parsing |
| salary_max | Gaji maksimum bulanan hasil parsing |
| salary_avg | Rata rata gaji bulanan |
| salary_range_id | Kode kelas target |
| salary_range_label | Label kelas target |
| salary_available | Penanda apakah lowongan punya informasi gaji |

Label target:

| salary_range_id | salary_range_label |
| --- | --- |
| 0 | < 4 juta |
| 1 | 4 sampai < 6 juta |
| 2 | 6 sampai < 8 juta |
| 3 | 8 sampai < 10 juta |
| 4 | >= 10 juta |

Fitur portofolio akademik:

| Kolom | Makna |
| --- | --- |
| f_education_d3 | Lowongan menerima atau mensyaratkan D3 |
| f_education_s1 | Lowongan menerima atau mensyaratkan S1 |
| f_education_s2 | Lowongan menerima atau mensyaratkan S2 |
| f_ipk_required | Ada syarat IPK atau GPA |
| f_min_gpa | Nilai IPK minimum jika ditemukan |
| f_major_it | Jurusan IT atau Informatika relevan |
| f_major_engineering | Jurusan teknik relevan |
| f_major_business | Jurusan bisnis, ekonomi, manajemen, atau akuntansi relevan |
| f_academic_achievement | Ada kata kunci prestasi akademik, lomba, kompetisi, atau award |
| f_scholarship | Ada kata kunci beasiswa |
| f_transcript_required | Ada syarat transkrip |
| f_certificate_required | Ada syarat sertifikat |
| f_english_required | Ada syarat bahasa Inggris |
| f_computer_skill_required | Ada syarat kemampuan komputer atau software |
| f_portfolio_required | Ada syarat portofolio |
| f_project_required | Ada syarat project atau tugas akhir |

Fitur pengalaman organisasi:

| Kolom | Makna |
| --- | --- |
| f_organization_required | Ada kata kunci organisasi |
| f_leadership_required | Ada kata kunci leadership atau kepemimpinan |
| f_student_cadre_proxy | Proxy student cadre, seperti pengurus, ketua organisasi, aktif organisasi |
| f_event_committee | Ada pengalaman panitia, event, committee, volunteer |
| f_communication_required | Ada syarat komunikasi |
| f_teamwork_required | Ada syarat teamwork atau kerja sama tim |

Fitur pengalaman kerja awal:

| Kolom | Makna |
| --- | --- |
| f_internship_required | Ada kata kunci magang atau internship |
| f_fresh_graduate_allowed | Lowongan menerima fresh graduate atau entry level |
| f_work_experience_required | Ada syarat pengalaman kerja |
| f_experience_years_min | Minimal tahun pengalaman jika ditemukan |

Fitur pasar kerja:

| Kolom | Makna |
| --- | --- |
| job_family | Keluarga pekerjaan, misalnya data, software, business, finance, administration, engineering |
| region_group | Kelompok wilayah kerja |

### outputs/graduate_salary_rf_ready.csv

Dataset yang sudah difilter untuk training Random Forest. Baris tanpa informasi gaji dibuang karena tidak memiliki target klasifikasi.

## Dataset ideal jika memakai survei alumni

Jika ingin benar benar sesuai dengan judul, data terbaik adalah survei alumni atau data career center kampus. Gunakan template berikut:

| Kolom | Makna |
| --- | --- |
| respondent_id | ID responden |
| gender | Jenis kelamin jika diperlukan sebagai variabel kontrol |
| graduation_year | Tahun lulus |
| degree | D3, S1, atau S2 |
| major_category | Kategori jurusan |
| gpa | IPK |
| certificate_count | Jumlah sertifikat |
| english_certificate | Sertifikat bahasa Inggris |
| computer_certificate | Sertifikat komputer atau teknis |
| portfolio_count | Jumlah portofolio karya |
| project_count | Jumlah project akademik atau non akademik |
| internship_experience | Pernah magang |
| organization_experience | Pernah organisasi |
| leadership_experience | Pernah menjabat posisi kepemimpinan |
| student_cadre_experience | Pernah menjadi pengurus atau kader mahasiswa |
| competition_count | Jumlah lomba atau kompetisi |
| scholarship_count | Jumlah beasiswa |
| employment_region | Wilayah kerja pertama |
| employment_industry | Industri kerja pertama |
| company_type | Jenis perusahaan |
| starting_salary | Gaji awal bulanan |
| salary_range_label | Target klasifikasi |

## Kesimpulan teknis

Project ini sudah menyesuaikan output scraping ke struktur dataset penelitian. Namun untuk validitas akademik, sebutkan dengan jelas bahwa data JobStreet adalah dataset proxy berbasis requirement lowongan. Jika dosen meminta data lulusan asli, gunakan template survei alumni.
