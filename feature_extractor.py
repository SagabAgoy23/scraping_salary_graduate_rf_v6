import re
from typing import Dict, Optional


def normalize_text(*parts: object) -> str:
    text = " ".join(str(part) for part in parts if part is not None)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text: str, keywords: list[str]) -> int:
    return int(any(keyword in text for keyword in keywords))


def find_min_gpa(text: str) -> Optional[float]:
    patterns = [
        r"ipk\s*(?:minimal|min|minimum|>=|di atas|lebih dari)?\s*[:\-]?\s*([0-4](?:[\.,][0-9]{1,2})?)",
        r"gpa\s*(?:minimal|min|minimum|>=)?\s*[:\-]?\s*([0-4](?:[\.,][0-9]{1,2})?)",
        r"minimum\s+gpa\s*([0-4](?:[\.,][0-9]{1,2})?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1).replace(",", "."))
                if 0 <= value <= 4:
                    return value
            except ValueError:
                pass
    return None


def find_experience_years(text: str) -> int:
    patterns = [
        r"(?:minimal|min|minimum|at least)?\s*(\d+)\s*(?:tahun|thn|year|years)\s*(?:pengalaman|experience)",
        r"(?:pengalaman|experience)\s*(?:minimal|min|minimum)?\s*(\d+)\s*(?:tahun|thn|year|years)",
    ]

    values = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            try:
                values.append(int(match))
            except ValueError:
                continue

    return min(values) if values else 0


def infer_job_family(text: str) -> str:
    rules = {
        "data": ["data analyst", "data scientist", "data engineer", "business intelligence", "sql", "python", "machine learning"],
        "software": ["programmer", "developer", "software", "frontend", "backend", "fullstack", "web developer", "mobile developer"],
        "business": ["business analyst", "management trainee", "sales", "marketing", "account executive"],
        "finance": ["finance", "accounting", "akuntansi", "tax", "audit"],
        "administration": ["admin", "administrasi", "staff operasional", "operation staff"],
        "engineering": ["engineer", "teknik", "mechanical", "electrical", "civil"],
    }
    for label, keywords in rules.items():
        if contains_any(text, keywords):
            return label
    return "other"


def infer_region_group(location: str) -> str:
    text = normalize_text(location)
    if any(city in text for city in ["jakarta", "bogor", "depok", "tangerang", "bekasi"]):
        return "jabodetabek"
    if any(city in text for city in ["bandung", "cimahi"]):
        return "jawa barat"
    if any(city in text for city in ["surabaya", "malang", "sidoarjo"]):
        return "jawa timur"
    if any(city in text for city in ["semarang", "solo", "yogyakarta", "jogja"]):
        return "jawa tengah diy"
    if any(city in text for city in ["balikpapan", "samarinda", "bontang", "kalimantan"]):
        return "kalimantan"
    if any(city in text for city in ["medan", "palembang", "pekanbaru", "batam", "sumatera"]):
        return "sumatera"
    if any(city in text for city in ["makassar", "manado", "sulawesi"]):
        return "sulawesi"
    if text:
        return "lainnya"
    return "unknown"


def extract_features(row: Dict[str, object]) -> Dict[str, object]:
    text = normalize_text(
        row.get("job_title", ""),
        row.get("company", ""),
        row.get("location", ""),
        row.get("salary_text", ""),
        row.get("job_description", ""),
    )

    min_gpa = find_min_gpa(text)
    experience_years = find_experience_years(text)

    features = {
        # Portofolio akademik, mengikuti ide academic performance, degree, specialization, certificates.
        "f_education_d3": contains_any(text, ["d3", "diploma 3", "diploma iii"]),
        "f_education_s1": contains_any(text, ["s1", "sarjana", "bachelor", "undergraduate"]),
        "f_education_s2": contains_any(text, ["s2", "magister", "master"]),
        "f_ipk_required": int(min_gpa is not None or contains_any(text, ["ipk", "gpa"])),
        "f_min_gpa": min_gpa if min_gpa is not None else 0.0,
        "f_major_it": contains_any(text, ["informatika", "teknik informatika", "sistem informasi", "computer science", "information system", "it", "software"]),
        "f_major_engineering": contains_any(text, ["teknik", "engineering", "electrical", "mechanical", "industrial engineering", "civil engineering"]),
        "f_major_business": contains_any(text, ["manajemen", "management", "akuntansi", "accounting", "ekonomi", "business", "administrasi bisnis"]),
        "f_academic_achievement": contains_any(text, ["prestasi akademik", "academic achievement", "competition", "lomba", "kompetisi", "award", "penghargaan"]),
        "f_scholarship": contains_any(text, ["beasiswa", "scholarship"]),
        "f_transcript_required": contains_any(text, ["transkrip", "transcript"]),
        "f_certificate_required": contains_any(text, ["sertifikat", "certificate", "certification", "toefl", "ielts", "toeic", "brevet", "bnsp"]),
        "f_english_required": contains_any(text, ["english", "bahasa inggris", "toefl", "ielts", "toeic"]),
        "f_computer_skill_required": contains_any(text, ["komputer", "computer", "microsoft office", "excel", "word", "powerpoint", "sql", "python", "java", "javascript"]),
        # Portofolio karya.
        "f_portfolio_required": contains_any(text, ["portofolio", "portfolio", "github", "behance", "showcase"]),
        "f_project_required": contains_any(text, ["project", "proyek", "capstone", "final project", "tugas akhir", "skripsi"]),
        # Pengalaman organisasi dan social capital proxy.
        "f_organization_required": contains_any(text, ["organisasi", "organization", "organisational", "organizational", "himpunan", "bem", "ukm"]),
        "f_leadership_required": contains_any(text, ["leadership", "kepemimpinan", "memimpin", "leader", "ketua", "koordinator"]),
        "f_student_cadre_proxy": contains_any(text, ["student cadre", "class cadre", "pengurus", "ketua organisasi", "anggota organisasi", "aktif organisasi"]),
        "f_event_committee": contains_any(text, ["panitia", "committee", "event organizer", "event", "volunteer"]),
        "f_communication_required": contains_any(text, ["komunikasi", "communication", "communicative", "presentation", "presentasi"]),
        "f_teamwork_required": contains_any(text, ["teamwork", "team work", "kerja sama tim", "bekerja dalam tim", "collaboration", "kolaborasi"]),
        # Pengalaman kerja awal.
        "f_internship_required": contains_any(text, ["magang", "internship", "intern", "praktek kerja", "pkl"]),
        "f_fresh_graduate_allowed": contains_any(text, ["fresh graduate", "fresh graduates", "lulusan baru", "entry level", "graduate trainee", "management trainee"]),
        "f_work_experience_required": int(experience_years > 0 or contains_any(text, ["pengalaman kerja", "work experience", "experienced"])),
        "f_experience_years_min": experience_years,
        # Konteks pasar kerja.
        "job_family": infer_job_family(text),
        "region_group": infer_region_group(str(row.get("location", ""))),
    }

    return features
