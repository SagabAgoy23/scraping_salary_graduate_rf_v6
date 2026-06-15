import re
from typing import Optional, Tuple

from config import SALARY_BINS, SALARY_LABELS


def _normalize_number(raw: str) -> Optional[int]:
    if raw is None:
        return None

    text = str(raw).lower().strip()
    text = text.replace("rp", "").replace("idr", "").replace("/", " ")
    text = text.replace("per bulan", "").replace("per month", "")
    text = text.replace("bulan", "").replace("month", "")
    text = text.replace(" ", "")

    multiplier = 1
    if "juta" in text or "jt" in text:
        multiplier = 1_000_000
        text = text.replace("juta", "").replace("jt", "")
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]

    # 4,5 juta atau 4.5 juta
    if multiplier == 1_000_000 and ("," in text or "." in text):
        text = text.replace(",", ".")
        try:
            return int(float(text) * multiplier)
        except ValueError:
            return None

    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None

    value = int(digits)

    # Angka 4 atau 6 yang muncul bersama kata juta sebelumnya diperlakukan sebagai jutaan.
    if multiplier == 1_000_000 and value < 1000:
        return value * multiplier

    # Angka seperti 4500 kemungkinan ribuan.
    if 1000 <= value < 100_000:
        return value * 1000

    return value


def extract_salary_values(text: str) -> list[int]:
    if not text:
        return []

    normalized = str(text).lower()
    patterns = [
        r"(?:rp|idr)\s*[0-9][0-9\.,]*\s*(?:juta|jt|k)?",
        r"[0-9]+(?:[\.,][0-9]+)?\s*(?:juta|jt)",
        r"[0-9]{1,3}(?:\.[0-9]{3}){1,3}",
        r"[0-9]{6,9}",
    ]

    values = []
    for pattern in patterns:
        for match in re.findall(pattern, normalized):
            value = _normalize_number(match)
            if value and 500_000 <= value <= 100_000_000:
                values.append(value)

    # Hapus duplikat sambil menjaga urutan.
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    return unique_values


def parse_salary(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    values = extract_salary_values(text)
    if not values:
        return None, None, None

    salary_min = min(values)
    salary_max = max(values)

    # Jika hanya satu angka, pakai sebagai rata rata perkiraan.
    salary_avg = int((salary_min + salary_max) / 2)
    return salary_min, salary_max, salary_avg


def salary_to_range_id(salary_avg: Optional[int]) -> Optional[int]:
    if salary_avg is None:
        return None

    for idx in range(len(SALARY_BINS) - 1):
        lower = SALARY_BINS[idx]
        upper = SALARY_BINS[idx + 1]
        if lower <= salary_avg < upper:
            return idx
    return None


def salary_to_range_label(salary_avg: Optional[int]) -> str:
    range_id = salary_to_range_id(salary_avg)
    if range_id is None:
        return "unknown"
    return SALARY_LABELS[range_id]
