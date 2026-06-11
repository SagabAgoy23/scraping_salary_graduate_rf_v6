import pandas as pd

import config
from feature_extractor import extract_features, normalize_text
from salary_parser import parse_salary, salary_to_range_id, salary_to_range_label


def _safe_str(value):
    if pd.isna(value):
        return ""
    return str(value)


def build_proxy_dataset(raw_csv_path=config.RAW_OUTPUT_FILE, output_path=config.PROXY_DATASET_FILE):
    raw_csv_path = str(raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    records = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        salary_source_text = normalize_text(
            _safe_str(row_dict.get("salary_text", "")),
            _safe_str(row_dict.get("job_title", "")),
            _safe_str(row_dict.get("job_description", "")),
        )
        salary_min, salary_max, salary_avg = parse_salary(salary_source_text)
        features = extract_features(row_dict)

        record = {
            "source_id": idx + 1,
            "source_keyword": _safe_str(row_dict.get("source_keyword", "")),
            "scraped_at": _safe_str(row_dict.get("scraped_at", "")),
            "job_title": _safe_str(row_dict.get("job_title", "")),
            "company": _safe_str(row_dict.get("company", "")),
            "location": _safe_str(row_dict.get("location", "")),
            "job_url": _safe_str(row_dict.get("job_url", "")),
            "salary_text": _safe_str(row_dict.get("salary_text", "")),
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_avg": salary_avg,
            "salary_range_id": salary_to_range_id(salary_avg),
            "salary_range_label": salary_to_range_label(salary_avg),
            "salary_available": int(salary_avg is not None),
        }
        record.update(features)
        records.append(record)

    result = pd.DataFrame(records)

    ordered_columns = [
        "source_id", "source_keyword", "scraped_at", "job_title", "company", "location", "job_url",
        "salary_text", "salary_min", "salary_max", "salary_avg", "salary_range_id", "salary_range_label", "salary_available",
        "f_education_d3", "f_education_s1", "f_education_s2", "f_ipk_required", "f_min_gpa",
        "f_major_it", "f_major_engineering", "f_major_business", "f_academic_achievement", "f_scholarship",
        "f_transcript_required", "f_certificate_required", "f_english_required", "f_computer_skill_required",
        "f_portfolio_required", "f_project_required", "f_organization_required", "f_leadership_required",
        "f_student_cadre_proxy", "f_event_committee", "f_communication_required", "f_teamwork_required",
        "f_internship_required", "f_fresh_graduate_allowed", "f_work_experience_required", "f_experience_years_min",
        "job_family", "region_group",
    ]
    existing = [col for col in ordered_columns if col in result.columns]
    result = result[existing]
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def build_rf_ready_dataset(proxy_csv_path=config.PROXY_DATASET_FILE, output_path=config.RF_READY_DATASET_FILE):
    df = pd.read_csv(proxy_csv_path)

    # Dataset Random Forest hanya boleh berisi baris yang punya target gaji.
    rf_df = df[(df["salary_available"] == 1) & (df["salary_range_label"] != "unknown")].copy()

    feature_columns = [
        "f_education_d3", "f_education_s1", "f_education_s2", "f_ipk_required", "f_min_gpa",
        "f_major_it", "f_major_engineering", "f_major_business", "f_academic_achievement", "f_scholarship",
        "f_transcript_required", "f_certificate_required", "f_english_required", "f_computer_skill_required",
        "f_portfolio_required", "f_project_required", "f_organization_required", "f_leadership_required",
        "f_student_cadre_proxy", "f_event_committee", "f_communication_required", "f_teamwork_required",
        "f_internship_required", "f_fresh_graduate_allowed", "f_work_experience_required", "f_experience_years_min",
        "job_family", "region_group",
    ]
    target_column = "salary_range_label"
    metadata_columns = ["source_id", "job_title", "company", "location", "job_url", "salary_avg", "salary_range_id"]

    columns = metadata_columns + feature_columns + [target_column]
    columns = [col for col in columns if col in rf_df.columns]
    rf_df = rf_df[columns]
    rf_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return rf_df


# Backward compatibility dengan versi lama.
def build_ml_dataset(raw_csv_path=config.RAW_OUTPUT_FILE):
    proxy = build_proxy_dataset(raw_csv_path)
    ready = build_rf_ready_dataset()
    return proxy, ready
