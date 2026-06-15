from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config


def _make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _write_message(message):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "random_forest_results.txt").write_text(message, encoding="utf-8")
    print(message)


def _split_feature_types(X):
    numeric_features = []
    categorical_features = []

    for col in X.columns:
        if is_numeric_dtype(X[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def train_random_forest(dataset_path=config.RF_READY_DATASET_FILE):
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        _write_message(
            f"File dataset tidak ditemukan: {dataset_path}\n"
            "Jalankan python main.py terlebih dahulu agar file graduate_salary_rf_ready.csv dibuat."
        )
        return None

    df = pd.read_csv(dataset_path)

    target = "salary_range_label"
    if target not in df.columns:
        _write_message(f"Kolom target {target} tidak ditemukan di dataset.")
        return None

    df = df.dropna(subset=[target]).copy()
    df = df[df[target].astype(str).str.lower() != "unknown"].copy()

    drop_columns = [
        "source_id",
        "job_title",
        "company",
        "location",
        "job_url",
        "salary_avg",
        "salary_range_id",
        target,
    ]

    feature_columns = [col for col in df.columns if col not in drop_columns]

    if len(feature_columns) == 0:
        _write_message("Tidak ada kolom fitur yang bisa dipakai untuk training.")
        return None

    if len(df) < 20:
        _write_message(
            "Data berlabel gaji masih kurang dari 20 baris.\n"
            "Naikkan MAX_PAGE_PER_KEYWORD atau MAX_JOBS_PER_KEYWORD di config.py, "
            "lalu jalankan ulang python main.py."
        )
        return None

    if df[target].nunique() < 2:
        _write_message(
            "Target salary_range_label hanya memiliki satu kelas.\n"
            "Random Forest belum bisa dievaluasi karena model klasifikasi butuh minimal dua kelas target."
        )
        return None

    X = df[feature_columns].copy()
    y = df[target].astype(str)

    numeric_features, categorical_features = _split_feature_types(X)

    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))

    if categorical_features:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_encoder()),
            ]),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ])

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2 and len(df) >= len(class_counts) * 5
    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)

    result_text = (
        "RANDOM FOREST CLASSIFICATION RESULT\n"
        f"Dataset: {dataset_path}\n"
        f"Jumlah data: {len(df)}\n"
        f"Jumlah data latih: {len(X_train)}\n"
        f"Jumlah data uji: {len(X_test)}\n"
        f"Akurasi: {acc:.4f}\n\n"
        "Distribusi target:\n"
        f"{y.value_counts().to_string()}\n\n"
        "Numeric features:\n"
        f"{numeric_features}\n\n"
        "Categorical features:\n"
        f"{categorical_features}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )

    (config.OUTPUT_DIR / "random_forest_results.txt").write_text(result_text, encoding="utf-8")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        config.OUTPUT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    try:
        preprocessor_fitted = model.named_steps["preprocess"]
        feature_names = preprocessor_fitted.get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        (config.OUTPUT_DIR / "feature_importance_error.txt").write_text(str(exc), encoding="utf-8")

    joblib.dump(model, config.MODEL_DIR / "random_forest_salary_classifier.joblib")

    print("Training selesai.")
    print(result_text)
    return model


if __name__ == "__main__":
    train_random_forest()
