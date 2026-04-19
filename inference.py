"""
inference.py  ─  Model Inferensi Corn Yield Prediction
=======================================================
Load model RandomForest yang dilatih dari preprocessed_corn_data.csv
Fitur (28 kolom, sesuai urutan training):
  Household size, Acreage, Fertilizer amount, Laborers,
  Latitude, Longitude, + 22 kolom one-hot encoded

Cara pakai:
  from inference import predict_yield
  result = predict_yield({"Acreage": 0.2, "Fertilizer amount": 0.08, ...})
"""

import os
import logging
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("corn-inference")

# ── Kolom fitur (urutan harus sama persis dengan saat training) ───────────────
FEATURE_COLS = [
    "Household size", "Acreage", "Fertilizer amount", "Laborers",
    "Latitude", "Longitude",
    "Education_Degree", "Education_Diploma", "Education_Primary", "Education_Secondary",
    "Gender_Male",
    "Age bracket_36-45", "Age bracket_46-55", "Age bracket_56-65", "Age bracket_above 65",
    "Main credit source_Family", "Main credit source_Savings",
    "Farm records_Yes",
    "Main advisory source_Internet", "Main advisory source_Public gatherings",
    "Main advisory source_Radio", "Main advisory source_Television",
    "Extension provider_KALRO", "Extension provider_National Government",
    "Extension provider_Private Provider",
    "Advisory format_SMS text",
    "Advisory language_Kiswahili", "Advisory language_Vernacular",
]

# Default value (nilai median dari dataset) untuk fitur yang tidak diberikan
FEATURE_DEFAULTS = {
    "Household size"                      : 0.75,
    "Acreage"                             : 0.0667,
    "Fertilizer amount"                   : 0.0625,
    "Laborers"                            : 0.1667,
    "Latitude"                            : 0.5897,
    "Longitude"                           : 0.2778,
    "Education_Degree"                    : 0,
    "Education_Diploma"                   : 0,
    "Education_Primary"                   : 0,
    "Education_Secondary"                 : 1,
    "Gender_Male"                         : 1,
    "Age bracket_36-45"                   : 1,
    "Age bracket_46-55"                   : 0,
    "Age bracket_56-65"                   : 0,
    "Age bracket_above 65"                : 0,
    "Main credit source_Family"           : 0,
    "Main credit source_Savings"          : 1,
    "Farm records_Yes"                    : 1,
    "Main advisory source_Internet"       : 0,
    "Main advisory source_Public gatherings": 0,
    "Main advisory source_Radio"          : 1,
    "Main advisory source_Television"     : 0,
    "Extension provider_KALRO"            : 0,
    "Extension provider_National Government": 0,
    "Extension provider_Private Provider" : 1,
    "Advisory format_SMS text"            : 1,
    "Advisory language_Kiswahili"         : 1,
    "Advisory language_Vernacular"        : 0,
}

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "random_forest_corn_yield.pkl")
_model = None


def _load_model():
    """Load model sekali, reuse setelahnya."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model tidak ditemukan di '{MODEL_PATH}'. "
                "Jalankan modelling.py terlebih dahulu untuk melatih dan menyimpan model."
            )
        logger.info(f"Memuat model dari: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model berhasil dimuat.")
    return _model


# ── Feature builder ───────────────────────────────────────────────────────────
def _build_features(data: dict) -> pd.DataFrame:
    """
    Terima dict input (bisa parsial), lengkapi dengan default,
    kembalikan DataFrame 1 baris dengan urutan kolom yang benar.
    """
    row = {col: data.get(col, FEATURE_DEFAULTS[col]) for col in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ── Public API ────────────────────────────────────────────────────────────────
def predict_yield(data: dict) -> dict:
    """
    Prediksi hasil panen jagung (Yield, normalized 0-1).

    Parameters
    ----------
    data : dict
        Key = nama fitur (case-sensitive, sesuai FEATURE_COLS).
        Fitur yang tidak disertakan akan diisi nilai default.

    Returns
    -------
    dict dengan key:
        - yield_normalized : float (0.0 – 1.0)
        - yield_category   : str  ('rendah' | 'sedang' | 'tinggi')
        - input_features   : dict (fitur yang digunakan)
    """
    model  = _load_model()
    X      = _build_features(data)
    y_pred = float(model.predict(X)[0])
    y_clip = float(np.clip(y_pred, 0.0, 1.0))

    if y_clip < 0.33:
        kategori = "rendah"
    elif y_clip < 0.66:
        kategori = "sedang"
    else:
        kategori = "tinggi"

    logger.info(f"Prediksi yield: {y_clip:.4f} ({kategori})")
    return {
        "yield_normalized" : round(y_clip, 4),
        "yield_category"   : kategori,
        "input_features"   : X.iloc[0].to_dict(),
    }


# ── Test / standalone run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Contoh: petani dengan lahan 0.2 dan pupuk 0.1
    sample_input = {
        "Acreage"          : 0.2,
        "Fertilizer amount": 0.1,
        "Laborers"         : 0.1667,
        "Household size"   : 0.75,
        "Latitude"         : 0.5897,
        "Longitude"        : 0.2778,
        "Gender_Male"      : 1,
        "Farm records_Yes" : 1,
    }

    result = predict_yield(sample_input)
    print("\n── Hasil Prediksi ──")
    print(f"  Yield (normalized) : {result['yield_normalized']}")
    print(f"  Kategori           : {result['yield_category']}")
