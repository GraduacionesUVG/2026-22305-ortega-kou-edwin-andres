"""
Corre el pipeline real (load_and_clean -> train_test_split_by_flow ->
scale_features) contra data/raw. Se omite automáticamente si no hay CSVs
locales (data/raw está en .gitignore; cada integrante descarga su propia
copia). Automatiza la verificación manual descrita en
docs/adr/0001-split-cronologico-por-bloques-temporales.md.
"""
from pathlib import Path

import pytest

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
_tiene_csvs = RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.rglob("*.csv"))

pytestmark = pytest.mark.skipif(
    not _tiene_csvs,
    reason="data/raw no tiene CSVs localmente (dataset gitignored) — se omite",
)


def test_full_pipeline_end_to_end():
    from src.preprocessing.clean import load_and_clean, scale_features, train_test_split_by_flow

    df = load_and_clean("data/raw")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_flow(df)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # Criterios de aceptación del ADR 0001: ningún bloque vacío, ningún split sin ataques
    assert X_train["bloque"].nunique() == 5
    for y_split in (y_train, y_val, y_test):
        assert (y_split == 1).sum() > 0

    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert "split" not in X_train.columns
    assert not X_train_s[list(scaler.feature_names_in_)].isna().any().any()
