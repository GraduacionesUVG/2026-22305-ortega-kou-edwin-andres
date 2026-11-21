"""
Tests unitarios para train_test_split_by_flow() y scale_features()
(src/preprocessing/clean.py). Usan una fixture sintética en vez del dataset
real (multi-GB, gitignored) para correr en segundos en cualquier entorno.

outlier_quantile=0.0 en los tests (en vez del default 0.01) hace que las
fronteras de bloque sean exactamente el min/max de Timestamp por clase de
ataque, sin interpolación de cuantiles que razonar a mano.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing.clean import (
    COLUMNA_TIMESTAMP,
    IP_NAT_ATACANTE,
    IP_VICTIMA,
    excluir_respuestas_de_ataque,
    scale_features,
    train_test_split_by_flow,
)

BASE_TS = pd.Timestamp("2017-07-05 08:00:00")


def _bloque(start_minute, n, label):
    return pd.DataFrame({
        "Timestamp": [BASE_TS + pd.Timedelta(minutes=m) for m in range(start_minute, start_minute + n)],
        "Label": label,
    })


def _benigno_intercalado(minutos):
    # segundos=30 para no chocar con las filas de ataque, que ocupan el :00 de cada minuto
    return pd.DataFrame({
        "Timestamp": [BASE_TS + pd.Timedelta(minutes=m, seconds=30) for m in minutos],
        "Label": "BENIGN",
    })


@pytest.fixture
def synthetic_df():
    """110 filas sintéticas que reproducen los 5 bloques de train_test_split_by_flow()
    con fronteras exactas (min/max por clase, con outlier_quantile=0.0 en los tests).

    Layout (minutos desde BASE_TS):
      0-19  bloque 1: BENIGN (20 filas)
      20-39 bloque 2: DoS slowloris (20) + BENIGN intercalada (5) = 25 filas
      40-59 bloque 3: BENIGN (20 filas)
      60-79 bloque 4: DoS Slowhttptest (20) + BENIGN intercalada (5) = 25 filas
      80-99 bloque 5: BENIGN (20 filas)
    """
    b1 = _bloque(0, 20, "BENIGN")
    slowloris = _bloque(20, 20, "DoS slowloris")
    b2_benigno = _benigno_intercalado([21, 25, 30, 35, 38])
    b3 = _bloque(40, 20, "BENIGN")
    slowhttp = _bloque(60, 20, "DoS Slowhttptest")
    b4_benigno = _benigno_intercalado([61, 65, 70, 75, 78])
    b5 = _bloque(80, 20, "BENIGN")

    df = pd.concat(
        [b1, slowloris, b2_benigno, b3, slowhttp, b4_benigno, b5], ignore_index=True
    )
    df["label_binaria"] = (df["Label"] != "BENIGN").astype(int)

    rng = np.random.default_rng(42)
    df["Feature1"] = rng.normal(loc=0.0, scale=1.0, size=len(df))
    df["Feature2"] = rng.normal(loc=100.0, scale=10.0, size=len(df))

    # Desordenado a propósito: confirma que la función ordena por Timestamp
    # internamente en vez de depender del orden de entrada.
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


EXPECTED_BLOCK_COUNTS = {1: 20, 2: 25, 3: 20, 4: 25, 5: 20}
EXPECTED_SPLIT_SIZES = {
    1: {"train": 14, "val": 3, "test": 3},
    2: {"train": 18, "val": 4, "test": 3},
    3: {"train": 14, "val": 3, "test": 3},
    4: {"train": 18, "val": 4, "test": 3},
    5: {"train": 14, "val": 3, "test": 3},
}


# ---------------------------------------------------------------------------
# train_test_split_by_flow()
# ---------------------------------------------------------------------------

def test_block_assignment_matches_expected_boundaries(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)
    X_all = pd.concat([X_train, X_val, X_test])

    assert X_all["bloque"].value_counts().sort_index().to_dict() == EXPECTED_BLOCK_COUNTS

    def bloque_en_minuto(m):
        fila = X_all.loc[X_all[COLUMNA_TIMESTAMP] == BASE_TS + pd.Timedelta(minutes=m)]
        return fila["bloque"].iloc[0]

    assert bloque_en_minuto(19) == 1   # último minuto antes de slowloris
    assert bloque_en_minuto(20) == 2   # frontera inclusive: inicio de slowloris
    assert bloque_en_minuto(39) == 2   # frontera inclusive: fin de slowloris
    assert bloque_en_minuto(40) == 3   # primer minuto del hueco
    assert bloque_en_minuto(60) == 4   # frontera inclusive: inicio de slowhttptest
    assert bloque_en_minuto(79) == 4   # frontera inclusive: fin de slowhttptest
    assert bloque_en_minuto(80) == 5   # primer minuto después de slowhttptest


def test_split_excludes_split_column_but_keeps_metadata(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)

    for X in (X_train, X_val, X_test):
        assert "split" not in X.columns
        assert "label_binaria" not in X.columns
        assert "Label" not in X.columns
        assert "bloque" in X.columns
        assert COLUMNA_TIMESTAMP in X.columns


def test_positional_cut_sizes_and_chronological_order(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)

    for b, esperado in EXPECTED_SPLIT_SIZES.items():
        n_train = int((X_train["bloque"] == b).sum())
        n_val = int((X_val["bloque"] == b).sum())
        n_test = int((X_test["bloque"] == b).sum())
        assert (n_train, n_val, n_test) == (esperado["train"], esperado["val"], esperado["test"])

        max_train_ts = X_train.loc[X_train["bloque"] == b, COLUMNA_TIMESTAMP].max()
        min_val_ts = X_val.loc[X_val["bloque"] == b, COLUMNA_TIMESTAMP].min()
        max_val_ts = X_val.loc[X_val["bloque"] == b, COLUMNA_TIMESTAMP].max()
        min_test_ts = X_test.loc[X_test["bloque"] == b, COLUMNA_TIMESTAMP].min()
        assert max_train_ts <= min_val_ts
        assert max_val_ts <= min_test_ts


def test_output_is_six_tuple_with_consistent_shapes(synthetic_df):
    resultado = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)
    assert len(resultado) == 6
    X_train, X_val, X_test, y_train, y_val, y_test = resultado

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_val) + len(X_test) == len(synthetic_df)
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)


def test_random_state_has_no_effect(synthetic_df):
    out_a = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0, random_state=1)
    out_b = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0, random_state=999)
    for a, b in zip(out_a, out_b):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        else:
            pd.testing.assert_series_equal(a, b)


def test_raises_valueerror_when_timestamp_missing(synthetic_df):
    with pytest.raises(ValueError, match="Timestamp"):
        train_test_split_by_flow(synthetic_df.drop(columns=[COLUMNA_TIMESTAMP]))


def test_raises_valueerror_when_no_attack_rows(synthetic_df):
    df_solo_benigno = synthetic_df[synthetic_df["Label"] == "BENIGN"].copy()
    with pytest.raises(ValueError, match="slowloris"):
        train_test_split_by_flow(df_solo_benigno)


def test_stable_sort_preserves_order_of_duplicate_timestamps():
    empatadas = pd.DataFrame({
        "Timestamp": [BASE_TS, BASE_TS, BASE_TS],
        "Label": "BENIGN",
        "orden_original": [0, 1, 2],
    })
    resto = pd.DataFrame({
        "Timestamp": [BASE_TS + pd.Timedelta(minutes=m) for m in range(1, 40)],
        "Label": ["DoS slowloris"] * 20 + ["DoS Slowhttptest"] * 19,
        "orden_original": range(3, 42),
    })
    df = pd.concat([empatadas, resto], ignore_index=True)
    df["label_binaria"] = (df["Label"] != "BENIGN").astype(int)

    X_train, X_val, X_test, *_ = train_test_split_by_flow(df, outlier_quantile=0.0)
    X_all = pd.concat([X_train, X_val, X_test])
    empatadas_resultado = X_all[X_all["bloque"] == 1].sort_index()
    assert list(empatadas_resultado["orden_original"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# scale_features()
# ---------------------------------------------------------------------------

def test_scale_features_fits_only_on_train(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)
    X_val, X_test = X_val.copy(), X_test.copy()
    X_val.loc[X_val.index[0], "Feature1"] = 10_000.0
    X_test.loc[X_test.index[0], "Feature1"] = -10_000.0

    X_train_scaled, _, _, _ = scale_features(X_train, X_val, X_test)

    columnas = [c for c in X_train.columns if c not in (COLUMNA_TIMESTAMP, "bloque")]
    esperado = StandardScaler().fit(X_train[columnas]).transform(X_train[columnas])
    np.testing.assert_allclose(X_train_scaled[columnas].to_numpy(), esperado)


def test_default_columns_exclude_timestamp_and_bloque(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)
    X_train_scaled, _, _, scaler = scale_features(X_train, X_val, X_test)

    pd.testing.assert_series_equal(X_train_scaled[COLUMNA_TIMESTAMP], X_train[COLUMNA_TIMESTAMP])
    pd.testing.assert_series_equal(X_train_scaled["bloque"], X_train["bloque"])
    assert COLUMNA_TIMESTAMP not in scaler.feature_names_in_
    assert "bloque" not in scaler.feature_names_in_


def test_explicit_columns_to_scale_respected(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)
    X_train_scaled, _, _, scaler = scale_features(X_train, X_val, X_test, columns_to_scale=["Feature1"])

    assert abs(X_train_scaled["Feature1"].mean()) < 1e-8
    pd.testing.assert_series_equal(X_train_scaled["Feature2"], X_train["Feature2"])
    assert list(scaler.feature_names_in_) == ["Feature1"]


def test_scaler_reusable_on_new_data(synthetic_df):
    X_train, X_val, X_test, *_ = train_test_split_by_flow(synthetic_df, outlier_quantile=0.0)
    X_train_scaled, _, _, scaler = scale_features(X_train, X_val, X_test)

    columnas = list(scaler.feature_names_in_)
    transformado_directo = scaler.transform(X_train[columnas].iloc[[0]])[0]
    np.testing.assert_allclose(transformado_directo, X_train_scaled[columnas].iloc[0].to_numpy())


# --- excluir_respuestas_de_ataque() ---

def test_excluir_respuestas_de_ataque_solo_quita_benign_del_par():
    otra_ip = "192.168.10.3"
    df = pd.DataFrame({
        "Source IP": [IP_VICTIMA, IP_NAT_ATACANTE, f" {IP_VICTIMA} ", IP_NAT_ATACANTE, IP_VICTIMA, otra_ip],
        "Destination IP": [IP_NAT_ATACANTE, IP_VICTIMA, IP_NAT_ATACANTE, IP_VICTIMA, otra_ip, IP_VICTIMA],
        "Label": ["BENIGN", "BENIGN", "BENIGN", "DoS slowloris", "BENIGN", "BENIGN"],
    })

    resultado = excluir_respuestas_de_ataque(df)

    # Se van los BENIGN del par en ambos sentidos (también con espacios en la IP);
    # se quedan el ataque del par y los BENIGN de la víctima con otras IPs
    assert resultado.index.tolist() == [3, 4, 5]
    assert resultado["Label"].tolist() == ["DoS slowloris", "BENIGN", "BENIGN"]


def test_excluir_respuestas_de_ataque_raises_valueerror_sin_columnas_de_ip():
    df = pd.DataFrame({"Label": ["BENIGN", "DoS slowloris"]})
    with pytest.raises(ValueError, match="basic_cleaning"):
        excluir_respuestas_de_ataque(df)
