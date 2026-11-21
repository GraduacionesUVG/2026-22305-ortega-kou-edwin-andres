"""
Modelo final XGBoost: hiperparámetros, umbral de decisión, entrenamiento,
guardado/carga y predicción.

Funciones reutilizables desde notebook o script: importar este módulo no carga
datos, no entrena ni escribe archivos por sí solo. Los valores finales salen de
notebooks/xgboost/03_tuning.ipynb (hiperparámetros, elegidos con su regla de
selección sobre val) y notebooks/xgboost/04_umbral.ipynb (umbral, mediana de
los umbrales ganadores en 1,000 remuestreos bootstrap de val). El modelo se
entrena con el train real y scale_pos_weight = benignos / ataques de y_train
(variante C de notebooks/xgboost/02b_ablacion_smote.ipynb), con early stopping
en val. scale_pos_weight no es una constante: se calcula de y_train con
calcular_scale_pos_weight() cada vez que se entrena.

Los archivos del modelo entrenado van fuera de src/, en models/xgboost/ en la
raíz del repo (misma convención que models/gru/ y models/svm/):

    models/xgboost/xgboost_final.json            # modelo, formato JSON de XGBoost
    models/xgboost/xgboost_final_metadatos.json  # umbral, hiperparámetros, scale_pos_weight
                                                 # (con sus conteos de y_train), best_iteration

Uso típico:
    from src.models.xgboost.model import (
        train_xgboost_model, save_xgboost_model, load_xgboost_model,
        predict_xgboost_proba, apply_threshold,
    )

    modelo, segundos = train_xgboost_model(X_train, y_train, X_val, y_val)
    save_xgboost_model(modelo, y_train=y_train)

    modelo, metadatos = load_xgboost_model()
    y_proba = predict_xgboost_proba(modelo, X)
    y_pred = apply_threshold(y_proba)   # UMBRAL_FINAL por defecto
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import Timer

# Combinación elegida en 03_tuning.ipynb (dataset corregido, 443,614 filas); el
# resto es la configuración fija de 02/03. scale_pos_weight no va acá porque no
# es un valor fijo: lo calcula calcular_scale_pos_weight() a partir de y_train.
HIPERPARAMETROS = {
    "learning_rate": 0.3,
    "max_depth": 4,
    "min_child_weight": 1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# Elegido en 04_umbral.ipynb: entre los umbrales con recall >= 0.75 en val, el
# de mayor F2 (beta = 2); umbral final = mediana sobre 1,000 remuestreos
# bootstrap de val.
UMBRAL_FINAL = 0.87

# src/models/xgboost/model.py -> parents[3] es la raíz del repo.
RAIZ_REPO = Path(__file__).resolve().parents[3]
MODEL_DIR = RAIZ_REPO / "models" / "xgboost"
ARCHIVO_MODELO = "xgboost_final.json"
ARCHIVO_METADATOS = "xgboost_final_metadatos.json"


def gpu_disponible_para_xgboost() -> bool:
    """
    Indica si XGBoost puede entrenar en GPU: el build tiene CUDA y un fit de
    prueba chico con device="cuda" no falla. Misma verificación que en
    02_modelo.ipynb y 03_tuning.ipynb.
    """
    if not xgb.build_info().get("USE_CUDA", False):
        return False
    try:
        rng = np.random.default_rng(0)
        X_prueba = rng.normal(size=(200, 5))
        y_prueba = (X_prueba[:, 0] > 0).astype(int)
        xgb.XGBClassifier(n_estimators=2, tree_method="hist", device="cuda").fit(X_prueba, y_prueba)
        return True
    except xgb.core.XGBoostError:
        return False


def calcular_scale_pos_weight(y_train: np.ndarray) -> dict:
    """
    scale_pos_weight = benignos / ataques, contados en y_train (el train real,
    sin filas sintéticas). Es la fórmula de la variante C de
    02b_ablacion_smote.ipynb.

    Parameters
    ----------
    y_train : np.ndarray
        Etiquetas binarias de train (0 = benigno, 1 = ataque).

    Returns
    -------
    dict
        {"scale_pos_weight": float, "n_benignos_train": int, "n_ataques_train": int}
    """
    y_train = np.asarray(y_train)
    n_benignos = int((y_train == 0).sum())
    n_ataques = int((y_train == 1).sum())
    if n_benignos + n_ataques != len(y_train):
        raise ValueError("y_train tiene valores distintos de 0 y 1")
    if n_ataques == 0:
        raise ValueError("y_train no tiene ataques: no se puede calcular scale_pos_weight")
    return {
        "scale_pos_weight": n_benignos / n_ataques,
        "n_benignos_train": n_benignos,
        "n_ataques_train": n_ataques,
    }


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    device: str | None = None,
) -> tuple[xgb.XGBClassifier, float]:
    """
    Entrena el modelo final con X_train completo y early stopping en val
    (eval_set=[(X_val, y_val)], eval_metric="aucpr", 50 rondas sin mejora),
    con HIPERPARAMETROS y scale_pos_weight calculado de y_train con
    calcular_scale_pos_weight().

    Al terminar, el modelo se pasa a CPU (set_params(device="cpu")): los datos
    a predecir están en memoria de CPU, y si el booster sigue en cuda XGBoost
    avisa que los dispositivos no coinciden y predice por un camino más lento.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, np.ndarray
        Train real (sin filas sintéticas), con las mismas columnas que X_val.
    X_val, y_val : pd.DataFrame, np.ndarray
        Validación, usada solo para el early stopping.
    device : str, opcional
        "cuda" o "cpu". Si es None, usa cuda cuando gpu_disponible_para_xgboost()
        lo permite y cpu si no.

    Returns
    -------
    tuple[xgb.XGBClassifier, float]
        El modelo entrenado (ya en cpu) y el tiempo de entrenamiento en segundos.
    """
    if device is None:
        device = "cuda" if gpu_disponible_para_xgboost() else "cpu"
    scale_pos_weight = calcular_scale_pos_weight(y_train)["scale_pos_weight"]
    modelo = xgb.XGBClassifier(**HIPERPARAMETROS, scale_pos_weight=scale_pos_weight, device=device)
    with Timer() as t:
        modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    modelo.set_params(device="cpu")
    return modelo, t.elapsed


def save_xgboost_model(
    model: xgb.XGBClassifier,
    model_dir: str | Path = MODEL_DIR,
    threshold: float = UMBRAL_FINAL,
    *,
    y_train: np.ndarray,
) -> tuple[Path, Path]:
    """
    Guarda el modelo en formato JSON de XGBoost y, al lado, un JSON con el
    umbral, los hiperparámetros, scale_pos_weight con los conteos de y_train de
    los que sale, y best_iteration. Si los archivos ya existen, se sobrescriben.

    Los hiperparámetros y scale_pos_weight se leen del modelo (get_params()),
    no de HIPERPARAMETROS, para que el JSON describa el modelo que se guardó.
    Antes de escribir nada se verifica que el scale_pos_weight del modelo sea
    exactamente benignos / ataques de y_train.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Modelo entrenado con train_xgboost_model().
    model_dir : str | Path
        Carpeta de salida (default models/xgboost/ en la raíz del repo). Se
        crea si no existe.
    threshold : float
        Umbral de decisión que acompaña al modelo (default UMBRAL_FINAL).
    y_train : np.ndarray
        Las mismas etiquetas de train con las que se entrenó el modelo. Se usan
        para guardar los conteos de benignos y ataques en los metadatos.

    Returns
    -------
    tuple[Path, Path]
        Rutas del modelo y del JSON de metadatos.
    """
    parametros = model.get_params()
    pesos = calcular_scale_pos_weight(y_train)
    if parametros.get("scale_pos_weight") != pesos["scale_pos_weight"]:
        raise ValueError(
            f"scale_pos_weight del modelo ({parametros.get('scale_pos_weight')}) != benignos / ataques de "
            f"y_train ({pesos['n_benignos_train']} / {pesos['n_ataques_train']} = {pesos['scale_pos_weight']})"
        )

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ruta_modelo = model_dir / ARCHIVO_MODELO
    ruta_metadatos = model_dir / ARCHIVO_METADATOS

    model.save_model(ruta_modelo)

    metadatos = {
        "umbral": float(threshold),
        "hiperparametros": {nombre: parametros[nombre] for nombre in HIPERPARAMETROS},
        **pesos,
        "best_iteration": int(model.best_iteration),
    }
    ruta_metadatos.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta_modelo, ruta_metadatos


def load_xgboost_model(model_dir: str | Path = MODEL_DIR) -> tuple[xgb.XGBClassifier, dict]:
    """
    Carga el modelo y sus metadatos guardados con save_xgboost_model().

    El JSON de XGBoost conserva best_iteration, así que predict_proba() del
    modelo cargado usa los mismos árboles que el modelo original (los primeros
    best_iteration + 1).

    Parameters
    ----------
    model_dir : str | Path
        Carpeta donde están los archivos (default models/xgboost/ en la raíz
        del repo).

    Returns
    -------
    tuple[xgb.XGBClassifier, dict]
        El modelo y el dict de metadatos (umbral, hiperparametros, scale_pos_weight,
        n_benignos_train, n_ataques_train, best_iteration).
    """
    model_dir = Path(model_dir)
    modelo = xgb.XGBClassifier()
    modelo.load_model(model_dir / ARCHIVO_MODELO)
    metadatos = json.loads((model_dir / ARCHIVO_METADATOS).read_text(encoding="utf-8"))
    return modelo, metadatos


def predict_xgboost_proba(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Probabilidad de ataque (clase 1) por flujo, en float64.

    Se pasa a float64 para que la comparación contra el umbral use su valor
    exacto, igual que en 04_umbral.ipynb: umbrales como 0.87 no son
    representables en float32, y NumPy compara un array float32 contra un float
    de Python en float32.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Modelo entrenado o cargado.
    X : pd.DataFrame
        Features con las mismas columnas, en el mismo orden, que X_train.

    Returns
    -------
    np.ndarray
        Array 1D de probabilidades, float64.
    """
    return model.predict_proba(X)[:, 1].astype(np.float64)


def apply_threshold(y_proba: np.ndarray, threshold: float = UMBRAL_FINAL) -> np.ndarray:
    """
    Clasifica como ataque (1) los flujos con probabilidad >= threshold, y el
    resto como benignos (0).

    Parameters
    ----------
    y_proba : np.ndarray
        Salida de predict_xgboost_proba().
    threshold : float
        Umbral de decisión (default UMBRAL_FINAL).

    Returns
    -------
    np.ndarray
        Array 1D de 0/1 (int).
    """
    return (np.asarray(y_proba, dtype=np.float64) >= threshold).astype(int)
