"""
Métricas de evaluación estandarizadas para los tres modelos (GRU, SVM, XGBoost).

Usar este módulo desde los tres notebooks garantiza que la comparación en la
Fase 8 del cronograma sea justa: mismas métricas, mismo formato de salida.

Uso típico:

    from src.evaluation.metrics import evaluate_model

    resultados = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,          # probabilidades de la clase positiva (ataque)
        model_name="GRU",
        training_time_seconds=123.4,
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "modelo",
    training_time_seconds: Optional[float] = None,
    save_dir: str = "reports/metrics",
) -> dict:
    """
    Calcula el set de métricas estándar del proyecto y opcionalmente lo guarda
    como JSON en reports/metrics/<model_name>.json

    Incluye F2 (`f2_score`, F-beta con beta=2), que pondera el recall cuatro
    veces más que la precision; es el objetivo con el que los tres modelos eligen
    su umbral de decisión.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales (0 = benigno, 1 = ataque).
    y_pred : array-like
        Etiquetas predichas por el modelo (0/1).
    y_proba : array-like, opcional
        Probabilidad de la clase positiva (ataque). Necesario para ROC-AUC.
    model_name : str
        Nombre identificador del modelo ("GRU", "SVM", "XGBoost").
    training_time_seconds : float, opcional
        Tiempo de entrenamiento en segundos, para comparar viabilidad práctica.
    save_dir : str
        Carpeta donde se guarda el JSON de resultados.

    Returns
    -------
    dict con todas las métricas calculadas.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    resultados = {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "f2_score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "matriz_confusion": {
            "verdaderos_negativos": int(tn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdaderos_positivos": int(tp),
        },
        "tasa_falsos_negativos": fn / (fn + tp) if (fn + tp) > 0 else None,
        "tiempo_entrenamiento_segundos": training_time_seconds,
    }

    if y_proba is not None:
        try:
            resultados["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            resultados["roc_auc"] = None
            resultados["roc_auc_error"] = str(e)
    else:
        resultados["roc_auc"] = None

    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name.lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"Métricas guardadas en: {out_path}")

    _print_resumen(resultados)
    return resultados


def _print_resumen(resultados: dict) -> None:
    print(f"\n=== Resultados: {resultados['modelo']} ===")
    print(f"Accuracy:  {resultados['accuracy']:.4f}")
    print(f"Precision: {resultados['precision']:.4f}")
    print(f"Recall:    {resultados['recall']:.4f}  <- métrica principal")
    print(f"F1-score:  {resultados['f1_score']:.4f}")
    print(f"F2-score:  {resultados['f2_score']:.4f}")
    if resultados.get("roc_auc") is not None:
        print(f"ROC-AUC:   {resultados['roc_auc']:.4f}")
    mc = resultados["matriz_confusion"]
    print(
        f"Matriz de confusión -> TN: {mc['verdaderos_negativos']}, "
        f"FP: {mc['falsos_positivos']}, FN: {mc['falsos_negativos']}, "
        f"TP: {mc['verdaderos_positivos']}"
    )
    if resultados.get("tiempo_entrenamiento_segundos") is not None:
        print(f"Tiempo de entrenamiento: {resultados['tiempo_entrenamiento_segundos']:.2f}s")


class Timer:
    """
    Context manager simple para medir tiempo de entrenamiento.

    Uso:
        with Timer() as t:
            modelo.fit(X_train, y_train)
        tiempo = t.elapsed
    """

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
