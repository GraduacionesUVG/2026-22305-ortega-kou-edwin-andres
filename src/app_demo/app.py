"""
Demo interactiva (Streamlit) del módulo XGBoost para la detección de ataques
Low & Slow (Slowloris / Slowhttptest) en CIC-IDS2017.

Adaptación de la demo compartida del equipo que conserva solo el modelo
XGBoost. Carga el modelo congelado del repositorio
(src/models/xgboost/xgboost_final.json) y sus metadatos (umbral 0.87 y
best_iteration). No entrena ni modifica el modelo: solo hace inferencia.

Ejecución (desde src/app_demo/):
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# Carpeta de la app (src/app_demo/) y carpeta del modelo congelado del repo
# (src/models/xgboost/). Todas las rutas son relativas a este archivo.
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "xgboost"

MODELO = "XGBoost"
DATA_DIR = APP_DIR / "Data_Test" / "xgboost"
RUTA_MODELO = MODEL_DIR / "xgboost_final.json"
RUTA_METADATOS = MODEL_DIR / "xgboost_final_metadatos.json"

# Valor de respaldo si no se pueden leer los metadatos
UMBRAL_POR_DEFECTO = 0.87

st.set_page_config(
    page_title="Low & Slow IDS | Live Demo",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background: rgba(127,127,127,.07);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file.getvalue()))


# ============================================================
# CARGA DEL MODELO Y SUS METADATOS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(path: str) -> xgb.Booster:
    """Carga el Booster nativo de XGBoost (.json o .ubj)."""

    p = Path(path)

    if not p.is_file():
        raise FileNotFoundError(
            f"No existe el modelo: {p}"
        )

    if p.suffix.lower() not in (".json", ".ubj"):
        raise ValueError(
            f"Formato de modelo no admitido: {p.suffix}"
        )

    booster = xgb.Booster()
    booster.load_model(str(p))

    return booster


def leer_metadatos() -> dict:
    """Lee xgboost_final_metadatos.json; devuelve {} si no está disponible."""

    try:
        with open(RUTA_METADATOS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def umbral_xgboost_guardado() -> float:
    """Umbral de decisión elegido en validación (0.87)."""

    try:
        return float(leer_metadatos()["umbral"])
    except (KeyError, TypeError, ValueError):
        return UMBRAL_POR_DEFECTO


def best_iteration_guardada(model: xgb.Booster) -> int:
    """
    Mejor iteración del early stopping. Se toma de los metadatos y, si no
    está, del atributo guardado en el propio Booster.
    """

    metadatos = leer_metadatos()

    if "best_iteration" in metadatos:
        return int(metadatos["best_iteration"])

    return int(model.attributes().get("best_iteration", 0))


# ============================================================
# ETIQUETAS
# ============================================================

def extract_labels(y_frame: pd.DataFrame) -> np.ndarray:

    if y_frame.shape[1] != 1:
        raise ValueError(
            "y_test.csv debe contener una sola columna."
        )

    y = y_frame.iloc[:, 0]

    if y.isna().any():
        raise ValueError(
            "Las etiquetas contienen valores nulos."
        )

    if pd.api.types.is_numeric_dtype(y):

        values = y.to_numpy()

        if not np.isin(values, [0, 1]).all():
            raise ValueError(
                "Las etiquetas deben ser binarias (0/1)."
            )

        return values.astype(int)

    normalized = y.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0,
        "benign": 0,
        "benigno": 0,
        "normal": 0,
        "1": 1,
        "attack": 1,
        "ataque": 1,
        "slowloris": 1,
        "slowhttptest": 1,
        "slowhttp": 1,
    }

    if not normalized.isin(mapping).all():

        bad = normalized[
            ~normalized.isin(mapping)
        ].unique()[:5]

        raise ValueError(
            f"Etiquetas no reconocidas: {bad.tolist()}"
        )

    return normalized.map(mapping).to_numpy(dtype=int)


# ============================================================
# PREPARACIÓN DE CARACTERÍSTICAS
# ============================================================

def prepare_features(
    frame: pd.DataFrame,
    model: xgb.Booster,
) -> pd.DataFrame:
    """
    Valida X_test y lo deja con las 55 características en el mismo orden
    con el que se entrenó el modelo.
    """

    x = frame.copy()

    # Eliminar índices exportados por pandas
    x = x.loc[
        :,
        ~x.columns.astype(str).str.match(r"^Unnamed: \d+$")
    ]

    if x.columns.duplicated().any():
        raise ValueError(
            "X_test contiene columnas duplicadas."
        )

    expected = model.feature_names

    if expected:

        missing = [
            col for col in expected
            if col not in x.columns
        ]

        if missing:
            raise ValueError(
                f"Faltan características: {missing[:12]}"
            )

        # Respetar exactamente el orden del entrenamiento
        x = x.loc[:, list(expected)]

    non_numeric = x.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Características no numéricas: {non_numeric[:10]}"
        )

    if not np.isfinite(
        x.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "X_test contiene NaN o infinitos."
        )

    if x.shape[1] != model.num_features():
        raise ValueError(
            f"El modelo espera {model.num_features()} columnas, "
            f"pero recibió {x.shape[1]}."
        )

    return x


# ============================================================
# PREDICCIONES XGBOOST
# ============================================================

def predict_xgb(
    model: xgb.Booster,
    x: pd.DataFrame,
    threshold: float,
    best_iteration: int,
):
    """
    Probabilidad de ataque por flujo usando solo los árboles hasta
    best_iteration (igual que en el pipeline) y clase según el umbral.
    """

    scores = np.asarray(
        model.predict(
            xgb.DMatrix(
                x,
                feature_names=list(x.columns)
            ),
            iteration_range=(0, best_iteration + 1)
        )
    )

    if scores.ndim != 1 or len(scores) != len(x):
        raise ValueError(
            "XGBoost no devolvió una puntuación por flujo."
        )

    if (
        not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError(
            "XGBoost no devolvió probabilidades válidas."
        )

    predictions = (
        scores >= threshold
    ).astype(int)

    return predictions, scores


# ============================================================
# MÉTRICAS
# ============================================================

def evaluate(y_true, y_pred, scores):

    result = {
        "Accuracy": accuracy_score(
            y_true, y_pred
        ),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }

    if len(np.unique(y_true)) == 2:

        result["ROC-AUC"] = roc_auc_score(
            y_true, scores
        )

        result["AP"] = average_precision_score(
            y_true, scores
        )

    else:

        result["ROC-AUC"] = float("nan")
        result["AP"] = float("nan")

    return result


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def draw_confusion(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4)
    )

    ax.imshow(
        cm,
        cmap="Blues"
    )

    for (i, j), value in np.ndenumerate(cm):

        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=14,
            color=(
                "white"
                if value > cm.max() / 2
                else "black"
            )
        )

    ax.set_xticks(
        [0, 1],
        ["Benigno", "Ataque"]
    )

    ax.set_yticks(
        [0, 1],
        ["Benigno", "Ataque"]
    )

    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusion")

    fig.tight_layout()

    return fig, cm


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("Detección de ataques Low & Slow")

st.caption(
    "Demostración interactiva de inferencia sobre "
    "flujos de red · XGBoost"
)


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.header("Configuración")

    # Solo XGBoost; se mantiene el selector para conservar el diseño original
    selected = st.selectbox(
        "Modelo",
        [MODELO]
    )

    model_path = st.text_input(
        "Ruta del modelo",
        str(RUTA_MODELO)
    )

    st.divider()

    source = st.radio(
        "Origen de datos",
        ["Archivos locales", "Subir CSV"]
    )

    if source == "Archivos locales":

        x_path = st.text_input(
            "Ruta X_test.csv",
            str(DATA_DIR / "X_test.csv"),
            key="x_path_XGBoost"
        )

        y_path = st.text_input(
            "Ruta y_test.csv",
            str(DATA_DIR / "y_test.csv"),
            key="y_path_XGBoost"
        )

        x_upload = None
        y_upload = None

    else:

        x_upload = st.file_uploader(
            "X_test.csv",
            type="csv",
            key="x"
        )

        y_upload = st.file_uploader(
            "y_test.csv (opcional)",
            type="csv",
            key="y"
        )

        x_path = ""
        y_path = ""

    st.divider()

    threshold = st.number_input(
        "Umbral de probabilidad",
        min_value=0.0,
        max_value=1.0,
        value=umbral_xgboost_guardado(),
        step=0.01
    )

    st.caption(
        "Comprueba que este umbral corresponde "
        "al modelo exportado."
    )

    run = st.button(
        "Ejecutar predicciones",
        type="primary",
        use_container_width=True
    )


# ============================================================
# INFORMACIÓN
# ============================================================

st.info(
    "Esta demostración realiza inferencia con modelos "
    "previamente entrenados. No realiza entrenamiento "
    "ni modifica los parámetros del modelo."
)


# ============================================================
# EJECUCIÓN DE INFERENCIA
# ============================================================

if run:

    try:

        with st.spinner(
            "Cargando datos y ejecutando inferencia..."
        ):

            # Cargar dataset
            if source == "Archivos locales":

                if not Path(x_path).is_file():
                    raise FileNotFoundError(
                        f"No existe X_test: {x_path}"
                    )

                x_df = read_csv(
                    str(Path(x_path).resolve())
                )

                y_df = (
                    read_csv(
                        str(Path(y_path).resolve())
                    )
                    if Path(y_path).is_file()
                    else None
                )

            else:

                if x_upload is None:
                    raise ValueError(
                        "Debes subir X_test.csv."
                    )

                x_df = csv_from_upload(x_upload)

                y_df = (
                    csv_from_upload(y_upload)
                    if y_upload
                    else None
                )

            # Cargar modelo y preparar características
            model = load_model(
                str(Path(model_path).resolve())
            )

            best_iteration = best_iteration_guardada(model)

            x = prepare_features(
                x_df,
                model
            )

            # Cargar etiquetas
            y = (
                extract_labels(y_df)
                if y_df is not None
                else None
            )

            if y is not None and len(y) != len(x):
                raise ValueError(
                    "X_test y y_test tienen tamaños diferentes: "
                    f"{len(x)} y {len(y)}."
                )

            # Inferencia
            started = time.perf_counter()

            preds, scores = predict_xgb(
                model,
                x,
                threshold,
                best_iteration
            )

            elapsed = time.perf_counter() - started

            row_indices = np.arange(len(x))

            # Resultados individuales
            results = pd.DataFrame({
                "fila_original": row_indices,
                "prediccion": preds,
                "clase_predicha": np.where(
                    preds == 1,
                    "Ataque",
                    "Benigno"
                ),
                "score": scores,
            })

            if y is not None:

                results["etiqueta_real"] = y

                results["correcto"] = (
                    preds == y
                )

            # Guardar resultados en sesión
            st.session_state["demo_results"] = {
                "model": selected,
                "predictions": results,
                "labels": y,
                "scores": scores,
                "elapsed": elapsed,
                "n_input": len(x),
                "threshold": threshold,
            }

    except Exception as exc:

        st.error(
            f"No se pudo ejecutar la inferencia: {exc}"
        )

        st.info(
            "Revisa el modelo, las dependencias, "
            "las características y el preprocesamiento."
        )


# ============================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================

if "demo_results" in st.session_state:

    data = st.session_state["demo_results"]

    result = data["predictions"]
    y_true = data["labels"]

    pred = result["prediccion"].to_numpy()
    scores = data["scores"]

    st.subheader(
        f"Resultados · {data['model']}"
    )

    st.caption(
        f"{len(result):,} flujos evaluados "
        f"de {data['n_input']:,} entradas · "
        f"Inferencia: {data['elapsed']:.3f} s · "
        f"Umbral: {data['threshold']:.4f}"
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    if y_true is not None:

        metrics = evaluate(
            y_true,
            pred,
            scores
        )

        cols = st.columns(6)

        for col, (name, value) in zip(
            cols,
            metrics.items()
        ):

            col.metric(
                name,
                (
                    f"{value:.2%}"
                    if np.isfinite(value)
                    else "N/D"
                )
            )

        st.divider()

        # ====================================================
        # MATRIZ Y DISTRIBUCIÓN
        # ====================================================

        left, right = st.columns(2)

        with left:

            fig, cm = draw_confusion(
                y_true,
                pred
            )

            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()

            st.caption(
                f"TP: {tp:,} · "
                f"TN: {tn:,} · "
                f"FP: {fp:,} · "
                f"FN: {fn:,}"
            )

        with right:

            st.markdown(
                "**Distribución de clases**"
            )

            distribution = pd.DataFrame(
                {
                    "Real": np.bincount(
                        y_true,
                        minlength=2
                    ),
                    "Predicho": np.bincount(
                        pred,
                        minlength=2
                    ),
                },
                index=["Benigno", "Ataque"]
            )

            st.bar_chart(
                distribution,
                x_label="Clase",
                y_label="Cantidad de flujos",
                stack=False
            )

            st.markdown(
                "**Errores de clasificación**"
            )

            errors = pd.DataFrame(
                {
                    "Cantidad": [fp, fn]
                },
                index=[
                    "Falsos positivos",
                    "Falsos negativos"
                ]
            )

            st.bar_chart(errors)

        st.divider()

        # ====================================================
        # CURVAS ROC Y PRECISION-RECALL
        # ====================================================

        if len(np.unique(y_true)) == 2:

            c1, c2 = st.columns(2)

            with c1:

                fpr, tpr, _ = roc_curve(
                    y_true,
                    scores
                )

                fig, ax = plt.subplots(
                    figsize=(5.5, 3.5)
                )

                ax.plot(
                    fpr,
                    tpr,
                    label=(
                        f"AUC = "
                        f"{metrics['ROC-AUC']:.4f}"
                    )
                )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    "--",
                    color="gray"
                )

                ax.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title="Curva ROC"
                )

                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

            with c2:

                precision, recall, _ = (
                    precision_recall_curve(
                        y_true,
                        scores
                    )
                )

                fig, ax = plt.subplots(
                    figsize=(5.5, 3.5)
                )

                ax.plot(
                    recall,
                    precision,
                    label=(
                        f"AP = "
                        f"{metrics['AP']:.4f}"
                    )
                )

                ax.set(
                    xlabel="Recall",
                    ylabel="Precision",
                    title="Curva Precision-Recall",
                    xlim=(0, 1),
                    ylim=(0, 1.05)
                )

                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

        else:

            st.warning(
                "Solo existe una clase en y_test. "
                "No se pueden calcular ROC-AUC ni AP."
            )

    # ========================================================
    # SIN ETIQUETAS
    # ========================================================

    else:

        st.warning(
            "No se proporcionó y_test.csv. "
            "Se muestran predicciones sin métricas."
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Benignos predichos",
            f"{(pred == 0).sum():,}"
        )

        c2.metric(
            "Ataques predichos",
            f"{(pred == 1).sum():,}"
        )

        st.bar_chart(
            pd.Series(
                np.bincount(
                    pred,
                    minlength=2
                ),
                index=["Benigno", "Ataque"]
            )
        )

    # ========================================================
    # PREDICCIONES INDIVIDUALES
    # ========================================================

    st.divider()

    st.subheader(
        "Predicciones individuales"
    )

    st.dataframe(
        result.head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar todas las predicciones (CSV)",
        result.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"predicciones_"
            f"{data['model'].lower()}.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# INSTRUCCIONES INICIALES
# ============================================================

else:

    st.subheader("Instrucciones")

    st.markdown(
        """
        1. Los datos de prueba (`X_test.csv` y `y_test.csv`) están en
           `Data_Test/xgboost/`, dentro de `src/app_demo/`. También puedes
           subir tus propios CSV con la opción **Subir CSV**.

        2. Selecciona el modelo **XGBoost**. Se carga el modelo congelado
           del repositorio, `src/models/xgboost/xgboost_final.json`.

        3. El umbral de decisión (0.87) y la mejor iteración del modelo
           se leen automáticamente de
           `src/models/xgboost/xgboost_final_metadatos.json`.

        4. Presiona **Ejecutar predicciones**.

        5. Visualiza las métricas, la matriz de
           confusión y las gráficas.

        **Importante:** `X_test.csv` debe contener las 55 características
        del modelo, con el mismo preprocesamiento usado en el entrenamiento.
        """
    )
