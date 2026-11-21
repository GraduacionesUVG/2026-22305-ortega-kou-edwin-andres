"""
Preprocesamiento base compartido para el dataset CIC-IDS2017 (captura del miércoles).

Este módulo concentra la limpieza de datos, la codificación de etiquetas y la
partición cronológica en entrenamiento, validación y prueba que utilizan los tres
modelos del proyecto (GRU, SVM y XGBoost). Al centralizar estas etapas, los tres
modelos se entrenan y se evalúan sobre los mismos datos y las mismas particiones,
condición necesaria para que la comparación entre ellos sea válida.

Uso típico:
    from src.preprocessing.clean import load_and_clean, train_test_split_by_flow

    df = load_and_clean("data/raw/")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_flow(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columnas identificadoras. Se eliminan porque no describen el comportamiento del
# tráfico y permitirían al modelo memorizar hosts o conexiones concretas (fuga de
# información). "Source Port" es un puerto efímero que asigna el sistema operativo
# del cliente, por lo que tampoco caracteriza el flujo. Se eliminan después de
# quitar los duplicados exactos (ver basic_cleaning).
# "Timestamp" no está en esta lista porque se necesita para ordenar los flujos.
COLUMNAS_A_ELIMINAR_CANDIDATAS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
]

# Columnas que reflejan la configuración del entorno de laboratorio y no el
# comportamiento del ataque.
#   - "Destination Port": el 100% de los ataques se dirige al puerto 80, frente a
#     ~11% del tráfico benigno. Con esta columna un modelo puede obtener un recall
#     alto aprendiendo "puerto 80 = ataque", sin modelar el comportamiento lento
#     que es el objeto de la investigación. En una red real el ataque podría
#     dirigirse a otro puerto.
#   - "Protocol": Slowloris y Slowhttptest operan sobre HTTP y, por tanto, sobre TCP.
#     Como ~46% del tráfico benigno es UDP, con esta columna el modelo descarta ese
#     tráfico sin evaluar comportamiento Low & Slow, lo que puede inflar la
#     precision. Se excluye por criterio conservador.
# La decisión debe ser la misma para los tres modelos. Para el experimento de
# ablación "con atajos" se usa basic_cleaning(..., eliminar_atajos=False).
COLUMNAS_ATAJO_LABORATORIO = ["Destination Port", "Protocol"]

# Si es True, basic_cleaning() conserva y parsea "Timestamp". Esta columna es
# necesaria para ordenar los flujos cronológicamente (partición por bloques y
# construcción de secuencias del GRU). SVM y XGBoost la descartan antes de
# entrenar, ya que no es una feature.
PRESERVAR_TIMESTAMP = True
COLUMNA_TIMESTAMP = "Timestamp"

COLUMNA_ETIQUETA = "Label"

# El Timestamp de GeneratedLabelledFlows está expresado en reloj de 12 horas sin
# indicador AM/PM (por ejemplo, "5/7/2017 1:05" corresponde a las 13:05). La captura
# del miércoles abarca aproximadamente de 08:40 a 17:10, por lo que toda hora
# inferior a 8 pertenece a la tarde. Estas constantes definen el horario válido
# usado para corregir y para verificar el Timestamp.
HORA_INICIO_CAPTURA = 8
HORA_FIN_CAPTURA = 17

# Columnas que CICFlowMeter calcula dos veces. Se conserva una sola copia para no
# ponderar dos veces la misma información. Las siete primeras son idénticas o tienen
# correlación 1.0 con otra columna conservada (tanto del lado Fwd como del Bwd);
# "Fwd PSH Flags" coincide fila a fila con "SYN Flag Count".
COLUMNAS_REDUNDANTES = [
    "Subflow Fwd Packets",
    "Fwd Header Length.1",
    "Avg Fwd Segment Size",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Avg Bwd Segment Size",
    "Subflow Bwd Bytes",
    "Fwd PSH Flags",
]

# Columnas con un único valor en todo el dataset filtrado (BENIGN + Low & Slow).
# No tienen poder discriminativo, por lo que no aportan información para clasificar.
COLUMNAS_CONSTANTES = [
    "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Bwd PSH Flags", "CWE Flag Count",
]

# Ventanas horarias de cada ataque, documentadas en la descripción del dataset para
# el 5/7/2017. Permiten detectar filas etiquetadas como ataque cuyo inicio queda
# fuera de su ventana (ver excluir_ataques_fuera_de_ventana). La tolerancia
# compensa la granularidad de un minuto del Timestamp.
VENTANAS_ATAQUE_DOCUMENTADAS = {
    "DoS slowloris": ("2017-07-05 09:47", "2017-07-05 10:10"),
    "DoS Slowhttptest": ("2017-07-05 10:14", "2017-07-05 10:35"),
}
TOLERANCIA_VENTANA = pd.Timedelta(minutes=2)

# En la captura del miércoles los atacantes externos aparecen con la IP de la
# interfaz NAT del firewall, y el objetivo de los ataques DoS es el servidor web
# (ver excluir_respuestas_de_ataque).
IP_NAT_ATACANTE = "172.16.0.1"
IP_VICTIMA = "192.168.10.50"


def load_raw_csvs(raw_dir: str = "data/raw", filename: str | None = None) -> pd.DataFrame:
    """
    Carga el CSV crudo del miércoles (CIC-IDS2017).

    data/raw/ puede contener dos versiones del dataset: MachineLearningCSV (sin
    Timestamp ni IPs) y GeneratedLabelledFlows (con Timestamp e IPs). Se utiliza
    únicamente la segunda porque el ordenamiento cronológico de los flujos requiere
    la columna Timestamp. Ambas versiones no deben concatenarse: contienen el mismo
    tráfico con esquemas de columnas distintos.

    Parameters
    ----------
    filename : str, opcional
        Nombre exacto del archivo a cargar (se busca recursivamente dentro de
        raw_dir). Si no se especifica, se busca un CSV cuyas columnas incluyan
        "Timestamp". Si hay más de uno (GeneratedLabelledFlows trae todos los días
        de la semana), se usa el que contenga "wednesday" en el nombre; si no hay
        exactamente uno, se lanza un error pidiendo especificar el archivo.
    """
    raw_path = Path(raw_dir)

    if filename:
        matches = list(raw_path.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"No se encontró '{filename}' dentro de {raw_dir}")
        df = pd.read_csv(matches[0], low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"Cargado: {matches[0]}")
        return df

    csv_files = sorted(raw_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontraron CSVs en {raw_dir}. "
            "Descarga GeneratedLabelledFlows.zip del CIC-IDS-2017 y colócalo ahí."
        )

    # Solo se leen los encabezados para identificar los CSV que tienen Timestamp
    candidatos_con_timestamp = []
    for f in csv_files:
        cols = pd.read_csv(f, nrows=0).columns.str.strip()
        if "Timestamp" in cols:
            candidatos_con_timestamp.append(f)

    if not candidatos_con_timestamp:
        raise FileNotFoundError(
            f"Ningún CSV en {raw_dir} tiene columna 'Timestamp'. "
            "Asegúrate de usar GeneratedLabelledFlows.zip (no MachineLearningCSV.zip), "
            "que sí incluye Timestamp/Flow ID/IPs."
        )
    elegido = candidatos_con_timestamp[0]
    if len(candidatos_con_timestamp) > 1:
        candidatos_miercoles = [
            f for f in candidatos_con_timestamp if "wednesday" in f.name.lower()
        ]
        if len(candidatos_miercoles) != 1:
            raise ValueError(
                f"Se encontraron varios CSVs con Timestamp: {candidatos_con_timestamp}. "
                "Especifica cuál usar con el parámetro filename=."
            )
        elegido = candidatos_miercoles[0]
        print(
            f"AVISO: se encontraron {len(candidatos_con_timestamp)} CSVs con Timestamp "
            "(parece el dataset completo de la semana). Se usa automáticamente el que "
            f"corresponde al miércoles: {elegido}"
        )

    df = pd.read_csv(elegido, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Cargado: {elegido}")
    return df


def parsear_timestamp(serie: pd.Series) -> pd.Series:
    """
    Convierte el Timestamp de CIC-IDS2017 a datetime y corrige el reloj de 12 horas.

    - El formato del dataset es D/M/YYYY, por lo que se usa dayfirst=True
      ("5/7/2017 8:42" es el 5 de julio, no el 7 de mayo).
    - Si las cadenas no contienen AM/PM y ninguna hora es mayor o igual a 13, el
      archivo está en reloj de 12 horas sin indicador. En ese caso, las horas
      menores a HORA_INICIO_CAPTURA se desplazan 12 horas, pues corresponden a la
      tarde. Sin esta corrección, la tarde se ordenaría antes que la mañana y la
      partición cronológica dejaría de serlo.
    - Si el archivo ya trae AM/PM o alguna hora mayor o igual a 13, no se modifica,
      de modo que la función es segura para archivos de otros días.

    Lanza un error si, tras la corrección, alguna hora queda fuera del horario de
    captura.
    """
    crudo = serie.astype(str).str.strip()
    tiene_ampm = crudo.str.contains(r"\b[AP]M\b", case=False, regex=True).any()

    ts = pd.to_datetime(crudo, errors="coerce", dayfirst=True)
    horas = ts.dt.hour

    if not tiene_ampm and not (horas >= 13).any():
        a_corregir = horas < HORA_INICIO_CAPTURA
        print(
            f"Reloj de 12 h sin AM/PM detectado: {int(a_corregir.sum())} filas con hora "
            f"< {HORA_INICIO_CAPTURA} se desplazan +12 h (tarde)."
        )
        ts = ts.where(~a_corregir, ts + pd.Timedelta(hours=12))

    # Verificación posterior: todas las horas deben caer dentro del horario de captura
    horas_validas = ts.dropna().dt.hour
    fuera = ~horas_validas.between(HORA_INICIO_CAPTURA, HORA_FIN_CAPTURA)
    if fuera.any():
        raise ValueError(
            f"{int(fuera.sum())} timestamps fuera del horario de captura "
            f"({HORA_INICIO_CAPTURA}:00-{HORA_FIN_CAPTURA}:59) tras la corrección. "
            f"Horas encontradas: {sorted(horas_validas.unique().tolist())}"
        )
    return ts


def basic_cleaning(
    df: pd.DataFrame,
    columnas_a_eliminar: Iterable[str] = COLUMNAS_A_ELIMINAR_CANDIDATAS,
    columnas_redundantes: Iterable[str] = COLUMNAS_REDUNDANTES,
    columnas_constantes: Iterable[str] = COLUMNAS_CONSTANTES,
    eliminar_atajos: bool = True,
) -> pd.DataFrame:
    """
    Limpieza común a los tres modelos. Aplica, en orden:

    1. Parseo de Timestamp y corrección del reloj de 12 horas (parsear_timestamp).
    2. Reemplazo de infinitos por NaN y eliminación de filas con valores nulos y de
       duplicados exactos.
    3. Eliminación de columnas identificadoras, de atajos de laboratorio (salvo con
       eliminar_atajos=False), de columnas redundantes y de columnas constantes.
    4. Eliminación de filas con valores negativos en columnas que no admiten signo
       negativo (duración, tiempos entre paquetes y tasas).
    """
    df = df.copy()

    # El Timestamp se parsea antes del dropna para detectar de forma explícita las
    # fechas mal formadas, en lugar de que se pierdan en silencio al eliminar nulos.
    if PRESERVAR_TIMESTAMP and COLUMNA_TIMESTAMP in df.columns:
        antes = len(df)
        df[COLUMNA_TIMESTAMP] = parsear_timestamp(df[COLUMNA_TIMESTAMP])
        n_timestamps_invalidos = df[COLUMNA_TIMESTAMP].isna().sum()
        if n_timestamps_invalidos > 0:
            print(
                f"AVISO: {n_timestamps_invalidos} de {antes} filas tienen "
                f"'{COLUMNA_TIMESTAMP}' no parseable como fecha."
            )
        rango = (
            f"{df[COLUMNA_TIMESTAMP].min()} -> {df[COLUMNA_TIMESTAMP].max()}"
            if n_timestamps_invalidos < antes
            else "N/A"
        )
        print(f"Rango de Timestamp (corregido): {rango}")

    # Las columnas de tasa (bytes/s, paquetes/s) producen infinitos cuando la duración
    # del flujo es cero; se tratan como valores faltantes.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Se reportan las cantidades antes de eliminar para dejar registro en la bitácora
    n_nulos = df.isna().sum().sum()
    n_dup = df.duplicated().sum()
    print(f"Valores nulos/infinitos detectados: {n_nulos}")
    print(f"Filas duplicadas exactas detectadas: {n_dup}")

    df = df.dropna()
    # La deduplicación se realiza antes de eliminar IPs y puertos para remover
    # únicamente copias exactas (artefactos de captura). Dos flujos que difieren solo
    # en esas columnas son eventos distintos y deben conservarse, pues la GRU los
    # necesita dentro de la secuencia. Los vectores de features repetidos, que son los
    # que perciben SVM y XGBoost, se cuantifican aparte en
    # reportar_duplicados_de_features().
    df = df.drop_duplicates()

    existentes = [c for c in columnas_a_eliminar if c in df.columns]
    if existentes:
        df = df.drop(columns=existentes)
        print(f"Columnas identificadoras eliminadas: {existentes}")

    if eliminar_atajos:
        atajos = [c for c in COLUMNAS_ATAJO_LABORATORIO if c in df.columns]
        if atajos:
            df = df.drop(columns=atajos)
            print(f"Columnas atajo de laboratorio eliminadas: {atajos}")
    else:
        print(f"AVISO: se conservan {COLUMNAS_ATAJO_LABORATORIO} (corrida de ablación)")

    existentes_redundantes = [c for c in columnas_redundantes if c in df.columns]
    if existentes_redundantes:
        df = df.drop(columns=existentes_redundantes)
        print(f"Columnas redundantes eliminadas: {existentes_redundantes}")

    # Duración, tiempos entre paquetes y tasas no pueden ser negativos; los valores
    # negativos son datos corruptos del generador de flujos. En este dataset todas
    # esas filas pertenecen a la clase BENIGN, por lo que ningún ataque se pierde con
    # este filtro.
    columnas_deben_ser_no_negativas = [
        "Flow Duration", "Flow IAT Min", "Flow Bytes/s", "Flow Packets/s",
    ]
    presentes = [c for c in columnas_deben_ser_no_negativas if c in df.columns]
    if presentes:
        mask_negativo = (df[presentes] < 0).any(axis=1)
        n_negativas = mask_negativo.sum()
        if n_negativas > 0:
            print(f"Filas con valores negativos anómalos eliminadas: {n_negativas} "
                  f"({n_negativas/len(df)*100:.4f}%)")
            df = df[~mask_negativo]

    existentes_constantes = [c for c in columnas_constantes if c in df.columns]
    if existentes_constantes:
        df = df.drop(columns=existentes_constantes)
        print(f"Columnas constantes eliminadas: {existentes_constantes}")

    return df


# Además de BENIGN, el CSV del miércoles contiene ataques que no son Low & Slow
# (DoS Hulk, DoS GoldenEye y Heartbleed). El alcance del protocolo se limita a
# Slowloris y Slowhttptest, por lo que esas clases se excluyen en lugar de agruparse
# como "ataque". Los nombres de las etiquetas pueden variar entre versiones del
# dataset (por ejemplo, "DoS slowloris" y "DoS Slowloris").
ETIQUETAS_LOW_AND_SLOW = [
    "DoS slowloris",
    "DoS Slowhttptest",
]
ETIQUETA_BENIGNA = "BENIGN"


def filter_low_and_slow(df: pd.DataFrame, columna: str = COLUMNA_ETIQUETA) -> pd.DataFrame:
    """
    Conserva únicamente el tráfico BENIGN y los ataques Low & Slow (Slowloris y
    Slowhttptest), descartando los demás ataques de la captura.
    """
    df = df.copy()
    df[columna] = df[columna].astype(str).str.strip()

    etiquetas_presentes = df[columna].unique().tolist()
    print(f"Etiquetas encontradas en el CSV: {etiquetas_presentes}")

    etiquetas_relevantes = [ETIQUETA_BENIGNA] + ETIQUETAS_LOW_AND_SLOW
    faltantes = [e for e in ETIQUETAS_LOW_AND_SLOW if e not in etiquetas_presentes]
    if faltantes:
        print(
            f"AVISO: no se encontraron estas etiquetas esperadas: {faltantes}. "
            "Revisa el nombre exacto con df['Label'].value_counts() y ajusta "
            "ETIQUETAS_LOW_AND_SLOW en este archivo."
        )

    antes = len(df)
    df = df[df[columna].isin(etiquetas_relevantes)]
    print(f"Filas antes del filtro: {antes} | después (solo BENIGN + Low&Slow): {len(df)}")

    return df


def excluir_respuestas_de_ataque(df: pd.DataFrame, columna: str = COLUMNA_ETIQUETA) -> pd.DataFrame:
    """
    Excluye las filas BENIGN entre IP_NAT_ATACANTE e IP_VICTIMA, en cualquier sentido.

    En esta captura todo flujo NAT → víctima está etiquetado como ataque, y los
    BENIGN del par van todos en sentido víctima → NAT: son el lado servidor de
    conexiones de ataque (casi todos comparten el puerto de cliente con un flujo de
    ataque iniciado en el mismo minuto y solo ocurren durante los ataques). Es ruido
    de etiqueta; se excluyen y no se reetiquetan, igual que en
    excluir_ataques_fuera_de_ventana(). Debe llamarse antes de basic_cleaning(),
    que elimina las columnas de IP.
    """
    faltantes = [c for c in ("Source IP", "Destination IP") if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan las columnas {faltantes}: excluir_respuestas_de_ataque() debe "
            "llamarse antes de basic_cleaning(), que elimina las IPs."
        )

    origen = df["Source IP"].astype(str).str.strip()
    destino = df["Destination IP"].astype(str).str.strip()
    es_benigno = df[columna].astype(str).str.strip() == ETIQUETA_BENIGNA

    victima_a_nat = es_benigno & (origen == IP_VICTIMA) & (destino == IP_NAT_ATACANTE)
    nat_a_victima = es_benigno & (origen == IP_NAT_ATACANTE) & (destino == IP_VICTIMA)
    excluir = victima_a_nat | nat_a_victima

    n_benigno = int(es_benigno.sum())
    pct = excluir.sum() / n_benigno * 100 if n_benigno else 0.0
    print(
        f"Respuestas de ataque etiquetadas BENIGN excluidas: {int(excluir.sum())} "
        f"(víctima → NAT: {int(victima_a_nat.sum())}, NAT → víctima: {int(nat_a_victima.sum())}; "
        f"{pct:.2f}% de {n_benigno} BENIGN)"
    )
    return df[~excluir]


def encode_labels(df: pd.DataFrame, columna: str = COLUMNA_ETIQUETA) -> pd.DataFrame:
    """
    Codifica la etiqueta como variable binaria en 'label_binaria': 0 = BENIGN y
    1 = ataque Low & Slow. Debe llamarse después de filter_low_and_slow() para que la
    clase 1 represente únicamente Slowloris y Slowhttptest. La columna original se
    conserva para poder reportar el desempeño por tipo de ataque.
    """
    df = df.copy()
    df[columna] = df[columna].astype(str).str.strip()
    df["label_binaria"] = (df[columna] != ETIQUETA_BENIGNA).astype(int)
    return df


def excluir_ataques_fuera_de_ventana(
    df: pd.DataFrame,
    ventanas: dict[str, tuple[str, str]] = VENTANAS_ATAQUE_DOCUMENTADAS,
    tolerancia: pd.Timedelta = TOLERANCIA_VENTANA,
    columna: str = COLUMNA_ETIQUETA,
) -> pd.DataFrame:
    """
    Excluye las filas etiquetadas como ataque cuyo Timestamp cae fuera de la ventana
    documentada de ese ataque (± tolerancia). Requiere el Timestamp ya corregido.

    Justificación: CICFlowMeter registra el instante de INICIO del flujo. Una fila de
    ataque que empieza fuera de la ventana solo sería legítima si su duración la
    llevara dentro de ella (por ejemplo, una conexión larga iniciada antes del
    ataque). En la captura del miércoles, los pocos flujos de Slowloris en esta
    situación duran 5 segundos o menos y se ubican horas antes o después del ataque,
    por lo que no pueden pertenecer a él y se consideran ruido de etiquetado.

    Las filas se excluyen y no se reetiquetan como BENIGN, porque su clase real es
    desconocida. Si se excluyera más del 1% de los ataques, se lanza un error: esa
    magnitud indicaría un problema en el Timestamp y no ruido de etiquetado.
    """
    df = df.copy()
    fuera_total = pd.Series(False, index=df.index)

    for etiqueta, (ini, fin) in ventanas.items():
        ini_tol = pd.Timestamp(ini) - tolerancia
        fin_tol = pd.Timestamp(fin) + tolerancia
        es_clase = df[columna] == etiqueta
        fuera = es_clase & ~df[COLUMNA_TIMESTAMP].between(ini_tol, fin_tol)
        print(f"{etiqueta}: {int(fuera.sum())} de {int(es_clase.sum())} filas fuera de "
              f"[{ini_tol} .. {fin_tol}]")
        if fuera.any():
            cols = [c for c in (COLUMNA_TIMESTAMP, "Flow Duration") if c in df.columns]
            print(df.loc[fuera, cols].to_string())
        fuera_total |= fuera

    n_ataques = int(df[columna].isin(list(ventanas)).sum())
    if n_ataques and fuera_total.sum() > 0.01 * n_ataques:
        raise ValueError(
            f"Se excluirían {int(fuera_total.sum())} de {n_ataques} ataques (>1%). "
            "Probablemente el Timestamp no está corregido: revisa parsear_timestamp()."
        )

    print(f"Filas excluidas por estar fuera de ventana: {int(fuera_total.sum())}")
    return df[~fuera_total]


def load_and_clean(raw_dir: str = "data/raw", eliminar_atajos: bool = True) -> pd.DataFrame:
    """
    Pipeline completo: carga, filtro Low & Slow, exclusión de respuestas de ataque,
    limpieza básica, codificación de etiquetas y exclusión de ataques fuera de
    ventana.

    El filtro se aplica primero para no procesar filas que de todos modos se
    descartan (Hulk, GoldenEye y Heartbleed). La exclusión de respuestas de ataque va
    antes de la limpieza básica porque necesita las IPs, que esa limpieza elimina. La
    exclusión por ventana va al final porque requiere el Timestamp corregido y la
    etiqueta original.
    """
    df = load_raw_csvs(raw_dir)
    df = filter_low_and_slow(df)
    df = excluir_respuestas_de_ataque(df)
    df = basic_cleaning(df, eliminar_atajos=eliminar_atajos)
    df = encode_labels(df)
    df = excluir_ataques_fuera_de_ventana(df)
    return df


def train_test_split_by_flow(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    outlier_quantile: float = 0.0,
    desempate: str = "csv",
    minutos_hueco: int = 0,
    return_meta: bool = False,
) -> tuple:
    """
    Partición cronológica (no aleatoria) en train, val y test, idéntica para los tres
    modelos. La GRU requiere flujos temporalmente contiguos para construir
    secuencias, y una partición aleatoria mezclaría épocas distintas del tráfico.
    Justificación completa: docs/adr/0001-split-cronologico-por-bloques-temporales.md

    El DataFrame se ordena por Timestamp y se divide en cinco bloques. Las fronteras
    son los cuantiles [outlier_quantile, 1 - outlier_quantile] del Timestamp de cada
    ataque:

        Bloque 1: antes de la ventana de Slowloris        (BENIGN)
        Bloque 2: ventana de Slowloris                     (Slowloris + BENIGN)
        Bloque 3: intervalo entre ambos ataques            (BENIGN)
        Bloque 4: ventana de Slowhttptest                  (Slowhttptest + BENIGN)
        Bloque 5: después de la ventana de Slowhttptest    (BENIGN)

    Dentro de cada bloque, ya en orden cronológico, se corta por posición:
    (1 - test_size - val_size) para train, val_size para val y el resto para test.
    Así cada partición contiene tráfico de todas las fases de la captura. La
    proporción de ataques difiere entre particiones por diseño.

    Parameters
    ----------
    random_state : int
        Semilla del desempate aleatorio (solo se usa con desempate="aleatorio").
    outlier_quantile : float
        Cuantil usado para definir el inicio y el fin de la ventana de cada ataque.
        Con 0.0 se emplean el mínimo y el máximo. Valores mayores recortan los
        extremos y asignarían ataques reales a bloques benignos; no son necesarios
        porque excluir_ataques_fuera_de_ventana() ya elimina las filas aisladas.
    desempate : {"csv", "aleatorio"}
        El Timestamp tiene granularidad de un minuto y cada minuto contiene cientos
        de flujos. Con "csv" se conserva el orden del archivo dentro de cada minuto;
        ese orden no corresponde necesariamente al de inicio de los flujos, y los
        flujos de ataque aparecen agrupados. Con "aleatorio" el orden dentro del
        minuto se sortea con random_state; sirve como ablación para medir cuánto
        depende la GRU del orden del archivo.
    minutos_hueco : int
        Si es mayor que 0, se descartan, del lado anterior de cada frontera (train→val
        y val→test, dentro de cada bloque), las filas cuyo Timestamp está a menos de
        ese número de minutos del primer Timestamp de la partición siguiente.
        CICFlowMeter divide las conexiones largas en varios flujos consecutivos casi
        idénticos, de modo que fragmentos de una misma conexión pueden quedar a ambos
        lados de una frontera e inflar las métricas. El costo es perder filas de
        ataque en los bloques 2 y 4.
    return_meta : bool
        Si es True, devuelve además meta_train, meta_val y meta_test, con la columna
        Label original, para reportar el desempeño por tipo de ataque. Label nunca
        forma parte de X.

    Contrato de columnas: 'Timestamp' y 'bloque' permanecen en X como metadata (la
    GRU las necesita para no cruzar bloques al construir secuencias; SVM y XGBoost
    deben descartarlas). Las columnas 'Label', 'label_binaria' y 'split' nunca
    llegan a X.
    """
    target_col = "label_binaria"

    if COLUMNA_TIMESTAMP not in df.columns:
        raise ValueError(
            f"'{COLUMNA_TIMESTAMP}' no está en el DataFrame — se necesita para el split "
            "cronológico. Revisa que PRESERVAR_TIMESTAMP=True al llamar basic_cleaning()."
        )

    # El ordenamiento estable preserva el orden del archivo en los empates de
    # Timestamp; el algoritmo por defecto de pandas (quicksort) no lo garantiza.
    if desempate == "csv":
        df = df.sort_values(COLUMNA_TIMESTAMP, kind="stable")
    elif desempate == "aleatorio":
        rng = np.random.default_rng(random_state)
        df = (
            df.assign(_desempate=rng.random(len(df)))
            .sort_values([COLUMNA_TIMESTAMP, "_desempate"])
            .drop(columns="_desempate")
        )
    else:
        raise ValueError(f"desempate debe ser 'csv' o 'aleatorio' (recibido {desempate!r})")
    df = df.reset_index(drop=True)

    ts_slowloris = df.loc[df[COLUMNA_ETIQUETA] == "DoS slowloris", COLUMNA_TIMESTAMP]
    ts_slowhttp = df.loc[df[COLUMNA_ETIQUETA] == "DoS Slowhttptest", COLUMNA_TIMESTAMP]
    if ts_slowloris.empty or ts_slowhttp.empty:
        raise ValueError(
            "No hay filas de 'DoS slowloris' o 'DoS Slowhttptest' en df — ¿se llamó "
            "filter_low_and_slow() antes? No se pueden calcular las fronteras de bloque."
        )

    t_slowloris_ini = ts_slowloris.quantile(outlier_quantile)
    t_slowloris_fin = ts_slowloris.quantile(1 - outlier_quantile)
    t_slowhttp_ini = ts_slowhttp.quantile(outlier_quantile)
    t_slowhttp_fin = ts_slowhttp.quantile(1 - outlier_quantile)

    print(f"Fronteras de bloque (outlier_quantile={outlier_quantile}, desempate={desempate}):")
    print(f"  Slowloris:    [{t_slowloris_ini} .. {t_slowloris_fin}]  ({len(ts_slowloris)} filas)")
    print(f"  Slowhttptest: [{t_slowhttp_ini} .. {t_slowhttp_fin}]  ({len(ts_slowhttp)} filas)")

    # Asignación de cada flujo a su bloque según las fronteras calculadas
    ts = df[COLUMNA_TIMESTAMP]
    condiciones = [
        ts < t_slowloris_ini,
        (ts >= t_slowloris_ini) & (ts <= t_slowloris_fin),
        (ts > t_slowloris_fin) & (ts < t_slowhttp_ini),
        (ts >= t_slowhttp_ini) & (ts <= t_slowhttp_fin),
    ]
    df["bloque"] = np.select(condiciones, [1, 2, 3, 4], default=5)

    print("\nComposición por bloque:")
    for b in range(1, 6):
        sub = df[df["bloque"] == b]
        if len(sub) == 0:
            print(f"  Bloque {b}: 0 filas (VACÍO) — revisar outlier_quantile")
            continue
        n_ataque = int((sub[target_col] == 1).sum())
        print(
            f"  Bloque {b}: {len(sub)} filas | Timestamp [{sub[COLUMNA_TIMESTAMP].min()} .. "
            f"{sub[COLUMNA_TIMESTAMP].max()}] | ataques: {n_ataque} ({n_ataque/len(sub)*100:.2f}%)"
        )
        # Los bloques 1, 3 y 5 quedan fuera de las ventanas de ataque, por lo que no
        # deberían contener ataques
        if b in (1, 3, 5) and n_ataque > 0:
            print(f"    AVISO: el bloque {b} debería ser benigno y tiene {n_ataque} ataques")

    train_frac = 1 - test_size - val_size
    if train_frac <= 0:
        raise ValueError(f"test_size + val_size debe ser < 1 (recibido {test_size + val_size})")

    # Corte por posición dentro de cada bloque. Como el DataFrame ya está ordenado
    # cronológicamente, basta etiquetar cada fila con su partición.
    splits = pd.Series(index=df.index, dtype=object)
    for b in range(1, 6):
        idx_bloque = df.index[df["bloque"] == b]
        n = len(idx_bloque)
        if n == 0:
            continue
        n_train = round(n * train_frac)
        n_val = round(n * val_size)
        splits.loc[idx_bloque[:n_train]] = "train"
        splits.loc[idx_bloque[n_train:n_train + n_val]] = "val"
        splits.loc[idx_bloque[n_train + n_val:]] = "test"
    df["split"] = splits

    # Hueco opcional en las fronteras entre particiones (ver minutos_hueco)
    if minutos_hueco > 0:
        hueco = pd.Timedelta(minutes=minutos_hueco)
        descartar = pd.Series(False, index=df.index)
        for b in range(1, 6):
            en_bloque = df["bloque"] == b
            for anterior, siguiente in (("train", "val"), ("val", "test")):
                ts_sig = df.loc[en_bloque & (df["split"] == siguiente), COLUMNA_TIMESTAMP]
                if ts_sig.empty:
                    continue
                descartar |= (
                    en_bloque
                    & (df["split"] == anterior)
                    & (df[COLUMNA_TIMESTAMP] > ts_sig.min() - hueco)
                )
        resumen = df[descartar].groupby("split")[target_col].agg(["size", "sum"])
        resumen.columns = ["filas", "ataques"]
        print(f"\nHueco de {minutos_hueco} min en fronteras: se descartan {int(descartar.sum())} filas")
        print(resumen.to_string())
        df = df[~descartar]

    # Label se separa en un DataFrame aparte para que no pueda usarse como feature por
    # accidente
    meta = df[[COLUMNA_ETIQUETA]]
    columnas_a_excluir = [target_col, COLUMNA_ETIQUETA, "split"]
    X = df.drop(columns=columnas_a_excluir)
    y = df[target_col]

    mask_train = df["split"] == "train"
    mask_val = df["split"] == "val"
    mask_test = df["split"] == "test"

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    print(f"\nTrain: {len(X_train)} filas | Val: {len(X_val)} filas | Test: {len(X_test)} filas")
    print(
        f"Proporción de ataques -> train: {y_train.mean():.4f}, val: {y_val.mean():.4f}, "
        f"test: {y_test.mean():.4f}"
    )
    for nombre, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if (y_split == 1).sum() == 0:
            print(
                f"ADVERTENCIA: el split '{nombre}' tiene 0 ejemplos de ataque. "
                "outlier_quantile probablemente necesita ajustarse — no continuar así."
            )

    # Verificación del contrato de columnas
    assert "split" not in X_train.columns
    assert COLUMNA_ETIQUETA not in X_train.columns
    assert "bloque" in X_train.columns and COLUMNA_TIMESTAMP in X_train.columns

    if return_meta:
        return (
            X_train, X_val, X_test, y_train, y_val, y_test,
            meta[mask_train], meta[mask_val], meta[mask_test],
        )
    return X_train, X_val, X_test, y_train, y_val, y_test


def reportar_duplicados_de_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    columnas_metadata: Iterable[str] = (COLUMNA_TIMESTAMP, "bloque"),
) -> pd.DataFrame:
    """
    Diagnóstico de vectores de features repetidos, que es lo que perciben SVM y
    XGBoost. No elimina ninguna fila: la GRU puede encontrar flujos repetidos de
    forma legítima dentro de una secuencia, y solo se requiere conocer la magnitud
    del fenómeno.

    Reporta, por partición y clase:
      - dup_en_split: filas cuyo vector ya apareció antes en la misma partición.
      - en_train: filas de val o test cuyo vector exacto está en train. Estas filas
        evalúan al modelo con algo que ya vio durante el entrenamiento, lo que puede
        inflar las métricas.
    Además, reporta cuántas filas comparten vector con otra fila de etiqueta distinta.
    """
    columnas_metadata = set(columnas_metadata)
    feats = [c for c in X_train.columns if c not in columnas_metadata]

    # Cada vector de features se resume en un hash para compararlos eficientemente
    def _hash(X: pd.DataFrame) -> pd.Series:
        return pd.util.hash_pandas_object(X[feats], index=False)

    hashes = {"train": _hash(X_train), "val": _hash(X_val), "test": _hash(X_test)}
    ys = {"train": y_train, "val": y_val, "test": y_test}
    en_train = set(hashes["train"].to_numpy())

    filas = []
    for nombre in ("train", "val", "test"):
        h, y = hashes[nombre], ys[nombre].to_numpy()
        dup = h.duplicated().to_numpy()
        presente = h.isin(en_train).to_numpy() if nombre != "train" else np.zeros(len(h), bool)
        for clase in (0, 1):
            m = y == clase
            filas.append({
                "split": nombre, "clase": clase, "filas": int(m.sum()),
                "dup_en_split": int((dup & m).sum()),
                "en_train": int((presente & m).sum()) if nombre != "train" else None,
            })
    reporte = pd.DataFrame(filas)
    print(reporte.to_string(index=False))

    todo = pd.DataFrame({
        "h": pd.concat(hashes.values(), ignore_index=True),
        "y": pd.concat(ys.values(), ignore_index=True),
    })
    conflictivos = todo.groupby("h")["y"].transform("nunique") > 1
    print(f"\nFilas cuyo vector aparece con ambas etiquetas: {int(conflictivos.sum())}")
    return reporte


def _log_con_signo(X: pd.DataFrame) -> pd.DataFrame:
    # sign(x)·log1p(|x|): transformación monótona definida también para valores
    # negativos (por ejemplo, el -1 de Init_Win_bytes_*), a diferencia de log1p, que
    # produce -inf en x = -1.
    return np.sign(X) * np.log1p(np.abs(X))


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    columns_to_scale: list[str] | None = None,
    transformacion_log: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Estandariza las features (media 0 y desviación 1). El escalador se ajusta
    únicamente con X_train y esa misma transformación se aplica a X_val y X_test, de
    modo que ninguna información de validación o prueba influye en el entrenamiento.
    La utilizan GRU y SVM, que son sensibles a la escala; XGBoost no la necesita
    porque sus divisiones son invariantes a transformaciones monótonas.

    transformacion_log=True aplica sign(x)·log1p(|x|) antes de estandarizar. Las
    duraciones, los tiempos entre paquetes y las tasas abarcan varios órdenes de
    magnitud (de 0 a 1e8–1e9); con estandarización sola, casi todos los valores
    quedarían comprimidos cerca de 0. Si se usa, GRU y SVM deben emplear el mismo
    valor y debe quedar documentado.

    Por defecto se escalan todas las columnas numéricas excepto 'Timestamp' y
    'bloque', que son metadata. Devuelve también el escalador ajustado, que debe
    guardarse (por ejemplo, con joblib.dump) para poder aplicar el modelo a datos
    nuevos.
    """
    if columns_to_scale is None:
        columnas_metadata = {COLUMNA_TIMESTAMP, "bloque"}
        columns_to_scale = [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if c not in columnas_metadata
        ]
    print(f"Columnas a escalar ({len(columns_to_scale)}): {columns_to_scale}")

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    if transformacion_log:
        for X in (X_train_scaled, X_val_scaled, X_test_scaled):
            X[columns_to_scale] = _log_con_signo(X[columns_to_scale].astype(float))

    scaler = StandardScaler()
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_scaled[columns_to_scale])
    X_val_scaled[columns_to_scale] = scaler.transform(X_val_scaled[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def diagnosticar_calidad_datos(df: pd.DataFrame, columna_clase: str = "label_binaria") -> None:
    """
    Diagnóstico de calidad de datos que no modifica el DataFrame. Reporta (1) la
    cantidad de valores negativos en columnas que no los admiten, con su distribución
    por clase, y (2) la proporción que ocupa el valor dominante en columnas candidatas
    a ser constantes. Es una herramienta de apoyo para el análisis exploratorio;
    basic_cleaning() aplica los filtros correspondientes.
    """
    columnas_deben_ser_no_negativas = [
        "Flow Duration", "Flow IAT Min", "Flow Bytes/s", "Flow Packets/s",
    ]
    print("=== Valores negativos ===")
    for col in columnas_deben_ser_no_negativas:
        if col not in df.columns:
            continue
        negativos = df[col] < 0
        n_neg = negativos.sum()
        if n_neg == 0:
            print(f"{col}: sin valores negativos")
            continue
        print(f"{col}: {n_neg} filas negativas ({n_neg/len(df)*100:.3f}%)")
        print("  Distribución por clase entre las filas negativas:")
        print(df.loc[negativos, columna_clase].value_counts())

    columnas_candidatas_constantes = [
        "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    ]
    print("\n=== Columnas casi constantes ===")
    for col in columnas_candidatas_constantes:
        if col not in df.columns:
            continue
        valor_mas_comun_pct = df[col].value_counts(normalize=True).iloc[0] * 100
        print(f"{col}: valor más común representa {valor_mas_comun_pct:.2f}% de las filas "
              f"({df[col].nunique()} valores únicos)")

def cargar_dataset_limpio(path: str) -> pd.DataFrame:
    """
    Carga el CSV procesado (salida de load_and_clean / notebook de limpieza).

    El CSV no guarda tipos de dato, por lo que Timestamp vuelve como texto. Se
    convierte con format="ISO8601" (el formato con que to_csv lo escribe) y no con
    dayfirst=True, que en pandas 3 interpreta las fechas ISO al revés (julio ->
    mayo). Las horas ya vienen corregidas (reloj de 24 h), por lo que no se vuelve
    a aplicar la corrección de parsear_timestamp().
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df[COLUMNA_TIMESTAMP] = pd.to_datetime(df[COLUMNA_TIMESTAMP], format="ISO8601")

    horas = df[COLUMNA_TIMESTAMP].dt.hour
    assert horas.between(HORA_INICIO_CAPTURA, HORA_FIN_CAPTURA).all(), (
        "Timestamp fuera del horario de captura: ¿se cargó un CSV generado antes "
        "de la corrección AM/PM?"
    )
    return df