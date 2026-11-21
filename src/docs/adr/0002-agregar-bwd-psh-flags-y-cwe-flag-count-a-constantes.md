# ADR 0002: Agregar `Bwd PSH Flags` y `CWE Flag Count` a `COLUMNAS_CONSTANTES`

## Estado

Aceptado.

## Contexto

`COLUMNAS_CONSTANTES` (`src/preprocessing/clean.py`) ya listaba 8 columnas de varianza cero confirmada sobre el dataset filtrado (`Fwd URG Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`), descartadas en `basic_cleaning()` por no aportar señal a ningún modelo.

Al revisar `wednesday_low_and_slow_clean.csv` (salida actual del pipeline, 450,382 filas) se identificaron dos columnas adicionales con el mismo problema, que no estaban en ninguna lista de exclusión (`COLUMNAS_A_ELIMINAR_CANDIDATAS`, `COLUMNAS_REDUNDANTES` ni `COLUMNAS_CONSTANTES`) y por lo tanto seguían llegando intactas hasta X: `Bwd PSH Flags` y `CWE Flag Count`.

Se evaluó explícitamente que no se trata de una fuga ni de un caso de `COLUMNAS_REDUNDANTES` (esa lista es para columnas con correlación 1.0 frente a otra ya presente, no para varianza cero). El problema es genuinamente el mismo que motivó `COLUMNAS_CONSTANTES` en primer lugar, así que corresponde ampliar esa lista, no crear una nueva categoría.

## Decisión

Agregar `Bwd PSH Flags` y `CWE Flag Count` a `COLUMNAS_CONSTANTES` en `src/preprocessing/clean.py`. `basic_cleaning()` las descarta junto con las 8 anteriores; no se requiere ningún otro cambio de código, porque el resto del pipeline (`train_test_split_by_flow()`, `scale_features()`) ya opera sobre el conjunto de columnas resultante sin asumir un número fijo.

## Validación

Corrido contra el pipeline real (`load_raw_csvs → filter_low_and_slow → basic_cleaning → encode_labels → train_test_split_by_flow → scale_features`, 450,382 filas):

1. Varianza cero confirmada para las 10 columnas (las 8 ya existentes + las 2 nuevas), corriendo `basic_cleaning(columnas_constantes=[])` para retener también las 8 ya cubiertas y validarlas igual que las 2 nuevas:

```text
Bwd PSH Flags: nunique=1
Fwd URG Flags: nunique=1
Bwd URG Flags: nunique=1
CWE Flag Count: nunique=1
Fwd Avg Bytes/Bulk: nunique=1
Fwd Avg Packets/Bulk: nunique=1
Fwd Avg Bulk Rate: nunique=1
Bwd Avg Bytes/Bulk: nunique=1
Bwd Avg Packets/Bulk: nunique=1
Bwd Avg Bulk Rate: nunique=1
```

2. `scale_features()` no genera NaN/Inf al pasar por esas 10 columnas (relevante porque `StandardScaler` divide por la desviación estándar, que es 0 en una columna constante):

```text
NaN total: 0
Inf total: 0
```

scikit-learn maneja este caso internamente (`_handle_zeros_in_scale`): cuando la varianza es cero, usa un factor de escala de 1 en vez de 0 para evitar la división por cero. El resultado para una columna constante es `X - media = 0` en todas las filas: sin NaN/Inf, pero también sin ninguna señal, lo que confirma que descartarlas no pierde información real.

## Alternativas consideradas

* Dejarlas en X y confiar en que el modelo (o `StandardScaler`) las neutralice solo. Descartada: aunque no rompen nada numéricamente (validado arriba), son dimensiones muertas que solo añaden ruido y costo de cómputo sin aportar señal, el mismo criterio ya aplicado a las 8 columnas originales de `COLUMNAS_CONSTANTES`.
* Agregarlas a `COLUMNAS_REDUNDANTES`. Descartada: esa lista está reservada para columnas con correlación exacta 1.0 frente a otra columna ya presente; `Bwd PSH Flags` y `CWE Flag Count` no son redundantes frente a otra columna, son constantes en sí mismas. El caso correcto es `COLUMNAS_CONSTANTES`.

## Consecuencias

* X (y por tanto `X_train`/`X_val`/`X_test`) pasa de 68 a 66 columnas tras `basic_cleaning()`. Ningún modelo (GRU/SVM/XGBoost) tenía código dependiente de estas dos columnas todavía, así que no hay migración pendiente.
* `data/processed/wednesday_low_and_slow_clean.csv` queda desactualizado (sigue incluyendo ambas columnas) hasta que se regenere corriendo `notebooks/Cleaning/limpieza_low_and_slow.ipynb`. Se deja pendiente intencionalmente, fuera del alcance de este cambio (lo corre el equipo manualmente).
* No afecta `train_test_split_by_flow()` ni la Fase 5 (construcción de secuencias del GRU): esa función no asume una lista fija de columnas de features, solo excluye `Timestamp`/`bloque` de X por nombre explícito.
