# ADR 0001: Split cronológico por bloques temporales para train/val/test

## Estado

Se define aquí la convención `docs/adr/NNNN-titulo-en-kebab-case.md` para decisiones de arquitectura que afectan a los tres modelos. Este es el primer registro.

## Contexto

`train_test_split_by_flow()` (`src/preprocessing/clean.py`) hacía hasta ahora un split aleatorio estratificado (`train_test_split` de scikit-learn, `random_state=42` fijo). Esa función es compartida por los tres modelos del proyecto (GRU, SVM y XGBoost) precisamente para que la comparación de la Fase 7-8 del protocolo sea válida: los tres deben entrenarse y evaluarse sobre exactamente la misma partición de datos.

El problema es que el modelo GRU necesita, en la Fase 5, construir secuencias tomando "los N flujos anteriores en el tiempo" como contexto de cada predicción. Un split aleatorio dispersa filas temporalmente contiguas entre train/val/test de forma independiente, lo que:

* Hace imposible reconstruir secuencias coherentes dentro de una misma partición (los vecinos temporales de un flujo pueden haber quedado en otra partición).
* Filtra información temporal del futuro hacia el pasado: una fila de test cronológicamente anterior a una fila de train ya "vio" implícitamente parte de lo que pasó después, algo que no puede ocurrir en un escenario real de detección en producción.

SVM y XGBoost no tienen esta necesidad por su propia naturaleza (no dependen de orden secuencial), pero adoptan el mismo split de todas formas, no porque lo requieran, sino porque los tres modelos deben compararse sobre la misma partición para que la Fase 7-8 sea válida. Diseñar un split distinto por modelo habría invalidado esa comparación.

## Decisión

Reemplazar el split aleatorio estratificado por un split cronológico dividido en 5 bloques temporales, construido alrededor de las ventanas de ataque (Slowloris, Slowhttptest), e introducir `scale_features()` como utilidad de escalado compartida. Ambos viven en `src/preprocessing/clean.py`.

## Los 5 bloques

El DataFrame completo (BENIGN + Low&Slow) se ordena cronológicamente por Timestamp (`sort_values(..., kind="stable")`; la estabilidad es necesaria porque Timestamp tiene granularidad de minuto y el default de pandas, quicksort, no garantiza preservar el orden real de captura del CSV en los empates). Luego se divide en 5 bloques:

| Bloque | Condición                                         | Contenido esperado              |
| ------ | ------------------------------------------------- | ------------------------------- |
| 1      | `Timestamp < t_slowloris_ini`                     | benigno previo a los ataques    |
| 2      | `t_slowloris_ini <= Timestamp <= t_slowloris_fin` | Slowloris + benigno de fondo    |
| 3      | `t_slowloris_fin < Timestamp < t_slowhttp_ini`    | benigno entre ataques           |
| 4      | `t_slowhttp_ini <= Timestamp <= t_slowhttp_fin`   | Slowhttptest + benigno de fondo |
| 5      | `Timestamp > t_slowhttp_fin`                      | benigno posterior a los ataques |

Las 4 fronteras (`t_slowloris_ini/fin`, `t_slowhttp_ini/fin`) se calculan como los cuantiles `[outlier_quantile, 1 - outlier_quantile]` de Timestamp, tomados por separado sobre las filas de cada clase de ataque (`outlier_quantile` default 0.01).

Por qué cuantiles y no mínimo/máximo exacto: el EDA (`notebooks/Cleaning/ limpieza_low_and_slow.ipynb`, sección 7, "Verificación de integridad temporal por clase de ataque") ya había identificado 6 filas de DoS slowloris (0.1% de 5,796) con Timestamp anterior a las 9:47 a.m. La ventana documentada del ataque es 9:47-10:10 a.m. Se inspeccionaron individualmente: las 6 son puerto 80, con Flow Duration ≈ 5,000,000 µs (~83 min), consistentes con el propio mecanismo de Slowloris (mantiene conexiones HTTP abiertas deliberadamente). CICFlowMeter registra el Timestamp como el inicio del flujo, no el momento del ataque, así que estas 6 filas son legítimas: se conservaron en el EDA, no se trataron como ruido. Un límite basado en el mínimo/máximo exacto habría absorbido estas 6 filas dentro de la definición del bloque 2 (ventana de Slowloris), extendiéndola artificialmente ~7 horas hacia atrás (el mínimo real de Slowloris es las 02:24 a.m.). Usar cuantiles en vez de extremos exactos evita ese problema.

Validación empírica de `outlier_quantile=0.01` (contra `load_and_clean('data/raw')` regenerado, 450,382 filas; ver la nota de reconciliación de datos más abajo):

```text
Fronteras de bloque (outlier_quantile=0.01):
  Slowloris:    [2017-07-05 09:48:00 .. 2017-07-05 10:10:00]  (5796 filas)
  Slowhttptest: [2017-07-05 10:15:00 .. 2017-07-05 10:37:00]  (5499 filas)

Bloque 1: 280280 filas | Timestamp [01:00:00 .. 09:47:00] | ataques: 6     (0.00%)
Bloque 2:  30170 filas | Timestamp [09:48:00 .. 10:10:00] | ataques: 5788 (19.18%)
Bloque 3:   4132 filas | Timestamp [10:11:00 .. 10:14:00] | ataques: 2     (0.05%)
Bloque 4:  33830 filas | Timestamp [10:15:00 .. 10:37:00] | ataques: 5499 (16.25%)
Bloque 5: 101970 filas | Timestamp [10:38:00 .. 12:59:00] | ataques: 0     (0.00%)

Train: 315267 filas | Val: 67558 filas | Test: 67557 filas
Proporción de ataques -> train: 0.0256, val: 0.0312, test: 0.0163
```

A q=0.01, el bloque 1 aísla exactamente las 6 filas anómalas conocidas de Slowloris (ni más ni menos): el límite queda en 09:48, un minuto después del inicio documentado del ataque. El bloque 3 solo recibe 2 filas de "spillover" de Slowloris (0.05% de ese bloque), y la ventana de Slowhttptest queda completamente intacta, sin recorte, porque su densidad de filas por minuto en ese tramo hace que el 1% de cada cola no alcance a mover el límite (no hay anomalías conocidas de todas formas en esa clase).

Se probó también q=0.001 (más cercano a la tasa real de anomalía, 0.1%), pensando que sería "más preciso". El resultado fue peor: con tan pocas observaciones cerca de la cola y granularidad de minuto en Timestamp, la interpolación de cuantiles se vuelve inestable y el límite calculado queda en 09:38:21, nueve minutos antes del inicio documentado del ataque, arrastrando ~8,000 filas puramente benignas hacia el bloque 2 sin ninguna justificación. Se descarta 0.001 por esta razón; 0.01 queda como default.

Bloque 3 (el hueco entre Slowloris y Slowhttptest, apenas ~4 minutos de duración real) no resultó degenerado pese a su angostura temporal: 4,132 filas, 2,892/620/620 en el corte 70/15/15. Cae en horario de tráfico denso (mediodía), lo que compensa lo angosto de la ventana.

## Corte 70/15/15 por bloque

Dentro de cada bloque, ya en orden cronológico, se corta por posición: el primer 1 - test_size - val_size (default 70%) de las filas va a train, el siguiente val_size (default 15%) a val, y el test_size final (default 15%) a test. El resultado se guarda en una columna interna split ("train"/"val"/"test").

`random_state` se mantiene en la firma de `train_test_split_by_flow()` solo por compatibilidad con código y documentación existente. Ya no tiene ningún efecto sobre el resultado, porque el split ahora es determinista (posición cronológica, no muestreo aleatorio).

La proporción de ataques no es igual entre train/val/test, y eso es por diseño (ver los números de la validación empírica arriba: train 2.56%, val 3.12%, test 1.63%). Es la consecuencia esperada de que test sea literalmente "el último tramo cronológico" de cada bloque, y el tráfico de ataque no se lanza a una tasa perfectamente uniforme dentro de su ventana. No es un bug, ni algo que deba corregirse forzando una proporción igual: eso requeriría volver a un split no-cronológico, deshaciendo el propósito de este cambio.

## Columnas: qué entra y qué sale de X

**Siempre excluidas de X:**

* Label: etiqueta original en texto; se excluye para evitar fuga directa (igual que antes de este cambio).
* label_binaria: es el target (y).
* split: columna interna usada solo para hacer el corte; nunca debe llegar a X.

**Preservadas en X como metadata (no features):**

* Timestamp: igual que antes de este cambio.
* bloque: columna entera 1-5, nueva en este cambio.

Ambas son necesarias para la Fase 5 (`src/models/gru/`): una ventana de secuencia del GRU solo puede construirse con filas que compartan el mismo split y el mismo bloque, lo que evita cruzar discontinuidades temporales sin necesidad de una lógica de recorte manual ("embargo"; ver la sección "Qué sigue" más abajo). SVM y XGBoost deben descartar Timestamp y bloque antes de entrenar si no las usan como feature.

## scale_features()

Nueva función que ajusta un StandardScaler únicamente sobre X_train (`.fit_transform`) y aplica esa misma transformación (`.transform()`, nunca un nuevo `.fit()`) a X_val y X_test, para no filtrar información de validación/test hacia los parámetros de escalado. Por default escala todas las columnas numéricas excepto Timestamp (no es numérica, es datetime) y bloque (metadata). Devuelve también el scaler ya ajustado, para poder guardarlo (joblib) y reusarlo en inferencia.

La usan GRU y SVM, ambos sensibles a la escala de las features. XGBoost no la necesita: los splits de árboles son invariantes a transformaciones monótonas como el escalado estándar, así que puede ignorarla.

## Alternativas consideradas

* Mantener el split aleatorio estratificado. Descartada: es la causa raíz del problema que motiva este ADR, hace imposible construir secuencias coherentes para el GRU en Fase 5.
* outlier_quantile=0.001 (para acercarse más a la tasa real de anomalía conocida, 0.1%). Descartada tras validación empírica: con pocas observaciones cerca de la cola y granularidad de minuto en Timestamp, la interpolación de cuantiles se vuelve inestable y produce un límite peor (9 minutos antes de lo esperado) que con 0.01 (ver detalle arriba).
* Mínimo/máximo exacto en vez de cuantiles para las fronteras de bloque. Descartada: absorbería las 6 filas anómalas conocidas de Slowloris dentro del bloque 2, extendiendo su definición ~7 horas hacia atrás sin justificación real.
* Parametrizar columnas_a_excluir_de_X (como en la versión anterior de la función). Descartada: nunca se llamó con un valor distinto al default en ningún lugar del repo, y "split" debe excluirse siempre sin excepción, una lista configurable solo añadía el riesgo de que alguien olvidara incluirla.

## Consecuencias

* Esta función es compartida por los tres modelos, así que Juan Pablo (SVM) y Edwin (XGBoost) deben construir sus notebooks contra esta versión. Cualquier resultado ya obtenido contra el split aleatorio anterior queda invalidado y debe regenerarse antes de compararse en Fase 7-8 (aviso ya coordinado con el equipo antes de esta implementación).
* La interfaz de salida no cambia: sigue siendo una tupla de 6 elementos (`X_train, X_val, X_test, y_train, y_val, y_test`), así que el único lugar del repo que la invoca (`notebooks/gru/00_setup_ejemplo.md`, un doc de referencia no ejecutado) sigue siendo válido tal cual.
* X ahora incluye dos columnas de metadata (Timestamp, bloque) que no son features. SVM y XGBoost deben recordar descartarlas antes de entrenar.
* La proporción de ataque difiere entre splits (ver arriba); debe tenerse en cuenta al interpretar métricas de validación/test, no es indicativo de un split roto.
* Se introduce StandardScaler/sklearn.preprocessing en el repo por primera vez.

## Validación

Sin suite de tests en el repo en ese momento, la verificación

```python
from src.preprocessing.clean import load_and_clean, train_test_split_by_flow, scale_features

df = load_and_clean("data/raw")
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_flow(df)
X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)
```

Resultados (ver también los prints reproducidos en la sección de diseño arriba):

* Ningún bloque quedó vacío; ningún split quedó con 0 ejemplos de ataque (criterio de aceptación cumplido).
* `'split' in X_train.columns → False` (confirmado, no debe estar ahí).
* `'bloque' in X_train.columns` y `'Timestamp' in X_train.columns → True` (confirmado).
* `scale_features()` escaló 65 de 67 columnas de X (excluyó correctamente Timestamp y bloque), ajustando el scaler solo sobre X_train.
* Total de filas repartidas: 315,267 + 67,558 + 67,557 = 450,382, consistente con el total de `load_and_clean()`.

## Nota de reconciliación de datos: 450,382 vs. 450,963 filas

Durante esta implementación se detectó que el CSV procesado localmente reporta 450,382 filas tras `load_and_clean()`, mientras que la Sección 7 de `notebooks/Cleaning/limpieza_low_and_slow.ipynb` ("Verificación de integridad temporal por clase de ataque") cita 450,963. La causa raíz está identificada: no es un bug de datos ni de código.

450,382 (439,087 BENIGN + 5,796 Slowloris + 5,499 Slowhttptest) es la cifra correcta y vigente. La produce el commit 53158f9 ("feat: drop redundant/constant columns and anomalous negative rows in basic_cleaning"), que agregó el filtro de filas con valores negativos anómalos en `basic_cleaning()` (confirmado 100% clase BENIGN, ver el comentario en el propio filtro, `src/preprocessing/clean.py`). El commit posterior 71008bc ("chore: re-run cleaning notebook with updated basic_cleaning") volvió a ejecutar las celdas principales del pipeline de ese notebook, pero no volvió a ejecutar la Sección 7 (agregada antes de ese filtro), cuyos números (450,963 filas, 439,668 BENIGN) quedaron desactualizados. La aritmética lo confirma exactamente: 439,668 − 439,087 = 581, que son las 581 filas BENIGN que ese filtro elimina. Los conteos de ataque (5,796 Slowloris + 5,499 Slowhttptest) son idénticos en ambas versiones, porque el filtro no toca esa clase.

Esta explicación queda documentada en tres lugares consistentes, como única fuente de verdad citada desde cada uno: este ADR y un comentario en el propio filtro dentro de `src/preprocessing/clean.py`.

No se requiere ningún cambio de código adicional por esto. Como limpieza de higiene menor (fuera del alcance de esta rama), sería conveniente que alguien del equipo re-ejecute `notebooks/Cleaning/limpieza_low_and_slow.ipynb` completo ("Restart & Run All") en un commit chore: aparte, para que la Sección 7 vuelva a coincidir con el resto del notebook.

## Qué sigue (Fase 5, fuera de esta rama)

Al construir las secuencias del GRU, una ventana solo debe formarse con filas que compartan el mismo split y el mismo bloque. Esto resuelve automáticamente el problema de las discontinuidades temporales invisibles en las costuras entre bloques (por ejemplo, entre el final de train en el bloque 1 y el inicio de train en el bloque 2 existieron, en el tiempo real, las filas de val/test del bloque 1, que no están en ese X_train) sin necesidad de una lógica de recorte manual adicional ("embargo"): las primeras W−1 filas de cada segmento split+bloque simplemente no tienen suficiente historia y se descartan ahí, no en `train_test_split_by_flow()`.
