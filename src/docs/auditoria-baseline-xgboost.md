# Auditoría de XGBoost: pipeline actual (v2)

Este documento describe el pipeline de XGBoost tal como está hoy (v2) y lo que se sabe de él. Las cifras salen de las salidas de los notebooks y de los JSON guardados; no se recalculó ninguna métrica para escribirlo.

| Pieza | Dónde está |
|---|---|
| Limpieza compartida | `src/preprocessing/clean.py`, `notebooks/Cleaning/limpieza_low_and_slow.ipynb` |
| Preparación de datos | `notebooks/xgboost/01_feature_engineering.ipynb` |
| Baseline sin tratamiento del desbalance | `notebooks/xgboost/02_modelo.ipynb` |
| Ablación de SMOTE y pesos de clase | `notebooks/xgboost/02b_ablacion_smote.ipynb` |
| Tuning | `notebooks/xgboost/03_tuning.ipynb` |
| Umbral | `notebooks/xgboost/04_umbral.ipynb` |
| Módulo y modelo congelado | `src/models/xgboost/model.py`, `notebooks/xgboost/05_verificacion_modulo.ipynb`, `models/xgboost/` |
| Evaluación final en test | `notebooks/xgboost/06_evaluacion_final.ipynb`, `reports/metrics/`, `reports/xgboost/` |
| Versión anterior (v1) | tag `xgboost-eval-v1`, `reports/xgboost/v1/` |
| Versión actual (v2) | tags `xgboost-v2-frozen` (modelo congelado) y `xgboost-eval-v2` (evaluación en test) |

Las versiones anteriores de este documento, con la auditoría de los baselines de 59 y 57 features, siguen en el historial de git (por ejemplo, el commit `3fe2343`).

Los conteos del dataset limpio original, 450,376 filas y 439,087 BENIGN, son los del pipeline v1. Se verificaron en la salida versionada de `notebooks/Cleaning/limpieza_low_and_slow.ipynb` en el tag `xgboost-eval-v1` ("Tras excluir fuera de ventana: 450,376" y "BENIGN: 439,087 | ataque: 11,289"). 450,376 aparece también en la salida de `01_feature_engineering.ipynb` en ese mismo tag.

## Contexto

| | v2 (actual) |
|---|---|
| Datos | Captura del miércoles 5/7/2017 de CIC-IDS2017 (`GeneratedLabelledFlows`), solo BENIGN, Slowloris y Slowhttptest, con la corrección de ruido de etiqueta: 443,614 filas |
| Split | Cronológico por bloques (ADR 0001): 310,529 train / 66,541 val / 66,544 test |
| Features | 55 |
| Desbalance | Sin SMOTE; `scale_pos_weight` = benignos / ataques de `y_train` = 302,299 / 8,230 = 36.7313 |
| Modelo | XGBoost 3.3.0, `tree_method="hist"`, `learning_rate=0.3`, `max_depth=4`, `min_child_weight=1`, `n_estimators=1000`, `early_stopping_rounds=50` sobre val con `eval_metric="aucpr"`, `random_state=42`; `best_iteration` 68 |
| Umbral | 0.87: mayor F2 con recall >= 0.75 en val, mediana de 1,000 remuestreos bootstrap |
| Criterio de aceptación | recall >= 0.75 en la clase ataque |

## Origen de los datos y verificación de etiquetas

`load_raw_csvs` carga `Wednesday-workingHours.pcap_ISCX.csv`, de la variante `GeneratedLabelledFlows`, que trae `Timestamp`, `Flow ID` e IPs. El notebook de limpieza verifica sobre los datos crudos dos cosas que determinan decisiones posteriores:

- **`Timestamp`:** ninguna cadena trae AM/PM y las horas parseadas son 1–5 y 8–12. `basic_cleaning` corrige ese reloj de 12 horas.
- **Redundancia:** `SYN Flag Count` coincide con `Fwd PSH Flags` en el 100.0000 % de las filas, así que se conserva solo una de las dos.

El modelo nunca ve la columna `Label`. 06 la reconstruye con `train_test_split_by_flow(return_meta=True)` para reportar por tipo de ataque, y antes de usarla verifica que la partición reconstruida coincida fila por fila con `X_test.csv` / `y_test.csv`: la etiqueta binaria y las 55 features, idénticas bit a bit.

### Corrección de ruido de etiqueta (v2)

Los atacantes aparecen con la IP de la interfaz NAT del firewall (172.16.0.1) y la víctima es 192.168.10.50. Un compañero encontró que los flujos BENIGN de ese par, en sentido víctima → NAT, son el lado servidor de conexiones de ataque. Según el análisis documentado en el notebook de limpieza, se emparejan con flujos de ataque por el puerto del cliente y ocurren solo mientras hay ataques.

Esos flujos se **excluyen** y no se reetiquetan (`excluir_respuestas_de_ataque` en `clean.py`). Así queda el pipeline de limpieza, según la salida del notebook de limpieza:

| Etapa | Filas |
|---|---|
| Crudas (miércoles) | 692,703 |
| Tras filtro BENIGN + Slowloris + Slowhttptest | 451,326 |
| Respuestas de ataque etiquetadas BENIGN excluidas (todas víctima → NAT) | 6,789 (1.54 % de 440,031 BENIGN) |
| Tras excluir esas respuestas | 444,537 |
| Tras la limpieza básica | 443,620 |
| Tras excluir 6 Slowloris fuera de su ventana documentada | **443,614** |

| Dataset limpio | Original (v1) | Corregido (v2) |
|---|---|---|
| Filas | 450,376 | 443,614 |
| BENIGN | 439,087 | 432,325 |
| Ataques | 11,289 | 11,289 (5,790 Slowloris y 5,499 Slowhttptest) |

La corrección solo quita filas benignas; los ataques no cambian.

Se excluyen 6,789 flujos, pero el dataset limpio baja solo 6,762 filas (450,376 → 443,614). La diferencia, 27 filas, aparece en `basic_cleaning`:

- En v1 (salida del notebook de limpieza en el tag `xgboost-eval-v1`), `basic_cleaning` pasa de 451,326 a 450,382 filas: elimina 944. Después, la exclusión de los 6 Slowloris fuera de ventana deja 450,376. En v1 no había exclusión de respuestas.
- En v2, `basic_cleaning` pasa de 444,537 a 443,620 filas: elimina 917.

La diferencia, 944 − 917 = 27, coincide con la del filtro de valores negativos anómalos, que elimina 581 filas en v1 y 554 en v2. Es decir, 27 de los flujos excluidos ya los habría eliminado ese filtro.

Las otras dos cifras que imprime `basic_cleaning` **no son filas eliminadas** y no se pueden sumar con las anteriores:

- "Valores nulos/infinitos detectados: 696" cuenta **celdas** nulas o infinitas, no filas, y una fila puede tener varias.
- "Filas duplicadas exactas detectadas: 16" se cuenta **antes** de eliminar las filas con nulos, así que puede solaparse con ellas.

Las dos son iguales en v1 y v2. El notebook no imprime cuántas filas elimina cada uno de esos dos pasos.

Las cifras de v2 (554, 917 y las de la tabla de etapas de arriba) salen de ejecutar el notebook de limpieza sobre el dataset corregido; la versión del notebook guardada en el repositorio no conserva esas salidas.

## 1. Composición por split

Mismo split cronológico por bloques en v1 y v2: cinco bloques definidos por las ventanas de cada ataque (Slowloris 09:48–10:11, Slowhttptest 10:15–10:37), cada uno cortado 70/15/15 en orden cronológico.

| Bloque (v2) | Filas | Ataques |
|---|---|---|
| 1 (antes de Slowloris) | 53,375 | 0 |
| 2 (Slowloris) | 30,362 | 5,790 (19.07 %) |
| 3 (entre ataques) | 3,578 | 0 |
| 4 (Slowhttptest) | 33,203 | 5,499 (16.56 %) |
| 5 (después de Slowhttptest) | 323,096 | 0 |

| Split | v1: filas | v1: ataques | v2: filas | v2: ataques |
|---|---|---|---|---|
| train | 315,264 | 8,230 | 310,529 | 8,230 |
| val | 67,556 | 1,955 (789 Slowloris + 1,166 Slowhttptest) | 66,541 | 1,884 (789 Slowloris + 1,095 Slowhttptest) |
| test | 67,556 | 1,104 (915 Slowloris + 189 Slowhttptest) | 66,544 | 1,175 (915 Slowloris + 260 Slowhttptest) |

El bloque 4 perdió filas benignas con la corrección. Como el corte es por posición dentro del bloque, las fronteras 70/15/15 se movieron y **71 ataques Slowhttptest pasaron de val a test**. Train no cambió en ataques.

En v2, la proporción de ataques es 2.6503 % en train, 2.8313 % en val y 1.7657 % en test. Que difiera entre particiones es esperado: el split es cronológico, no estratificado.

## 2. Duplicados de features

`reportar_duplicados_de_features` (01, solo diagnóstico, no elimina filas) cuenta vectores de las 55 features repetidos dentro de cada split y cuántos de val y test ya aparecieron en train:

| split | clase | filas | repetidos dentro del split | también en train |
|---|---|---|---|---|
| train | benigno | 302,299 | 29,556 | — |
| train | ataque | 8,230 | 288 | — |
| val | benigno | 64,657 | 4,807 | 7,848 |
| val | ataque | 1,884 | 180 | 121 |
| test | benigno | 65,369 | 4,142 | 8,084 |
| test | ataque | 1,175 | 109 | 33 |

Ningún vector aparece con las dos etiquetas. Estos grupos de flujos repetidos o casi repetidos importan para la incertidumbre (ver Limitaciones).

## 3. Features: 55 y columnas excluidas

`clean.py` (pipeline compartido por los tres modelos) elimina:

- **Identificadores de conexión:** `Flow ID`, `Source IP`, `Destination IP`, `Source Port`.
- **Atajos del laboratorio:** `Destination Port`, `Protocol`. En los datos filtrados, el 100 % de los ataques va al puerto 80, contra el 11.08 % de los benignos. Todos los ataques son TCP, mientras que en los benignos el protocolo 6 es el 53.81 % y el 17 el 46.12 %. Estas columnas separan las clases por cómo se armó la captura, no por el comportamiento lento.
- **Redundantes:** `Subflow Fwd Packets`, `Fwd Header Length.1`, `Avg Fwd Segment Size`, `Subflow Fwd Bytes`, `Subflow Bwd Packets`, `Avg Bwd Segment Size`, `Subflow Bwd Bytes`, `Fwd PSH Flags` (idéntica a `SYN Flag Count`).
- **Constantes:** `Fwd URG Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`, `Bwd PSH Flags`, `CWE Flag Count`.

El CSV limpio queda con 59 features más `Timestamp`, `Label` y `label_binaria`. `01_feature_engineering.ipynb` quita además:

- `Timestamp` y `bloque`, que son metadatos del split;
- cuatro columnas casi gemelas de otras: `Flow IAT Std`, `Bwd IAT Std`, `Bwd IAT Mean` y `Packet Length Std`.

El resultado son **55 features**. 01 verifica con asserts que train, val y test tienen las mismas 55 columnas en el mismo orden, sin `Destination Port`, `Source Port`, `Protocol`, `Fwd PSH Flags`, `Timestamp` ni `bloque`.

**Historia.** Que los puertos son un atajo del laboratorio salió de la auditoría del primer baseline (59 features, dataset original, con SMOTE): quitar `Destination Port` bajó claramente el recall en val. En el baseline siguiente, de 57 features (sin los dos puertos), todos los falsos negativos en val eran Slowhttptest de las 10:32–10:33: flujos muy cortos y rápidos, sin la señal low & slow. Las cifras de esa auditoría corresponden a una configuración que ya no existe y están en la versión anterior de este documento en git. Después, la exclusión de puertos y `Protocol` pasó de 01 a `clean.py`, para que valga para los tres modelos.

## 4. Baseline sin tratamiento del desbalance (02)

`02_modelo.ipynb` entrena con valores por defecto, sin SMOTE ni pesos de clase (la variante B de 02b), y evalúa solo en val. Con el dataset corregido:

| Val | v1 (original) | v2 (corregido) |
|---|---|---|
| best_iteration | 114 | 55 |
| PR-AUC | 0.997764 | 0.994515 |
| ROC-AUC | 0.999938 | 0.999917 |
| Umbral 0.5: precision / recall / F1 | 0.9939 / 0.9217 / 0.9565 | 0.9920 / 0.9183 / 0.9537 |
| Umbral 0.5: FP / FN | 11 / 153 | 14 / 154 |

Verificaciones de fuga en v2:

- **Overfitting:** con umbral 0.5, el recall es 0.9996 en train y 0.9183 en val; FP 74 contra 14 y FN 3 contra 154.
- **Etiquetas de train mezcladas** (semillas 0–2): en val, ROC-AUC 0.4384–0.5400, PR-AUC 0.0509–0.0898 (la proporción de ataques en val, el nivel del azar, es 0.0283) y recall a 0.5 igual a 0. No hay señal de fuga.

## 5. Ablación de SMOTE (02b)

Las tres variantes usan los hiperparámetros del baseline y se evalúan en el mismo val:

- **A:** con SMOTE (`sampling_strategy=0.4`);
- **B:** train real, sin pesos;
- **C:** train real con `scale_pos_weight` = benignos / ataques de `y_train`.

**Regla pre-registrada** (escrita en el notebook antes de ver resultados). X "supera" a Y si:

1. su PR-AUC en val es mayor por más de 0.0005, y
2. su recall es estrictamente mayor con los dos presupuestos de FP, 10 y 66.

C reemplaza a B solo si la supera. A gana solo si supera a B y a C. Los empates favorecen a la variante más simple.

| Val | v1: A | v1: B | v1: C | v2: A | v2: B | v2: C |
|---|---|---|---|---|---|---|
| best_iteration | 64 | 114 | 189 | 41 | 55 | 52 |
| PR-AUC | 0.998612 | 0.997764 | 0.997060 | 0.998512 | 0.994515 | 0.997996 |
| ROC-AUC | 0.999960 | 0.999938 | 0.999910 | 0.999953 | 0.999917 | 0.999940 |
| recall@FP<=10 | 0.924808 | 0.914066 | 0.911509 | 0.916136 | 0.901274 | 0.916667 |
| recall@FP<=66 | 0.998977 | 0.998977 | 0.998977 | 0.998938 | 0.998408 | 0.998938 |
| FP / FN a 0.5 | 12 / 58 | 11 / 153 | 12 / 153 | 13 / 149 | 14 / 154 | 16 / 58 |

| Regla | v1: dif. PR-AUC | v1: supera | v2: dif. PR-AUC | v2: supera |
|---|---|---|---|---|
| C vs B | −0.000703 | no | +0.003481 | **sí** (los tres criterios) |
| A vs B | +0.000848 | no (empate en recall@FP<=66) | +0.003997 | sí |
| A vs C | +0.001551 | no (empate en recall@FP<=66) | +0.000516 | **no** (recall@FP<=10 menor; empate en recall@FP<=66) |
| **Ganadora** | **B** | | **C** | |

- Con el dataset original la regla eligió **B**; con el corregido eligió **C**. Desde 03 se entrena con C: `scale_pos_weight` = 302,299 / 8,230 = 36.7313 (en v1, la C de la ablación tenía 37.3067). `02_modelo.ipynb` se conserva como el baseline B de referencia.
- **Las diferencias que decidieron la regla en v2 son chicas.** En recall@FP<=66, C 0.998938 contra B 0.998408. En recall@FP<=10, A 0.916136 contra C 0.916667. La ventaja de A sobre C en PR-AUC (+0.000516) apenas pasa el margen de 0.0005.
- **Es una comparación con una sola semilla y un solo split.** Volver a ejecutar 02b en la misma máquina dio exactamente los mismos valores, pero no se midió cuánto cambiaría la decisión con otra semilla u otra partición.
- El presupuesto de 66 FP se definió como "≈ 0.1 % de los benignos de val" con el dataset original y se mantuvo literal; val corregido tiene 64,657 benignos.

## 6. Tuning (03)

Hay 18 combinaciones: `learning_rate` ∈ {0.3, 0.1, 0.05} × `max_depth` ∈ {4, 6, 8} × `min_child_weight` ∈ {1, 5}. Cada una usa `scale_pos_weight` fijo (36.7313), early stopping en val y no usa validación cruzada, porque el split es cronológico.

La regla, escrita antes de la grilla:

1. Solo son elegibles las combinaciones con recall >= 0.75 a umbral 0.5.
2. Se busca la mayor PR-AUC en val.
3. Se consideran equivalentes las que quedan a menos de 0.0005 de la mejor.
4. Entre ellas se desempata por menor `max_depth`, luego por menor best_iteration y por último por mayor PR-AUC.

La fila de valores por defecto (0.3 / 6 / 1, con `scale_pos_weight`) reproduce la variante C de 02b: best_iteration 52, FP 16, FN 58, PR-AUC 0.997996.

Resultado: las 18 combinaciones son elegibles. La mejor PR-AUC es 0.998817 (0.1 / 4 / 5) y **12 combinaciones quedan dentro del margen** (PR-AUC > 0.998317). En orden de desempate:

| lr / depth / mcw | best_iteration | PR-AUC |
|---|---|---|
| **0.3 / 4 / 1** | **68** | **0.998685** |
| 0.3 / 4 / 5 | 138 | 0.998450 |
| 0.1 / 4 / 5 | 371 | 0.998817 |
| 0.05 / 4 / 5 | 442 | 0.998612 |
| 0.05 / 4 / 1 | 517 | 0.998468 |
| 0.3 / 6 / 5 | 82 | 0.998612 |
| 0.1 / 6 / 5 | 187 | 0.998489 |
| 0.05 / 6 / 5 | 349 | 0.998666 |
| 0.3 / 8 / 5 | 51 | 0.998505 |
| 0.1 / 8 / 1 | 148 | 0.998338 |
| 0.05 / 8 / 5 | 285 | 0.998356 |
| 0.05 / 8 / 1 | 294 | 0.998405 |

**Gana 0.3 / 4 / 1.** El reentrenamiento coincide con su fila de la grilla.

| Val, umbral 0.5 | baseline 0.3 / 6 / 1 | elegida 0.3 / 4 / 1 |
|---|---|---|
| best_iteration | 52 | 68 |
| PR-AUC | 0.997996 | 0.998685 |
| ROC-AUC | 0.999940 | 0.999967 |
| precision / recall / F1 | 0.991314 / 0.969214 / 0.980140 | 0.993136 / 0.998408 / 0.995765 |
| FP / FN | 16 / 58 | 13 / 3 |

En v1 (dataset original, sin pesos), 7 combinaciones quedaron dentro del margen y ganó 0.3 / 6 / 1, con best_iteration 114 y PR-AUC 0.997764.

## 7. Umbral (04)

**Regla:**

- **Grilla:** 108 umbrales, de 0.001 a 0.009 en pasos de 0.001 y de 0.01 a 0.99 en pasos de 0.01. La parte por debajo de 0.01 ya está en la primera y única versión de 04 en v1 (commit `ee947ab`, incluido en el tag `xgboost-eval-v1`); no hay en git una versión de 04 sin ella. El propio notebook dice que se agregó antes de ejecutarlo porque en un análisis previo el mejor umbral cayó en el borde 0.01. Eso coincide con la salida de `02_modelo.ipynb` en ese tag, donde la búsqueda fina (de 0.01 a 0.99) da 0.01 como mejor umbral. El orden de ejecución no se puede confirmar con el repositorio más allá de eso. La grilla se mantuvo en v2.
- **Elegibilidad:** recall >= 0.75 en val.
- **Objetivo:** **F2** (F-beta con beta = 2).
- **Desempate:** el umbral más alto.
- **Bootstrap:** 1,000 remuestreos de val estratificados por clase, con semilla 42.
- **Umbral final:** la mediana de los umbrales ganadores.

El único cambio deliberado respecto de v1 es el objetivo: de F1 a F2, para usar el mismo criterio que SVM y GRU. F1 se sigue reportando.

Resultados en val:

- **Reproducción:** el modelo reproduce 03 (best_iteration 68, FP 13 y FN 3 a 0.5).
- **(a) Val completo:** los 108 umbrales son elegibles. Empatan en el F2 máximo 0.19 y 0.20, y el desempate da 0.20.
- **(b) Bootstrap bimodal:** 494 remuestreos eligen 0.20, 503 eligen 0.87 y 3 eligen 0.91.

| mín | p2.5 | p5 | p25 | mediana | p75 | p95 | p97.5 | máx |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.2 | 0.2 | 0.2 | **0.87** | 0.87 | 0.87 | 0.87 | 0.91 |

- **(c) Umbral final = 0.87.** Los valores ordenados 500 y 501 son ambos 0.87, así que la mediana cae sobre la grilla. Las dos modas casi no se distinguen en val completo: F2 vale 0.997562 con 0.20 y 0.997560 con 0.87.
- **Meseta y borde superior.** En val, todos los umbrales de 0.81 a 0.87 dan el mismo resultado: precision 0.994186, recall 0.998408, FP 11, FN 3. 0.87 es el extremo superior de esa meseta: 0.88 ya no está entre los 10 umbrales de mayor F2 de 04, y la tabla informativa de 06 muestra que en 0.90 los FN de val suben a 8 y en 0.95 a 59.
- **F1 y F2 eligen el mismo umbral final.** Como registro, 04 aplica la regla anterior (mayor F1) a los mismos remuestreos. En val completo elige 0.87 y la mediana del bootstrap también es 0.87.

| Val | umbral 0.5 | 0.20 (ganador en val completo) | **0.87 (final)** |
|---|---|---|---|
| precision | 0.993136 | 0.992093 | 0.994186 |
| recall | 0.998408 | 0.998938 | 0.998408 |
| F2 | 0.997349 | 0.997562 | 0.997560 |
| F1 | 0.995765 | 0.995504 | 0.996292 |
| FP / FN | 13 / 3 | 15 / 2 | 11 / 3 |

Tabla informativa de val en 06 (no elige ni cambia nada):

| umbral | precision | recall | F1 | F2 | FP | FN |
|---|---|---|---|---|---|---|
| 0.50 | 0.9931 | 0.9984 | 0.9958 | 0.9973 | 13 | 3 |
| 0.20 | 0.9921 | 0.9989 | 0.9955 | 0.9976 | 15 | 2 |
| 0.80 | 0.9937 | 0.9984 | 0.9960 | 0.9975 | 12 | 3 |
| **0.87** | **0.9942** | **0.9984** | **0.9963** | **0.9976** | **11** | **3** |
| 0.90 | 0.9942 | 0.9958 | 0.9950 | 0.9954 | 11 | 8 |
| 0.95 | 0.9940 | 0.9687 | 0.9812 | 0.9736 | 11 | 59 |

## 8. Modelo congelado y reproducibilidad en GPU

`src/models/xgboost/model.py` fija `HIPERPARAMETROS` (0.3 / 4 / 1 más la configuración fija) y `UMBRAL_FINAL = 0.87`. `scale_pos_weight` no es una constante: `train_xgboost_model` lo calcula de `y_train` cada vez. `save_xgboost_model` lo guarda en los metadatos junto con sus conteos y verifica antes que coincida con el modelo:

```
umbral 0.87 | scale_pos_weight 36.73134872417983 | n_benignos_train 302299 | n_ataques_train 8230 | best_iteration 68
```

En `05_verificacion_modulo.ipynb`:

- el módulo reproduce 03/04 en val: best_iteration 68, con FP 13 / FN 3 a 0.5 y FP 11 / FN 3 a 0.87;
- a 0.87 da recall 0.9984, precision 0.9942, F1 0.9963 y F2 0.9976;
- guardar y volver a cargar da probabilidades idénticas en val (diferencia máxima 0);
- el `scale_pos_weight` del booster cargado (36.7313499, en float32) coincide con el de los metadatos.

**Nota de reproducibilidad.** El entrenamiento corre en GPU (`device="cuda"`) y la predicción en CPU. El entrenamiento en GPU puede no reproducirse bit a bit en CPU, por eso 06 **no reentrena**: carga el modelo guardado en JSON (`models/xgboost/xgboost_final.json`). En esta máquina, reentrenar en GPU con la misma configuración dio siempre el mismo resultado: 02b ejecutado dos veces, 04 dos veces y el reentrenamiento de referencia del análisis de sensibilidad, que dio probabilidades idénticas a las del modelo congelado en val.

## 9. Evaluación final en test (06)

06 hace todas las verificaciones sobre los datos de test antes de predecir, y predice test **una sola vez** con el modelo y el umbral congelados. Las probabilidades quedan en `reports/xgboost/probabilidades_test.csv`. Las métricas compartidas están en `reports/metrics/xgboost.json` (test completo) y `reports/metrics/xgboost_comun.json` (subconjunto común); todo lo demás, en `reports/xgboost/resultados_test.json`.

**Filas comunes con la GRU.** `results/gru/03_evaluacion_test/idx_test_comun.npy` trae las etiquetas del índice de las 66,454 filas de test que evaluó la GRU. 06 verifica que todas existen en el índice de test antes de predecir. Las métricas del subconjunto salen de filtrar las probabilidades ya guardadas; no hay una segunda predicción.

Quedan fuera de la lista 90 filas de test: 79 BENIGN, 1 Slowloris y 10 Slowhttptest.

| bloque | BENIGN | Slowhttptest | Slowloris |
|---|---|---|---|
| 1 | 18 | 0 | 0 |
| 2 | 17 | 0 | 1 |
| 3 | 18 | 0 | 0 |
| 4 | 8 | 10 | 0 |
| 5 | 18 | 0 | 0 |

**Resultados a umbral 0.87.** Entre corchetes va el IC 95 % por bootstrap: 1,000 remuestreos estratificados en benignos, Slowloris y Slowhttptest, semilla 42, percentiles 2.5 y 97.5. Ese bootstrap solo remuestrea las predicciones ya hechas.

| Test, umbral 0.87 | test completo | subconjunto común (GRU) |
|---|---|---|
| flujos / ataques | 66,544 / 1,175 (915 Slowloris, 260 Slowhttptest) | 66,454 / 1,164 (914 Slowloris, 250 Slowhttptest) |
| recall | 0.9906 [0.9847, 0.9966] | 0.9905 [0.9845, 0.9957] |
| precision | 0.9932 [0.9881, 0.9974] | 0.9931 [0.9880, 0.9974] |
| F1 | 0.9919 [0.9880, 0.9953] | 0.9918 [0.9883, 0.9953] |
| F2 | 0.9911 [0.9862, 0.9957] | 0.9911 [0.9864, 0.9952] |
| FP | 8 [3, 14] | 8 [3, 14] |
| FN / TP / TN | 11 / 1,164 / 65,361 | 11 / 1,153 / 65,282 |
| PR-AUC / ROC-AUC | 0.9977 / 0.9999 | 0.9977 / 0.9999 |
| recall Slowloris | 0.9913 [0.9847, 0.9967] (907/915) | 0.9912 [0.9847, 0.9967] (906/914) |
| recall Slowhttptest | 0.9885 [0.9731, 1.0000] (257/260) | 0.9880 [0.9720, 1.0000] (247/250) |

El recall en test cumple el criterio de aceptación (>= 0.75).

**Sensibilidad (umbrales no elegidos, solo sensibilidad):**

| | umbral | precision | recall | F1 | F2 | FP | FN |
|---|---|---|---|---|---|---|---|
| test completo | 0.5 | 0.9923 | 0.9923 | 0.9923 | 0.9923 | 9 | 9 |
| test completo | 0.20 | 0.9898 | 0.9932 | 0.9915 | 0.9925 | 12 | 8 |
| subconjunto común | 0.5 | 0.9923 | 0.9923 | 0.9923 | 0.9923 | 9 | 9 |
| subconjunto común | 0.20 | 0.9897 | 0.9931 | 0.9914 | 0.9924 | 12 | 8 |

## 10. Historia: pipeline v1 (tag `xgboost-eval-v1`)

v1 recorrió todo el flujo una vez con el dataset original (450,376 filas) y quedó archivado en el tag `xgboost-eval-v1` y en `reports/xgboost/v1/`. Sus diferencias con v2:

- datos sin la corrección de ruido de etiqueta, con 1,104 ataques en test (189 Slowhttptest);
- variante B (sin pesos de clase), elegida por la regla de 02b;
- hiperparámetros 0.3 / 6 / 1, con best_iteration 114;
- umbral elegido con objetivo **F1**, sobre la grilla extendida por debajo de 0.01: **0.002**. En val, 999 de los 1,000 remuestreos eligieron 0.002 (el otro, 0.007). Con 0.002, val dio precision 0.9726, recall 0.9990, F1 0.9856, FP 55 y FN 2.

Resultados de v1 en test (`reports/xgboost/v1/resultados_test.json`), umbral 0.002:

| Test v1 | valor [IC 95 %] |
|---|---|
| flujos / ataques | 67,556 / 1,104 (915 Slowloris, 189 Slowhttptest) |
| recall | 0.9728 [0.9629, 0.9819] |
| precision | 0.9521 [0.9397, 0.9640] |
| F1 | 0.9624 [0.9541, 0.9699] |
| FP | 54 [40, 69] |
| FN / TP / TN | 30 / 1,074 / 66,398 |
| PR-AUC / ROC-AUC | 0.9738 / 0.9792 |
| recall Slowloris | 0.9694 [0.9584, 0.9803] (887/915) |
| recall Slowhttptest | 0.9894 [0.9735, 1.0000] (187/189) |
| sensibilidad 0.5 | recall 0.8188, precision 0.9923, F1 0.8973, FP 7, FN 200 |
| sensibilidad 0.01 | recall 0.9484, precision 0.9785, F1 0.9632, FP 23, FN 57 |

**De v1 a v2 cambiaron varias cosas a la vez:**

- los datos (exclusión de las respuestas del servidor);
- la composición de val y test (71 Slowhttptest pasaron de val a test);
- el tratamiento del desbalance (B → C);
- los hiperparámetros (`max_depth` 6 → 4);
- el objetivo del umbral (F1 → F2) y el umbral mismo (0.002 → 0.87).

Las métricas de test mejoraron, pero este pipeline no permite atribuir esa mejora a ninguno de esos cambios en particular.

## 11. Diagnóstico interno

Los análisis de esta sección son descriptivos y no cambian el modelo, el umbral ni los resultados. Están en `analisis_interno/diagnostico_xgboost.ipynb` y `analisis_interno/sensibilidad_tcp.ipynb`, una carpeta excluida de git que **no es parte del repositorio**.

- El diagnóstico lee el modelo congelado de `models/xgboost/` y las probabilidades de test guardadas en `reports/xgboost/probabilidades_test.csv`.
- El análisis de sensibilidad TCP usa solo train y val, nunca carga test y entrena variantes reducidas que no se guardan.

Ninguno de los dos reentrena ni reemplaza el modelo congelado, ni vuelve a predecir test.

### Importancia de features

El booster guardado tiene 119 árboles, porque el early stopping sigue 50 rondas después de la mejor; la predicción usa los primeros 69. Sobre esos 69, el gain está muy concentrado:

| Proporción del gain | sobre gain promedio | sobre total_gain |
|---|---|---|
| top 1 (`Bwd Packet Length Mean`) | 0.5631 | 0.5380 |
| top 3 | 0.7650 | 0.7757 |
| top 5 | 0.8640 | 0.8666 |

Por gain, el top 5 es `Bwd Packet Length Mean`, `Fwd IAT Total`, `Total Length of Fwd Packets`, `Fwd IAT Mean` y `Fwd Packet Length Min`. El modelo usa 49 de las 55 features; nunca usa `Bwd Packet Length Min`, `Bwd Packet Length Max`, `ACK Flag Count`, `RST Flag Count`, `FIN Flag Count` ni `ECE Flag Count`.

### Los 11 falsos negativos de test

Todos caen al final de las ventanas de ataque, y los 11 están dentro de la lista de filas comunes con la GRU.

- **8 Slowloris:** todos con `Timestamp` 10:10, el 95.7 % de la ventana 09:48–10:11. Son los Slowloris número 767 a 913 de los 915 que hay en test.
- **3 Slowhttptest:** uno a las 10:34 (86.4 % de la ventana 10:15–10:37) y dos a las 10:37. Esos dos son los últimos flujos Slowhttptest de todo el dataset (5,498 y 5,499 de 5,499).
- **Vectores:** son 11 vectores de features distintos y ninguno aparece en train. Los pares más parecidos difieren en 9 de las 55 features (tres Slowloris entre sí) y en 10 (otros dos Slowloris, que tienen la misma probabilidad).

| Probabilidad | TP (1,164) | FN (11) |
|---|---|---|
| [0, 0.001) | 0 | 2 |
| [0.001, 0.01) | 0 | 4 |
| [0.01, 0.1) | 0 | 2 |
| [0.2, 0.5) | 0 | 1 |
| [0.8, 0.87) | 0 | 2 |
| [0.87, 0.9) | 3 | 0 |
| [0.9, 0.95) | 24 | 0 |
| [0.95, 0.99) | 8 | 0 |
| [0.99, 0.999) | 117 | 0 |
| [0.999, 1] | 1,012 | 0 |

**El umbral está cerca del borde inferior de la distribución de los TP.** La probabilidad mínima de un TP en test es 0.877938 (percentil 1: 0.93545; mediana: 0.999905), y los dos FN más altos tienen 0.860831. De los otros 9 FN, el mayor tiene 0.326815 y 8 están por debajo de 0.03.

### Sensibilidad a features de huella TCP

Este análisis está en `analisis_interno/sensibilidad_tcp.ipynb`, que **no es parte del repositorio**. Es descriptivo y solo usa val.

Se reentrenó con los hiperparámetros finales exactos (`scale_pos_weight` de `y_train`, early stopping en val, `random_state=42`) quitando features que reflejan la pila TCP o el sistema operativo de los hosts más que el comportamiento del tráfico. Las métricas son en val, a umbral 0.87.

| Variante | recall | FN | PR-AUC |
|---|---|---|---|
| modelo congelado (55 features) | 0.9984 | 3 | 0.9987 |
| sin `min_seg_size_forward` | 0.9687 | 59 | 0.9987 (sin cambio) |
| sin las 3 features claramente de la pila (`min_seg_size_forward`, `Init_Win_bytes_forward`, `Init_Win_bytes_backward`) | 0.9204 | 150 | 0.9949 |
| sin las 7 features marcadas (las 3 anteriores más `Fwd Header Length`, `Bwd Header Length`, `PSH Flag Count`, `ECE Flag Count`) | 0.9209 | 149 | 0.9937 |

- **Dónde están las pérdidas:** casi todas son Slowhttptest, cuyo recall baja de 0.9973 a entre 0.8639 y 0.9461 según la variante. Slowloris queda prácticamente sin cambios.
- **`min_seg_size_forward` no tiene un valor exclusivo de los ataques.** En train, los benignos toman 20 (52.9 %), 32 (44.1 %) y 40 (2.5 %); los ataques, 32 (60.9 %) y 40 (39.0 %).
- **Lectura:** el modelo muestra una dependencia parcial de estas features. Sin ellas, el recall en val baja, pero sigue por encima del criterio de 0.75.

Advertencias:

- Val ya se usó para el early stopping (también el de los modelos reducidos), para elegir los hiperparámetros y para elegir el umbral 0.87. Ese umbral se eligió para el modelo de 55 features y se aplicó sin cambios a las variantes reducidas.
- Es una sola semilla.
- Val tiene grupos de flujos casi duplicados (sección 2), así que las diferencias en conteos de FN no son observaciones independientes.
- Este experimento no puede distinguir una huella de la pila TCP de un comportamiento de la herramienta de ataque: una misma feature puede reflejar las dos cosas.

## Limitaciones

- **Las mismas sesiones de ataque están en train, val y test.** El split corta cada ventana de ataque 70/15/15 en orden cronológico, así que las tres particiones tienen flujos consecutivos de las mismas sesiones (misma herramienta y mismo origen). Test mide el desempeño en los minutos finales de esas sesiones, no en sesiones nuevas.
- **Una sola captura y un solo atacante.** Todo sale del miércoles 5/7/2017 de CIC-IDS2017. Los ataques llegan desde una sola IP de origen visible (la NAT 172.16.0.1) hacia una sola víctima (192.168.10.50). No se evaluó otra red, otra víctima ni otra herramienta.
- **Se excluyeron los flujos de respuesta del servidor.** Los 6,789 flujos víctima → NAT se excluyeron, no se reetiquetaron. El modelo no se entrenó ni se evaluó con ellos, y no se sabe cómo los clasificaría; en un despliegue real aparecerían.
- **Los IC por bootstrap son más estrechos que la incertidumbre real.** El bootstrap trata cada flujo como independiente, pero hay grupos de flujos repetidos o casi repetidos (sección 2; los FN incluyen pares que difieren en solo 9 features). Además, los IC solo remuestrean las predicciones de un único modelo: no incluyen la variabilidad del entrenamiento, la de la semilla ni la de la partición.
- **v1 y v2 no son directamente comparables.** Cambian los datos, la composición de val y test, el tratamiento del desbalance, los hiperparámetros, el objetivo del umbral y el umbral (sección 10).
- **La evaluación de v1 se vio antes de v2.** Los resultados de v1 en test ya se conocían cuando se hizo v2, y el test de v2 sale de la misma captura y de los mismos bloques que el de v1. v2 siguió las mismas reglas pre-registradas, con un solo cambio deliberado (F1 → F2, para alinearse con SVM y GRU), pero su test no es una primera mirada a esos datos.
- **Decisiones con márgenes chicos y una sola semilla.** La elección de C en 02b y la de 0.87 en 04 dependen de diferencias pequeñas en val (secciones 5 y 7), y el umbral queda en el extremo superior de la meseta de val y cerca del borde inferior de la distribución de probabilidades de los TP en test.
