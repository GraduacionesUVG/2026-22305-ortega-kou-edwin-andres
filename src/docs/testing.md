# Tests

Referencia de la suite de tests del proyecto: qué cubre, cómo correrla, y la convención
a seguir cuando alguien agregue más tests (por ejemplo, cuando GRU/SVM/XGBoost empiecen a
tener código propio en `src/models/`).

## Cómo correrla

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m pytest -v
```

Comandos útiles:

```bash
# Solo los tests rápidos (sin el de integración con data/raw)
python -m pytest tests/test_clean.py -v

# Solo el test de integración (requiere data/raw local; si no lo tienes, se salta solo)
python -m pytest tests/test_integration.py -v

# Un test específico
python -m pytest tests/test_clean.py::test_scale_features_fits_only_on_train -v
```

## Estructura

```
low-and-slow-ids/
├── conftest.py                # raíz — agrega el repo a sys.path para que
│                               #   `from src.preprocessing...` funcione desde pytest
│                               #   (el repo no está empaquetado/instalado)
└── tests/
    ├── test_clean.py          # tests unitarios, con datos sintéticos generados en la
    │                           #   fixture — no requieren data/raw
    └── test_integration.py    # corre el pipeline real contra data/raw; se salta solo
                                #   (pytest.mark.skipif) si no hay CSVs locales
```

No se espeja la estructura de `src/` (`tests/preprocessing/test_clean.py`, etc.) — con un
solo módulo bajo test no aporta nada. Si más adelante hay tests para varios módulos,
vale la pena revisar si conviene organizar por subcarpeta.

## Qué cubre ahora mismo

`tests/test_clean.py` prueba `train_test_split_by_flow()` y `scale_features()`
(`src/preprocessing/clean.py`, agregadas en la rama `fix/shared-temporal-split` — ver
`docs/adr/0001-split-cronologico-por-bloques-temporales.md` para el porqué de esas
funciones). Usa una fixture sintética de 110 filas (5 bloques con fronteras conocidas,
`outlier_quantile=0.0` para que sean exactas) en vez del dataset real, así que corre en
segundos en cualquier entorno, sin necesitar `data/raw`.

Casos cubiertos:

- **`train_test_split_by_flow()`**: asignación correcta de cada fila a su bloque
  (incluye las fronteras inclusive/exclusive de cada uno); que `split` nunca llegue a
  `X` y que `Timestamp`/`bloque` sí queden como metadata; tamaños del corte 70/15/15 por
  bloque y que no haya fuga de orden cronológico entre train/val/test; forma de la
  salida (tupla de 6, tamaños consistentes); que `random_state` no tenga ningún efecto
  sobre el resultado; los dos `ValueError` esperados (falta `Timestamp`, no hay filas de
  ataque); que el sort estable interno preserve el orden original en timestamps
  empatados.
- **`scale_features()`**: que el `StandardScaler` se ajuste únicamente con `X_train`
  (sin fuga de información de val/test hacia los parámetros de escalado — el caso más
  importante); que excluya `Timestamp`/`bloque` por default; que respete una lista
  explícita de `columns_to_scale`; que el scaler devuelto sea reusable vía
  `.transform()` en datos nuevos.

`tests/test_integration.py` corre `load_and_clean → train_test_split_by_flow →
scale_features` contra el dataset real y verifica los criterios de aceptación del ADR
0001 (ningún bloque vacío, ningún split sin ejemplos de ataque) — automatiza lo que antes
había que revisar a mano en la salida impresa del script de verificación.

No hay tests para `load_raw_csvs()`, `basic_cleaning()`, `filter_low_and_slow()`,
`encode_labels()` ni `load_and_clean()` todavía — son código anterior a esta suite, fuera
de lo que se cubrió al crearla.

## Convención para agregar tests nuevos

- **Un archivo de test por módulo de `src/`** (`tests/test_<modulo>.py`), plano, sin
  espejar subcarpetas a menos que la cantidad de módulos lo justifique.
- **Preferir datos sintéticos sobre el dataset real.** `data/raw/` y `data/processed/`
  están en `.gitignore` — no todos los entornos los tienen. Un test que dependa de ellos
  sin un `skipif` rompe la suite para cualquiera sin el dataset descargado.
- **Si un test sí necesita el dataset real** (por ejemplo, para dar confianza end-to-end
  además de la cobertura unitaria), va en un archivo aparte con
  `pytest.mark.skipif(...)` sobre la existencia de `data/raw`/`data/processed`, siguiendo
  el patrón de `tests/test_integration.py` — así nunca falla la suite por no tener el
  dataset, solo se salta.
- **Fixtures compartidas por un solo archivo de test van inline** en ese archivo con
  `@pytest.fixture`. Solo promoverlas a un `tests/conftest.py` cuando más de un archivo
  las necesite — no antes.
- **`conftest.py` de la raíz solo se encarga del `sys.path`.** No agregarle fixtures ni
  lógica de test ahí; ese archivo debe seguir siendo mínimo y de un solo propósito.
