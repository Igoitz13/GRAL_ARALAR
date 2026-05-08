# TFG Aralar — Diagnóstico pastoral por teledetección

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Status](https://img.shields.io/badge/status-en%20curso-yellow.svg)]()

Pipeline integrado para el diagnóstico pastoral del Parque Natural de Aralar (ZEC ES2120011) mediante teledetección Sentinel-2, modelización climática con GAM y triangulación cuádruple del Índice de Presión Pastoral (IPP).

**Trabajo de Fin de Grado**, Grado en Geografía y Ordenación del Territorio, UPV/EHU. Desarrollado en el marco del proyecto LIFE Oreka Mendian (LIFE15 NAT/ES/000762).

> **Web del proyecto**: https://moxkix.github.io/ARALAR_TFG/
>
> **Documentación**: la carpeta `docs/` se publica en la web automáticamente. Cualquier archivo que se suba ahí (Word, PDF, etc.) aparece como tarjeta en la sección Documentos sin tocar el HTML.

---

## Resumen del trabajo

El TFG diagnostica la presión pastoral en los pastos subalpinos de Aralar usando series de LAI de Sentinel-2 invertido con PROSAIL-NN, modelizadas climáticamente con GAM y validadas por **cuatro estrategias independientes** entre 2018 y 2025. La aproximación más original es la **Estrategia D** (referencia hayedo), que usa la cubierta forestal climácica adyacente como termómetro climático para aislar la señal pastoral de la climática.

### Resultados principales

- **Patrón cuasi-bianual**: crisis pastoral en 2020, 2022 y 2025 alternada con relajación en 2023 (concordancia 4/4 entre las cuatro estrategias).
- **Gradiente espacial coherente** con la distancia a las bordas: pasto cercano > intermedio > remoto.
- **Modelo climático G6** validado mediante LOYO (leave-one-year-out) sobre la serie 2018-2025.
- **Diagnóstico operativo** para el seguimiento adaptativo del LIFE Oreka Mendian.

---

## Estructura del repositorio

```
ARALAR_TFG/
├── index.html               Página de GitHub Pages (raíz del sitio)
├── style.css                Estilos de la web
├── README.md                Este archivo
├── LICENSE                  CC BY-NC-ND 4.0
│
├── docs/                    Documentación del TFG (Word, PDF…)
│                            Listado automático en la web vía API de GitHub.
│
├── data/                    Salidas tabulares del pipeline
│   ├── diagnostico_pastoral.csv
│   ├── validacion_loyo.csv
│   ├── comparacion_modelos.csv
│   └── hayedo_serie_referencia.csv
│
├── figures/                 Figuras de síntesis del trabajo
│   ├── IPP_triangulacion_ABCD.png
│   ├── IPP_barras_interanual.png
│   ├── IPP_por_habitat.png
│   ├── evolucion_IPP_interanual.png
│   ├── gradiente_IPP_distancia_bordas.png
│   ├── diagnostico_integrado_global.png
│   ├── diag_A_pasto_{cercano,intermedio,remoto}.png
│   ├── diag_A_hayedo_{cercano,intermedio,remoto}.png
│   └── diag_C_anomalias.png
│
├── maps/                    Cartografía pastoral por año
│   ├── 2018/  …  2025/      Serie principal (16 mapas + resumen .txt por año)
│   └── 2018_LIFE/  …  2025_LIFE/
│                            Variante etiquetada para LIFE Oreka Mendian
│
└── scripts/
    ├── python/              Inversión PROSAIL local + entrenamiento NN + cartografía
    │   ├── main.py
    │   ├── prosail_pure.py
    │   ├── train_gee_coefficients.py
    │   └── cartografia_pastoral_v5.py
    ├── javascripts/         Pipeline en Google Earth Engine
    │   └── GEE_Pipeline_Integrado_Aralar.js
    ├── R/                   Modelización climática y diagnóstico cuádruple
    │   └── diagnostico_pastoral_aralar_v9_4.R
    └── batch/               Lanzadores Windows
        └── ejecutar_cartografia5.bat
```

### Productos cartográficos por año

Cada carpeta `maps/<año>/` contiene los siguientes mapas en PNG (nomenclatura `aralar_mapa_*_<año>.png` / `aralar_pastos_mapa_*_<año>.png`):

| Familia | Mapas |
|---|---|
| Triangulación IPP | `IPP_A_pastos`, `IPP_A_todos`, `IPP_B_pastos`, `IPP_C_pastos`, `IPP_D_pastos`, `IPP_D_todos` |
| Síntesis | `IPP_consenso_pastos`, `IPP_concordancia_pastos`, `IPP_espacializado` |
| Gestión | `mapa_biomasa`, `mapa_calidad`, `mapa_carga`, `mapa_balance`, `mapa_riesgo` |
| Control | `mapa_pendiente` |

Además se genera un `aralar_resumen_<año>.txt` con los metadatos del año.

---

## Cómo reproducir

### Requisitos

- **Python 3.10+** con: `prosail`, `scikit-learn`, `numpy`, `rasterio`, `matplotlib`, `pandas`
- **R 4.3+** con: `mgcv`, `SPEI`, `ggplot2`, `dplyr`, `tidyr`, `lubridate`
- **Cuenta Google Earth Engine** activa

### Pasos del pipeline

**1. LUTs y entrenamiento de redes neuronales (Python local)**

```bash
cd scripts/python
python main.py                      # genera la LUT y entrena la NN
python train_gee_coefficients.py    # exporta los coeficientes para inyectar en GEE
```

**2. Inversión a escala en GEE (JavaScript)**

Subir los assets de Aralar al proyecto GEE (límite, HIC, bordas, MDT5).
Pegar `scripts/javascripts/GEE_Pipeline_Integrado_Aralar.js` en el editor de código y ejecutar. Las exportaciones a Drive incluyen los CSV de LAI zonal, los CSV climáticos y los GeoTIFF anuales.

> ⚠️ **Bug crítico documentado**: en la reducción zonal, las llamadas a `ee.Dictionary.combine()` deben llevar el segundo argumento `false` (overwrite=false). Sin esa precaución, los defaults sobrescriben los valores reales y el CSV exportado sale entero a -9999.

**3. Modelización climática y cuatro estrategias del IPP (R)**

```r
source("scripts/R/diagnostico_pastoral_aralar_v9_4.R")
```

Salida: `data/diagnostico_pastoral.csv`, `data/validacion_loyo.csv`, `data/comparacion_modelos.csv` y las figuras de diagnóstico en `figures/`.

**4. Cartografía pastoral (Python)**

```bash
cd scripts/python
python cartografia_pastoral_v5.py --year 2022
```

O ejecutar `scripts/batch/ejecutar_cartografia5.bat` para procesar la serie 2018-2025 completa. Salida en `maps/<año>/`.

---

## Web del proyecto

La web (`index.html` + `style.css`) se publica con GitHub Pages desde la rama `main` y la raíz del repositorio. El listado de la sección Documentos se construye al cargar la página consultando la API de GitHub (`/repos/Moxkix/ARALAR_TFG/contents/docs`). Esto significa que **basta con subir un archivo a `docs/` para que aparezca como tarjeta**, sin tocar el HTML.

---

## Citación

Si usas este código o los resultados en trabajos derivados, por favor cita:

```
[Apellido, Nombre del alumno] (2026). Diagnóstico pastoral del Parque Natural
de Aralar mediante teledetección y modelización climática. Trabajo de Fin de
Grado, Grado en Geografía y Ordenación del Territorio, Universidad del País
Vasco / Euskal Herriko Unibertsitatea.
https://github.com/Moxkix/ARALAR_TFG
```

---

## Licencia

Este trabajo se publica bajo licencia [Creative Commons Atribución-NoComercial-SinDerivadas 4.0 Internacional (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es).

Puedes:
- Compartir el material en cualquier medio o formato
- Citarlo en trabajos académicos siempre con atribución correcta

No puedes:
- Usarlo con fines comerciales sin autorización expresa del autor
- Distribuir versiones modificadas o derivadas

Para usos no contemplados en la licencia, contactar con el autor.

---

## Agradecimientos

- **LIFE Oreka Mendian** (LIFE15 NAT/ES/000762), proyecto coordinado por HAZI Fundazioa, por proporcionar el contexto operativo de este trabajo.
- **Dirección del TFG**, por el seguimiento iterativo y la propuesta del enfoque de triangulación cuádruple del IPP.
- **Servicio de Conservación de la Naturaleza, Diputación Foral de Guipúzcoa**: cartografía HIC del Parque Natural de Aralar.
- **Gobierno de Navarra**: serie meteorológica de la estación de San Miguel de Aralar 1991-2025.
- **GeoEuskadi**: MDT5 del macizo de Aralar.
- **ECMWF / Copernicus / ESA**: datos Sentinel-2 SR Harmonized y ERA5-Land.
- **Google Earth Engine**: plataforma de cómputo en la nube.
