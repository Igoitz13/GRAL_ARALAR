# TFG Aralar â€” DiagnÃ³stico pastoral por teledetecciÃ³n

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Status](https://img.shields.io/badge/status-en%20curso-yellow.svg)]()

Pipeline integrado para el diagnÃ³stico pastoral del Parque Natural de Aralar (ZEC ES2120011) mediante teledetecciÃ³n Sentinel-2, modelizaciÃ³n climÃ¡tica con GAM y triangulaciÃ³n cuÃ¡druple del Ãndice de PresiÃ³n Pastoral (IPP).

**Trabajo de Fin de Grado**, Grado en GeografÃ­a y OrdenaciÃ³n del Territorio, UPV/EHU. Desarrollado en el marco del proyecto LIFE Oreka Mendian (LIFE15 NAT/ES/000762).

> **Web del proyecto**: https://igoitz13.github.io/GRAL_ARALAR/
>
> **DocumentaciÃ³n**: la carpeta `docs/` se publica en la web automÃ¡ticamente. Cualquier archivo que se suba ahÃ­ (Word, PDF, etc.) aparece como tarjeta en la secciÃ³n Documentos sin tocar el HTML.

---

## Resumen del trabajo

El TFG diagnostica la presiÃ³n pastoral en los pastos subalpinos de Aralar usando series de LAI de Sentinel-2 invertido con PROSAIL-NN, modelizadas climÃ¡ticamente con GAM y validadas por **cuatro estrategias independientes** entre 2018 y 2025. La aproximaciÃ³n mÃ¡s original es la **Estrategia D** (referencia hayedo), que usa la cubierta forestal climÃ¡cica adyacente como termÃ³metro climÃ¡tico para aislar la seÃ±al pastoral de la climÃ¡tica.

### Resultados principales

- **PatrÃ³n cuasi-bianual**: crisis pastoral en 2020, 2022 y 2025 alternada con relajaciÃ³n en 2023 (concordancia 4/4 entre las cuatro estrategias).
- **Gradiente espacial coherente** con la distancia a las bordas: pasto cercano > intermedio > remoto.
- **Modelo climÃ¡tico G6** validado mediante LOYO (leave-one-year-out) sobre la serie 2018-2025.
- **DiagnÃ³stico operativo** para el seguimiento adaptativo del LIFE Oreka Mendian.

---

## Estructura del repositorio

```
ARALAR_TFG/
â”œâ”€â”€ index.html               PÃ¡gina de GitHub Pages (raÃ­z del sitio)
â”œâ”€â”€ style.css                Estilos de la web
â”œâ”€â”€ README.md                Este archivo
â”œâ”€â”€ LICENSE                  CC BY-NC-ND 4.0
â”‚
â”œâ”€â”€ docs/                    DocumentaciÃ³n del TFG (Word, PDFâ€¦)
â”‚                            Listado automÃ¡tico en la web vÃ­a API de GitHub.
â”‚
â”œâ”€â”€ data/                    Salidas tabulares del pipeline
â”‚   â”œâ”€â”€ diagnostico_pastoral.csv
â”‚   â”œâ”€â”€ validacion_loyo.csv
â”‚   â”œâ”€â”€ comparacion_modelos.csv
â”‚   â”œâ”€â”€ hayedo_serie_referencia.csv
â”‚   â””â”€â”€ coefficients/        LUTs PROSAIL exportadas y coeficientes NN
â”‚                            ({PASTOS,HAYEDO}_gee.csv y *_nn_coefficients.{js,json})
â”‚                            Generados por la web-app y consumidos por el script GEE.
â”‚
â”œâ”€â”€ figures/                 Figuras de sÃ­ntesis del trabajo
â”‚   â”œâ”€â”€ IPP_triangulacion_ABCD.png
â”‚   â”œâ”€â”€ IPP_barras_interanual.png
â”‚   â”œâ”€â”€ IPP_por_habitat.png
â”‚   â”œâ”€â”€ evolucion_IPP_interanual.png
â”‚   â”œâ”€â”€ gradiente_IPP_distancia_bordas.png
â”‚   â”œâ”€â”€ diagnostico_integrado_global.png
â”‚   â”œâ”€â”€ diag_A_pasto_{cercano,intermedio,remoto}.png
â”‚   â”œâ”€â”€ diag_A_hayedo_{cercano,intermedio,remoto}.png
â”‚   â””â”€â”€ diag_C_anomalias.png
â”‚
â”œâ”€â”€ maps/                    CartografÃ­a pastoral por aÃ±o
â”‚   â”œâ”€â”€ 2018/  â€¦  2025/      Serie principal (16 mapas + resumen .txt por aÃ±o)
â”‚   â””â”€â”€ 2018_LIFE/  â€¦  2025_LIFE/
â”‚                            Variante etiquetada para LIFE Oreka Mendian
â”‚
â””â”€â”€ scripts/
    â”œâ”€â”€ python/
    â”‚   â”œâ”€â”€ web-app/         AplicaciÃ³n web FastAPI para generar LUTs PROSAIL
    â”‚   â”‚                    y hacer la inversiÃ³n Sentinel-2 â†’ variables biofÃ­sicas
    â”‚   â”‚   â”œâ”€â”€ main.py             Backend FastAPI (REST + WebSocket)
    â”‚   â”‚   â”œâ”€â”€ prosail_pure.py     Modelo PROSPECT-D + 4SAIL puro Python
    â”‚   â”‚   â”œâ”€â”€ requirements.txt    Dependencias
    â”‚   â”‚   â”œâ”€â”€ README.md           GuÃ­a de instalaciÃ³n y uso
    â”‚   â”‚   â””â”€â”€ static/index.html   Frontend SPA
    â”‚   â”œâ”€â”€ train_gee_coefficients.py   Entrenamiento de la NN para inyectar en GEE
    â”‚   â””â”€â”€ cartografia_pastoral_v5.py  GeneraciÃ³n de los mapas anuales
    â”œâ”€â”€ javascripts/         Pipeline en Google Earth Engine
    â”‚   â””â”€â”€ GEE_Pipeline_Integrado_Aralar.js
    â”œâ”€â”€ R/                   ModelizaciÃ³n climÃ¡tica y diagnÃ³stico cuÃ¡druple
    â”‚   â””â”€â”€ diagnostico_pastoral_aralar_v9_4.R
    â””â”€â”€ batch/               Lanzadores Windows
        â””â”€â”€ ejecutar_cartografia5.bat
```

### Web-app PROSAIL (inversiÃ³n Sentinel-2)

La aplicaciÃ³n de `scripts/python/web-app/` es un servidor FastAPI con frontend incluido. Permite configurar de forma interactiva los parÃ¡metros de la LUT (rangos PROSAIL, geometrÃ­a solar, distribuciÃ³n de muestreo, ruido), generarla en memoria con el modelo PROSPECT-D + 4SAIL real y ejecutar la inversiÃ³n sobre imÃ¡genes Sentinel-2 (BEAM-DIMAP de SNAP o GeoTIFF de 10 bandas) por mÃ­nima distancia con cKDTree.

```bash
cd scripts/python/web-app
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
# Abrir http://localhost:8000
```

Los productos finales (LUTs exportadas y coeficientes de la red neuronal listos para inyectar en GEE) se versionan en `data/coefficients/`. La cachÃ© de runtime (`temp_prosail/`) estÃ¡ excluida del repositorio.

### Productos cartogrÃ¡ficos por aÃ±o

Cada carpeta `maps/<aÃ±o>/` contiene los siguientes mapas en PNG (nomenclatura `aralar_mapa_*_<aÃ±o>.png` / `aralar_pastos_mapa_*_<aÃ±o>.png`):

| Familia | Mapas |
|---|---|
| TriangulaciÃ³n IPP | `IPP_A_pastos`, `IPP_A_todos`, `IPP_B_pastos`, `IPP_C_pastos`, `IPP_D_pastos`, `IPP_D_todos` |
| SÃ­ntesis | `IPP_consenso_pastos`, `IPP_concordancia_pastos`, `IPP_espacializado` |
| GestiÃ³n | `mapa_biomasa`, `mapa_calidad`, `mapa_carga`, `mapa_balance`, `mapa_riesgo` |
| Control | `mapa_pendiente` |

AdemÃ¡s se genera un `aralar_resumen_<aÃ±o>.txt` con los metadatos del aÃ±o.

---

## CÃ³mo reproducir

### Requisitos

- **Python 3.10+** con: `prosail`, `scikit-learn`, `numpy`, `rasterio`, `matplotlib`, `pandas`
- **R 4.3+** con: `mgcv`, `SPEI`, `ggplot2`, `dplyr`, `tidyr`, `lubridate`
- **Cuenta Google Earth Engine** activa

### Pasos del pipeline

**1. LUTs y entrenamiento de redes neuronales (Python local)**

```bash
# GeneraciÃ³n de la LUT con la web-app (interfaz web en localhost:8000)
cd scripts/python/web-app
pip install -r requirements.txt
python main.py

# Entrenamiento de la NN sobre las LUTs exportadas y exportaciÃ³n de coeficientes para GEE
cd ../
python train_gee_coefficients.py
```

Las LUTs exportadas y los coeficientes finales quedan en `data/coefficients/`.

**2. InversiÃ³n a escala en GEE (JavaScript)**

Subir los assets de Aralar al proyecto GEE (lÃ­mite, HIC, bordas, MDT5).
Pegar `scripts/javascripts/GEE_Pipeline_Integrado_Aralar.js` en el editor de cÃ³digo y ejecutar. Las exportaciones a Drive incluyen los CSV de LAI zonal, los CSV climÃ¡ticos y los GeoTIFF anuales.

> âš ï¸ **Bug crÃ­tico documentado**: en la reducciÃ³n zonal, las llamadas a `ee.Dictionary.combine()` deben llevar el segundo argumento `false` (overwrite=false). Sin esa precauciÃ³n, los defaults sobrescriben los valores reales y el CSV exportado sale entero a -9999.

**3. ModelizaciÃ³n climÃ¡tica y cuatro estrategias del IPP (R)**

```r
source("scripts/R/diagnostico_pastoral_aralar_v9_4.R")
```

Salida: `data/diagnostico_pastoral.csv`, `data/validacion_loyo.csv`, `data/comparacion_modelos.csv` y las figuras de diagnÃ³stico en `figures/`.

**4. CartografÃ­a pastoral (Python)**

```bash
cd scripts/python
python cartografia_pastoral_v5.py --year 2022
```

O ejecutar `scripts/batch/ejecutar_cartografia5.bat` para procesar la serie 2018-2025 completa. Salida en `maps/<aÃ±o>/`.

---

## Web del proyecto

La web (`index.html` + `style.css`) se publica con GitHub Pages desde la rama `main` y la raÃ­z del repositorio. El listado de la secciÃ³n Documentos se construye al cargar la pÃ¡gina consultando la API de GitHub (`/repos/Igoitz13/GRAL_ARALAR/contents/docs`). Esto significa que **basta con subir un archivo a `docs/` para que aparezca como tarjeta**, sin tocar el HTML.

---

## CitaciÃ³n

Si usas este cÃ³digo o los resultados en trabajos derivados, por favor cita:

```
[Apellido, Nombre del alumno] (2026). DiagnÃ³stico pastoral del Parque Natural
de Aralar mediante teledetecciÃ³n y modelizaciÃ³n climÃ¡tica. Trabajo de Fin de
Grado, Grado en GeografÃ­a y OrdenaciÃ³n del Territorio, Universidad del PaÃ­s
Vasco / Euskal Herriko Unibertsitatea.
https://github.com/Igoitz13/GRAL_ARALAR
```

---

## Licencia

Este trabajo se publica bajo licencia [Creative Commons AtribuciÃ³n-NoComercial-SinDerivadas 4.0 Internacional (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es).

Puedes:
- Compartir el material en cualquier medio o formato
- Citarlo en trabajos acadÃ©micos siempre con atribuciÃ³n correcta

No puedes:
- Usarlo con fines comerciales sin autorizaciÃ³n expresa del autor
- Distribuir versiones modificadas o derivadas

Para usos no contemplados en la licencia, contactar con el autor.

---

## Agradecimientos

- **LIFE Oreka Mendian** (LIFE15 NAT/ES/000762), proyecto coordinado por HAZI Fundazioa, por proporcionar el contexto operativo de este trabajo.
- **DirecciÃ³n del TFG**, por el seguimiento iterativo y la propuesta del enfoque de triangulaciÃ³n cuÃ¡druple del IPP.
- **Servicio de ConservaciÃ³n de la Naturaleza, DiputaciÃ³n Foral de GuipÃºzcoa**: cartografÃ­a HIC del Parque Natural de Aralar.
- **Gobierno de Navarra**: serie meteorolÃ³gica de la estaciÃ³n de San Miguel de Aralar 1991-2025.
- **GeoEuskadi**: MDT5 del macizo de Aralar.
- **ECMWF / Copernicus / ESA**: datos Sentinel-2 SR Harmonized y ERA5-Land.
- **Google Earth Engine**: plataforma de cÃ³mputo en la nube.
