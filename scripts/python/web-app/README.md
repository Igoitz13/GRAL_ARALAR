# PROSAIL S2 Biophysical Variable Retrieval v2.1

Aplicación web para la **inversión de variables biofísicas** a partir de imágenes
Sentinel-2, usando el modelo radiativo **PROSPECT-D + 4SAIL (PROSAIL)** en Python.

## Arquitectura

```
prosail_app/
├── main.py              ← Backend FastAPI (API REST + WebSocket)
├── prosail_pure.py      ← Modelo PROSAIL Python puro (PROSPECT-D + 4SAIL)
├── requirements.txt     ← Dependencias Python
├── static/
│   └── index.html       ← Frontend web (UI completa)
└── README.md
```

**Backend Python** genera las LUTs con el modelo PROSAIL real (no aproximación JS),
realiza la inversión por mínima distancia (cKDTree) y sirve resultados como
float32 binario (visualización en canvas) y GeoTIFF (si rasterio disponible).

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) Para exportar GeoTIFF georeferenciado:
pip install rasterio
```

## Ejecución

```bash
python main.py
```

Abrir **http://localhost:8000** en el navegador.

## Flujo de trabajo

1. **Generar LUT** — Configura parámetros PROSAIL y geometría solar.
   El servidor ejecuta `generate_lut()` de `prosail_pure.py` y almacena
   la LUT + cKDTree en memoria.

2. **Cargar imagen** — Soporta dos formatos:
   - **BEAM-DIMAP (SNAP)**: Paso ① cargar `.dim`, paso ② seleccionar carpeta `.data`
   - **GeoTIFF**: Stack de 10 bandas S2

3. **Ejecutar inversión** — Los archivos se suben al servidor. La inversión
   usa K-best matches ponderados por distancia inversa sobre la LUT.

4. **Resultados** — Se muestran mapas coloreados en el navegador.
   Se pueden descargar como GeoTIFF (si rasterio está instalado) o binario float32.

## Variables estimadas

| Variable | Descripción | Unidades |
|----------|-------------|----------|
| LAI | Leaf Area Index | m²/m² |
| Cab | Clorofila a+b | µg/cm² |
| CCC | Canopy Chlorophyll Content (LAI×Cab/100) | g/m² |
| Cw | Equivalent Water Thickness | cm |
| σ LAI | Incertidumbre LAI (std K-best) | m²/m² |

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Estado del servidor |
| POST | `/api/lut/generate` | Genera LUT PROSAIL |
| POST | `/api/invert/image` | Sube imagen y ejecuta inversión |
| GET | `/api/job/{id}` | Estado del job |
| WS | `/ws/{id}` | Progreso en tiempo real |
| GET | `/api/download/{id}/{var}` | Descarga GeoTIFF/binario |
| GET | `/api/result/{id}/{var}/raw` | Float32 binario para canvas |
| GET | `/api/result/{id}/meta` | Metadatos del resultado |

## Bandas Sentinel-2 esperadas

B2 (490nm), B3 (560nm), B4 (665nm), B5 (705nm), B6 (740nm),
B7 (783nm), B8 (842nm), B8A (865nm), B11 (1610nm), B12 (2190nm)

## Notas

- El modelo PROSAIL en `prosail_pure.py` es una implementación vectorizada
  de PROSPECT-D + 4SAIL con coeficientes de absorción calibrados para S2.
- Sin `rasterio`, los resultados se exportan como binario float32 raw.
- Para imágenes grandes (>5000×5000 px), considerar aumentar la RAM disponible.
- La LUT se almacena en `./temp_prosail/` y se puede reutilizar entre sesiones.
