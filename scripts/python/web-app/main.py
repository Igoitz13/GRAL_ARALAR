"""
PROSAIL Inversion API - FastAPI Backend v2.1
Soporta GeoTIFF y BEAM-DIMAP (SNAP)
Usa prosail_pure.py para generación de LUT con modelo PROSPECT-D + 4SAIL real.
"""
import os, shutil, pickle, uuid, asyncio, re, glob, threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy.spatial import cKDTree

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prosail_pure import generate_lut, S2_WAVELENGTHS, run, SNAP8_BAND_INDICES

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# =============================================================================
# CONFIG
# =============================================================================
TEMP_DIR = "./temp_prosail"
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="PROSAIL S2 Biophysical Retrieval API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

active_jobs: Dict[str, dict] = {}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class Geometry(BaseModel):
    # Each angle can be either:
    #   - float           → fixed geometry (legacy, single-scene)
    #   - [lo, hi] list   → LHS-sampled over the range (multi-scene marginalisation)
    tts: Union[float, List[float]] = 30.0
    tto: Union[float, List[float]] = 0.0
    psi: Union[float, List[float]] = 0.0

class ParamRanges(BaseModel):
    N: Tuple[float, float] = (1.0, 3.0)
    Cab: Tuple[float, float] = (1, 90)
    Car: Tuple[float, float] = (0, 25)
    Cbrown: Tuple[float, float] = (0.0, 2.0)
    Cw: Tuple[float, float] = (0.002, 0.05)
    Cm: Tuple[float, float] = (0.001, 0.025)
    LAI: Tuple[float, float] = (0.0, 15.0)
    hot: Tuple[float, float] = (0.01, 0.50)
    ALA: Tuple[float, float] = (20.0, 80.0)
    psoil: Tuple[float, float] = (0.0, 1.0)
    rsoil: Tuple[float, float] = (0.2, 3.5)

class LUTRequest(BaseModel):
    n_samples: int = 50000
    ranges: ParamRanges = ParamRanges()
    geometry: Geometry = Geometry()
    noise_sigma: float = 0.01
    distribution: str = 'baret'       # 'baret' or 'uniform'
    band_selection: str = 'snap8'     # 'snap8' or 'all10'

# =============================================================================
# BEAM-DIMAP READER
# =============================================================================
S2_BAND_NAMES = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
S2_BAND_ALIASES = {
    'B2': ['B2','b2','B02','b02'], 'B3': ['B3','b3','B03','b03'],
    'B4': ['B4','b4','B04','b04'], 'B5': ['B5','b5','B05','b05'],
    'B6': ['B6','b6','B06','b06'], 'B7': ['B7','b7','B07','b07'],
    'B8': ['B8','b8','B08','b08'], 'B8A':['B8A','b8a','B8a','B08A'],
    'B11':['B11','b11'], 'B12':['B12','b12']
}
ENVI_DTYPES = {1:'uint8', 2:'int16', 3:'int32', 4:'float32', 5:'float64', 12:'uint16', 13:'uint32'}
NP_DTYPES = {
    'uint8': np.uint8, 'int16': np.int16, 'uint16': np.uint16,
    'int32': np.int32, 'uint32': np.uint32, 'float32': np.float32, 'float64': np.float64,
}

def parse_envi_hdr(hdr_path: str) -> dict:
    meta = {'samples': 0, 'lines': 0, 'bands': 1, 'data_type': 'float32', 'byte_order': 0}
    with open(hdr_path, 'r', errors='ignore') as f:
        for line in f:
            m = re.match(r'^\s*([\w\s]+?)\s*=\s*(.+)\s*$', line)
            if not m: continue
            key, val = m.group(1).strip().lower(), m.group(2).strip()
            if key == 'samples': meta['samples'] = int(val)
            elif key == 'lines': meta['lines'] = int(val)
            elif key == 'bands': meta['bands'] = int(val)
            elif key == 'data type': meta['data_type'] = ENVI_DTYPES.get(int(val), 'float32')
            elif key == 'byte order': meta['byte_order'] = int(val)
    return meta

def parse_dim_xml(dim_path: str) -> dict:
    """Parse BEAM-DIMAP .dim XML — extrae metadatos, CRS y geotransform"""
    from xml.etree import ElementTree as ET
    tree = ET.parse(dim_path)
    root = tree.getroot()
    meta = {'width': 0, 'height': 0, 'bands': [], 'nodata': -9999,
            'crs_wkt': None, 'transform': None}
    ncols = root.find('.//NCOLS')
    nrows = root.find('.//NROWS')
    if ncols is not None: meta['width'] = int(ncols.text)
    if nrows is not None: meta['height'] = int(nrows.text)
    for band_el in root.findall('.//Spectral_Band_Info'):
        name = band_el.findtext('BAND_NAME', '')
        meta['bands'].append(name)
    nodata_el = root.find('.//NO_DATA_VALUE')
    if nodata_el is not None:
        try: meta['nodata'] = float(nodata_el.text)
        except: pass

    # --- Extraer CRS ---
    wkt_el = root.find('.//WKT')
    if wkt_el is not None and wkt_el.text:
        meta['crs_wkt'] = wkt_el.text.strip()
    else:
        # SNAP a veces usa HORIZONTAL_CS_CODE (EPSG)
        epsg_el = root.find('.//HORIZONTAL_CS_CODE')
        if epsg_el is not None and epsg_el.text:
            code = epsg_el.text.strip().replace('EPSG:', '')
            meta['crs_epsg'] = int(code)

    # --- Extraer geotransform (Geoposition/MAP_INFO or EASTING/NORTHING) ---
    # SNAP DIMAP: <Geoposition> → <IMAGE_TO_MODEL_TRANSFORM> (6 params)
    itm = root.find('.//IMAGE_TO_MODEL_TRANSFORM')
    if itm is not None and itm.text:
        vals = [float(v) for v in itm.text.strip().split(',')]
        if len(vals) >= 6:
            # SNAP format: [scaleX, shearX, shearY, scaleY, transX, transY]
            # rasterio Affine: (scaleX, shearX, transX, shearY, scaleY, transY)
            meta['transform'] = (vals[0], vals[1], vals[4], vals[2], vals[3], vals[5])

    # Alternativa: EASTING / NORTHING + pixel size
    if meta['transform'] is None:
        easting_el = root.find('.//EASTING')
        northing_el = root.find('.//NORTHING')
        psx_el = root.find('.//PIXEL_SIZE_X')
        psy_el = root.find('.//PIXEL_SIZE_Y')
        if all(el is not None for el in [easting_el, northing_el, psx_el, psy_el]):
            try:
                east = float(easting_el.text)
                north = float(northing_el.text)
                psx = float(psx_el.text)
                psy = float(psy_el.text)
                meta['transform'] = (psx, 0.0, east, 0.0, -abs(psy), north)
            except: pass

    return meta

def read_dimap_bands(work_dir: str, log_fn=None) -> Tuple[np.ndarray, int, int, list, dict]:
    """
    Lee bandas S2 de un producto BEAM-DIMAP.
    Soporta resoluciones mixtas (10m/20m) con remuestreo nearest-neighbor.
    Returns: (stack [10, H, W], width, height, band_names, rasterio_profile_or_None)
    """
    def _log(msg):
        if log_fn: log_fn(msg)
        print(f"  [DIMAP] {msg}")

    # Find .dim file
    dim_files = glob.glob(os.path.join(work_dir, '**/*.dim'), recursive=True)
    dim_meta = parse_dim_xml(dim_files[0]) if dim_files else {'width': 0, 'height': 0, 'nodata': -9999}
    
    # Find all .img files recursively
    img_files = {}
    for f in glob.glob(os.path.join(work_dir, '**/*.img'), recursive=True):
        basename = os.path.splitext(os.path.basename(f))[0]
        img_files[basename] = f

    _log(f"Archivos .img encontrados: {len(img_files)}")

    # Determine target resolution (largest band dimensions = 10m grid)
    target_w, target_h = dim_meta['width'], dim_meta['height']
    
    # Match S2 bands
    matched = []
    for band_name in S2_BAND_NAMES:
        aliases = S2_BAND_ALIASES[band_name]
        found_path = None
        for alias in aliases:
            if alias in img_files:
                found_path = img_files[alias]
                break
            # Case-insensitive
            for k, v in img_files.items():
                if k.lower() == alias.lower():
                    found_path = v
                    break
            if found_path: break
        
        if found_path:
            matched.append((band_name, found_path))
            _log(f"✓ {band_name} → {os.path.basename(found_path)}")
        else:
            _log(f"✗ {band_name} no encontrada")

    if len(matched) < 4:
        raise FileNotFoundError(f"Solo {len(matched)} bandas S2 encontradas, se necesitan al menos 4")

    # Read each band with .hdr metadata
    bands_data = []
    for band_name, img_path in matched:
        hdr_path = img_path.replace('.img', '.hdr')
        if os.path.exists(hdr_path):
            hdr = parse_envi_hdr(hdr_path)
            dtype_str = hdr['data_type']
            band_w, band_h = hdr['samples'], hdr['lines']
        else:
            dtype_str = 'float32'
            band_w, band_h = target_w, target_h

        np_dtype = NP_DTYPES.get(dtype_str, np.float32)
        raw = np.fromfile(img_path, dtype=np_dtype)

        # Auto-detect type if pixel count doesn't match
        if raw.size != band_w * band_h:
            file_size = os.path.getsize(img_path)
            for try_dtype in [np.uint16, np.float32, np.int16]:
                n = file_size // np.dtype(try_dtype).itemsize
                if n == band_w * band_h:
                    raw = np.fromfile(img_path, dtype=try_dtype)
                    break
                # Try with target dims
                if n == target_w * target_h:
                    raw = np.fromfile(img_path, dtype=try_dtype)
                    band_w, band_h = target_w, target_h
                    break

        band_2d = raw[:band_w*band_h].reshape(band_h, band_w).astype(np.float32)

        # Update target dimensions to the largest grid found
        if band_w * band_h > target_w * target_h:
            target_w, target_h = band_w, band_h

        bands_data.append((band_name, band_2d, band_w, band_h))

    # Resample all bands to target grid and place at FIXED S2 band positions
    # Stack is ALWAYS [10, H, W] with B2=idx0, B3=idx1, ..., B12=idx9
    # Missing bands remain as NaN
    stack = np.full((10, target_h, target_w), np.nan, dtype=np.float32)
    for bname, data, bw, bh in bands_data:
        # Map band name to fixed S2 position
        try:
            s2_idx = S2_BAND_NAMES.index(bname)
        except ValueError:
            _log(f"  ⚠ Banda {bname} no reconocida, ignorada")
            continue

        if bw == target_w and bh == target_h:
            stack[s2_idx] = data
        else:
            _log(f"  Remuestreando {bname}: {bw}×{bh} → {target_w}×{target_h}")
            from scipy.ndimage import zoom
            stack[s2_idx] = zoom(data, (target_h / bh, target_w / bw), order=0)

    # Report which band positions are filled
    filled = [S2_BAND_NAMES[i] for i in range(10) if not np.all(np.isnan(stack[i]))]
    _log(f"Bandas en stack fijo: {', '.join(filled)} ({len(filled)}/10)")

    # Auto-scale to reflectance [0,1]
    valid = stack[0].ravel()
    valid = valid[(valid > 0) & (valid < 1e10) & np.isfinite(valid)]
    if len(valid) > 0:
        p99 = np.percentile(valid, 99)
        if p99 > 100:
            _log(f"Auto-escala: DN→reflectancia (÷10000), p99={p99:.0f}")
            stack /= 10000.0
        elif p99 > 1.5:
            _log(f"Auto-escala: ÷100, p99={p99:.2f}")
            stack /= 100.0

    # Mask nodata
    stack[(stack < 0) | (stack > 1.5) | ~np.isfinite(stack)] = np.nan

    # Build rasterio profile from .dim georeferencing
    profile = None
    if HAS_RASTERIO:
        from rasterio.transform import Affine
        from rasterio.crs import CRS

        profile = {
            'driver': 'GTiff',
            'height': target_h,
            'width': target_w,
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw',
            'nodata': -9999
        }

        # CRS
        if dim_meta.get('crs_wkt'):
            try:
                profile['crs'] = CRS.from_wkt(dim_meta['crs_wkt'])
                _log(f"CRS desde WKT: {profile['crs']}")
            except Exception as e:
                _log(f"Warning: CRS WKT inválido: {e}")
        elif dim_meta.get('crs_epsg'):
            try:
                profile['crs'] = CRS.from_epsg(dim_meta['crs_epsg'])
                _log(f"CRS EPSG:{dim_meta['crs_epsg']}")
            except Exception as e:
                _log(f"Warning: EPSG inválido: {e}")

        # Geotransform
        if dim_meta.get('transform'):
            t = dim_meta['transform']
            profile['transform'] = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
            _log(f"GeoTransform: pixel={t[0]:.1f}m, origin=({t[2]:.1f}, {t[5]:.1f})")
        else:
            _log("⚠ Sin geotransform en .dim — GeoTIFF sin coordenadas geográficas")

    band_names = [m[0] for m in matched]
    return stack, target_w, target_h, band_names, profile


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "prosail_pure v3 (PROSPECT-D + 4SAIL + Baret)", "version": "3.0",
            "rasterio": HAS_RASTERIO, "bands": S2_BAND_NAMES}


@app.post("/api/parse-metadata")
async def parse_metadata(file: UploadFile = File(...)):
    """
    Parsear fichero de metadatos Sentinel-2 (MTD_TL.xml o MTD_MSIL2A.xml)
    para extraer la geometría sol-sensor (SZA, VZA, RAA).
    """
    try:
        from xml.etree import ElementTree as ET
        content = await file.read()
        root = ET.fromstring(content)

        # --- Sun angles (Mean_Sun_Angle) ---
        sza, saa = None, None
        # Try MTD_TL.xml structure first
        for sun_el in root.iter('Mean_Sun_Angle'):
            z = sun_el.find('ZENITH_ANGLE')
            a = sun_el.find('AZIMUTH_ANGLE')
            if z is not None:
                sza = float(z.text)
            if a is not None:
                saa = float(a.text)
            break

        # Try MTD_MSIL2A.xml structure (Sun_Angles_Grid > Sun_Angles_List)
        if sza is None:
            for z_el in root.iter('ZENITH_ANGLE'):
                parent = z_el
                # Walk up to check if under Sun context
                sza = float(z_el.text)
                break

        if sza is None:
            return JSONResponse({"error": "No se encontró Mean_Sun_Angle en el XML. "
                                 "Asegúrate de subir el fichero MTD_TL.xml del granule."}, status_code=400)

        # --- Viewing angles (Mean_Viewing_Incidence_Angle_List) ---
        # Collect VZA and VAA per band, then average
        # SNAP8 bands: B3(2), B4(3), B5(4), B6(5), B7(6), B8A(8), B11(11), B12(12)
        snap8_band_ids = {2, 3, 4, 5, 6, 8, 11, 12}
        vza_vals, vaa_vals = [], []
        vza_all, vaa_all = [], []  # fallback: all bands

        for angle_el in root.iter('Mean_Viewing_Incidence_Angle'):
            bid_str = angle_el.get('bandId')
            z = angle_el.find('ZENITH_ANGLE')
            a = angle_el.find('AZIMUTH_ANGLE')
            if z is not None and a is not None:
                vz = float(z.text)
                va = float(a.text)
                vza_all.append(vz)
                vaa_all.append(va)
                if bid_str is not None:
                    bid = int(bid_str)
                    if bid in snap8_band_ids:
                        vza_vals.append(vz)
                        vaa_vals.append(va)

        # Use SNAP8 bands if available, otherwise all bands
        if vza_vals:
            vza_mean = sum(vza_vals) / len(vza_vals)
            vaa_mean = sum(vaa_vals) / len(vaa_vals)
        elif vza_all:
            vza_mean = sum(vza_all) / len(vza_all)
            vaa_mean = sum(vaa_all) / len(vaa_all)
        else:
            return JSONResponse({"error": "No se encontró Mean_Viewing_Incidence_Angle en el XML."}, status_code=400)

        # --- Compute RAA ---
        raa = abs(saa - vaa_mean)
        if raa > 180:
            raa = 360 - raa

        result = {
            "sza": round(sza, 2),
            "saa": round(saa, 2),
            "vza": round(vza_mean, 2),
            "vaa": round(vaa_mean, 2),
            "raa": round(raa, 2),
            "n_bands_used": len(vza_vals) if vza_vals else len(vza_all),
            "source": file.filename
        }
        print(f"  [Metadata] SZA={result['sza']}° VZA={result['vza']}° "
              f"RAA={result['raa']}° (SAA={result['saa']}° VAA={result['vaa']}°) "
              f"from {file.filename}")
        return result

    except ET.ParseError as e:
        return JSONResponse({"error": f"Error parseando XML: {e}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/lut/generate")
async def create_lut(request: LUTRequest):
    """Genera LUT usando modelo PROSAIL Python (PROSPECT-D + 4SAIL)"""
    lut_id = str(uuid.uuid4())
    try:
        spectra, params = generate_lut(
            request.n_samples,
            request.ranges.dict(),
            request.geometry.dict(),
            noise_sigma=request.noise_sigma,
            distribution=request.distribution,
            band_selection=request.band_selection
        )
        tree = cKDTree(spectra)

        # Store band selection info for inversion
        if request.band_selection == 'snap8':
            band_idx = SNAP8_BAND_INDICES
        else:
            band_idx = list(range(10))

        lut_data = {
            'spectra': spectra.astype(np.float32),
            'params': params,
            'tree': tree,
            'geometry': request.geometry.dict(),
            'ranges': request.ranges.dict(),
            'n_samples': spectra.shape[0],
            'n_bands': spectra.shape[1],
            'band_idx': band_idx,
            'band_selection': request.band_selection,
            'noise_sigma': request.noise_sigma,
            'distribution': request.distribution,
            'created': datetime.now().isoformat()
        }
        lut_path = os.path.join(TEMP_DIR, f"{lut_id}.pkl")
        with open(lut_path, 'wb') as f:
            pickle.dump(lut_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(lut_path) / (1024*1024)
        return {
            "lut_id": lut_id, "n_samples": lut_data['n_samples'],
            "n_bands": spectra.shape[1], "storage_mb": round(size_mb, 1),
            "band_selection": request.band_selection,
            "distribution": request.distribution,
            "noise_sigma": request.noise_sigma,
            "status": "ready"
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/invert/image")
async def invert_image(
    files: List[UploadFile] = File(...),
    lut_id: str = Form(...),
    k_best: int = Form(30),
    target_resolution: int = Form(20)
):
    """Inversión de imagen S2. Acepta GeoTIFF o archivos DIMAP (.dim + .img + .hdr)"""
    job_id = str(uuid.uuid4())
    work_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    has_dim = False
    has_tif = False

    for file in files:
        if not file.filename or file.filename.strip() == '':
            continue
        fname = file.filename.replace('\\', '/')
        # Preserve subdirectory structure (webkitdirectory paths)
        parts = fname.split('/')
        if len(parts) > 1:
            subdir = os.path.join(work_dir, *parts[:-1])
            os.makedirs(subdir, exist_ok=True)
            file_path = os.path.join(subdir, parts[-1])
        else:
            file_path = os.path.join(work_dir, parts[-1])
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        if fname.lower().endswith('.dim'): has_dim = True
        if fname.lower().endswith(('.tif', '.tiff')): has_tif = True

    active_jobs[job_id] = {
        "status": "processing", "progress": 0,
        "message": "Iniciando...", "format": "DIMAP" if has_dim else "GeoTIFF"
    }
    # Launch worker in a real OS thread (not in asyncio event loop)
    t = threading.Thread(
        target=_inversion_worker,
        args=(job_id, work_dir, lut_id, k_best, has_dim),
        daemon=True
    )
    t.start()
    return {"job_id": job_id, "status": "processing", "format": "DIMAP" if has_dim else "GeoTIFF"}

@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]

@app.get("/api/download/{job_id}/{variable}")
async def download_result(job_id: str, variable: str):
    """Descarga resultado como GeoTIFF o float32 binario"""
    if variable not in ['LAI', 'Cab', 'Cw', 'CCC', 'unc_LAI', 'all']:
        raise HTTPException(status_code=400, detail="Variable inválida")
    # Prefer GeoTIFF
    tif_path = os.path.join(TEMP_DIR, f"{job_id}_{variable}.tif")
    if os.path.exists(tif_path):
        return FileResponse(tif_path, filename=f"PROSAIL_{variable}.tif", media_type="image/tiff")
    bin_path = os.path.join(TEMP_DIR, f"{job_id}_{variable}.bin")
    if os.path.exists(bin_path):
        return FileResponse(bin_path, filename=f"PROSAIL_{variable}.bin", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Resultado no disponible aún")

@app.get("/api/result/{job_id}/{variable}/raw")
async def download_raw(job_id: str, variable: str):
    """Descarga float32 binario para visualización en frontend"""
    bin_path = os.path.join(TEMP_DIR, f"{job_id}_{variable}.bin")
    if not os.path.exists(bin_path):
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(bin_path, media_type="application/octet-stream")

@app.get("/api/result/{job_id}/meta")
async def result_meta(job_id: str):
    meta_path = os.path.join(TEMP_DIR, f"{job_id}_meta.pkl")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Not ready")
    with open(meta_path, 'rb') as f:
        return pickle.load(f)

# =============================================================================
# INVERSION TASK — runs in OS thread (not asyncio)
# =============================================================================

def _inversion_worker(job_id: str, work_dir: str, lut_id: str,
                      k_best: int, is_dimap: bool):
    """
    Synchronous worker — runs in OS thread.
    Updates active_jobs dict for HTTP polling progress.
    """
    try:
        # 1. Load LUT
        lut_path = os.path.join(TEMP_DIR, f"{lut_id}.pkl")
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT {lut_id} no encontrada")
        with open(lut_path, 'rb') as f:
            lut = pickle.load(f)
        if 'tree' not in lut or lut['tree'] is None:
            lut['tree'] = cKDTree(lut['spectra'])
        tree = lut['tree']

        # Get band selection from LUT
        band_idx = lut.get('band_idx', list(range(10)))
        n_lut_bands = lut['spectra'].shape[1]
        band_sel_name = lut.get('band_selection', 'all10')
        active_jobs[job_id].update({
            "progress": 5,
            "message": f"LUT cargada: {lut['n_samples']} muestras × {n_lut_bands} bandas ({band_sel_name})"
        })

        # 2. Read image
        active_jobs[job_id].update({"progress": 8, "message": "Leyendo imagen..."})
        profile = None

        if is_dimap:
            stack, width, height, band_names, profile = read_dimap_bands(work_dir)
        else:
            if not HAS_RASTERIO:
                raise ImportError("rasterio no instalado — necesario para GeoTIFF")
            tif_files = glob.glob(os.path.join(work_dir, '**/*.tif'), recursive=True)
            tif_files += glob.glob(os.path.join(work_dir, '**/*.tiff'), recursive=True)
            if not tif_files:
                raise FileNotFoundError("No se encontró GeoTIFF")
            with rasterio.open(tif_files[0]) as src:
                profile = src.profile.copy()
                # Sanitizar perfil: si nodata es NaN, eliminarlo para evitar
                # que GDAL/rasterio aplique masking inesperado en la lectura.
                # Los NaN en los datos se manejan explícitamente más adelante.
                src_nodata = src.nodata
                if src_nodata is not None and (isinstance(src_nodata, float) and np.isnan(src_nodata)):
                    profile.pop("nodata", None)
                    print("  [GeoTIFF] nodata=NaN detectado en perfil, eliminado para compatibilidad")
                n_bands = min(src.count, 10)
                height, width = src.height, src.width

                # --- Detect which S2 bands are in the file ---
                # Try band descriptions/names from metadata
                band_mapping = None  # will be list of S2 indices (0-9) for each file band

                # Check rasterio band descriptions
                if src.descriptions:
                    descs = [d for d in src.descriptions if d]
                    if len(descs) == n_bands:
                        mapping = []
                        for desc in descs:
                            matched_idx = None
                            desc_clean = desc.strip().lower()
                            # Pass 1: exact match (critical to distinguish B8 vs B8A)
                            for s2i, s2name in enumerate(S2_BAND_NAMES):
                                for alias in S2_BAND_ALIASES[s2name]:
                                    if desc_clean == alias.lower():
                                        matched_idx = s2i
                                        break
                                if matched_idx is not None:
                                    break
                            # Pass 2: substring match, but check longer names first
                            # (B8A before B8) to avoid false positives
                            if matched_idx is None:
                                for s2i in sorted(range(len(S2_BAND_NAMES)),
                                                  key=lambda i: len(S2_BAND_NAMES[i]), reverse=True):
                                    for alias in S2_BAND_ALIASES[S2_BAND_NAMES[s2i]]:
                                        if alias.lower() in desc_clean:
                                            matched_idx = s2i
                                            break
                                    if matched_idx is not None:
                                        break
                            mapping.append(matched_idx)
                        if all(m is not None for m in mapping):
                            band_mapping = mapping
                            print(f"  [GeoTIFF] Bandas desde metadata: {[S2_BAND_NAMES[i] for i in mapping]}")

                # Heuristic by band count if metadata didn't help
                if band_mapping is None:
                    if n_bands == 10:
                        # Assume standard S2 order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
                        band_mapping = list(range(10))
                        print(f"  [GeoTIFF] 10 bandas → asumiendo orden S2 estándar")
                    elif n_bands == 8:
                        # Assume SNAP Biophysical Processor output: B3,B4,B5,B6,B7,B8A,B11,B12
                        band_mapping = [1, 2, 3, 4, 5, 7, 8, 9]  # S2 indices for SNAP's 8 bands
                        print(f"  [GeoTIFF] 8 bandas → asumiendo orden SNAP (B3,B4,B5,B6,B7,B8A,B11,B12)")
                    else:
                        # Generic: place sequentially starting from B2
                        band_mapping = list(range(n_bands))
                        print(f"  [GeoTIFF] {n_bands} bandas → asumiendo orden secuencial desde B2")

                # Build fixed 10-band stack with bands at correct S2 positions
                stack = np.full((10, height, width), np.nan, dtype=np.float32)
                for file_band_i in range(n_bands):
                    s2_pos = band_mapping[file_band_i]
                    stack[s2_pos] = src.read(file_band_i + 1).astype(np.float32)

                # Auto-scale (use first non-NaN band for detection)
                first_valid = None
                for i in range(10):
                    if not np.all(np.isnan(stack[i])):
                        first_valid = i
                        break
                if first_valid is not None:
                    samp = stack[first_valid].ravel()
                    samp = samp[(samp > 0) & np.isfinite(samp)]
                    if len(samp) > 0:
                        p99 = np.percentile(samp, 99)
                        if p99 > 100:
                            stack /= 10000.0
                            print(f"  [GeoTIFF] Auto-escala: ÷10000 (p99={p99:.0f})")
                        elif p99 > 1.5:
                            stack /= 100.0
                            print(f"  [GeoTIFF] Auto-escala: ÷100 (p99={p99:.2f})")
                stack[(stack < 0) | (stack > 1.5) | ~np.isfinite(stack)] = np.nan

            filled = [S2_BAND_NAMES[i] for i in range(10) if not np.all(np.isnan(stack[i]))]
            band_names = filled
            print(f"  [GeoTIFF] Bandas en stack: {', '.join(filled)} ({len(filled)}/10)")

        n_bands_actual = sum(1 for i in range(10) if not np.all(np.isnan(stack[i])))
        n_pixels = width * height
        active_jobs[job_id].update({
            "progress": 15, "message": f"Imagen: {width}×{height}, {n_bands_actual} bandas disponibles",
            "width": width, "height": height
        })

        # 3. Prepare pixel matrix — select bands matching LUT from fixed 10-band stack
        # Stack is always [10, H, W] with fixed positions (B2=0, ..., B12=9)
        # Check which LUT bands are available in the image
        available = [i for i in band_idx if not np.all(np.isnan(stack[i]))]
        missing = [S2_BAND_NAMES[i] for i in band_idx if np.all(np.isnan(stack[i]))]
        if missing:
            msg = f"⚠ Bandas no disponibles: {', '.join(missing)} — se usarán {len(available)}/{n_lut_bands}"
            active_jobs[job_id].update({"progress": 18, "message": msg})
            print(f"  {msg}")

        stack_sel = stack[band_idx]  # [n_lut_bands, H, W]
        del stack  # free memory

        stack_2d = stack_sel.reshape(n_lut_bands, n_pixels).T  # [n_pixels, n_lut_bands]
        del stack_sel

        valid_mask = np.all(np.isfinite(stack_2d) & (stack_2d > 0) & (stack_2d < 1.0), axis=1)
        n_valid = int(valid_mask.sum())
        active_jobs[job_id].update({
            "progress": 20, "message": f"Píxeles válidos: {n_valid:,} / {n_pixels:,} ({100*n_valid/max(n_pixels,1):.1f}%)"
        })

        # Diagnostic: show why pixels are being rejected
        if n_valid == 0:
            for bi in range(n_lut_bands):
                col = stack_2d[:, bi]
                n_fin = np.sum(np.isfinite(col))
                n_pos = np.sum(col > 0) if n_fin > 0 else 0
                n_lt1 = np.sum(col < 1.0) if n_fin > 0 else 0
                band_name = S2_BAND_NAMES[band_idx[bi]] if bi < len(band_idx) else f"band_{bi}"
                print(f"  [Diag] {band_name}: finite={n_fin}/{n_pixels}, >0={n_pos}, <1={n_lt1}")
            samp = stack_2d[0] if n_pixels > 0 else []
            print(f"  [Diag] Primer píxel: {samp}")
        elif n_valid > 0:
            samp_idx = np.where(valid_mask)[0][0]
            samp = stack_2d[samp_idx]
            bands_str = [S2_BAND_NAMES[band_idx[i]] for i in range(n_lut_bands)]
            print(f"  [Diag] Espectro ejemplo ({bands_str}): {np.round(samp, 4)}")

        # 4. Invert in chunks using cKDTree
        results = {k: np.full(n_pixels, -9999, dtype=np.float32) for k in ['LAI','Cab','Cw','CCC','unc_LAI']}

        # Diagnostic: check LUT integrity
        lut_n = lut['spectra'].shape[0]
        lut_params_n = len(lut['params'].get('LAI', []))
        print(f"  [Diag] LUT: {lut_n} spectra, params LAI: {lut_params_n}, k_best: {k_best}")
        print(f"  [Diag] Valid pixels: {n_valid} / {n_pixels}")
        if lut_params_n == 0:
            raise ValueError("LUT params vacíos — regenera la LUT desde la webapp")
        if lut_params_n != lut_n:
            print(f"  ⚠ [Diag] MISMATCH: spectra={lut_n} vs params={lut_params_n}")

        if n_valid > 0:
            valid_spectra = stack_2d[valid_mask]
            valid_idx = np.where(valid_mask)[0]
            chunk_size = 50000

            for c0 in range(0, n_valid, chunk_size):
                c1 = min(c0 + chunk_size, n_valid)
                chunk = valid_spectra[c0:c1]

                # Limit k to LUT size
                k_actual = min(k_best, lut_n)
                if k_actual < 1:
                    raise ValueError(f"LUT demasiado pequeña ({lut_n}) para k_best={k_best}")

                try:
                    distances, indices = tree.query(chunk, k=k_actual)
                    # Ensure 2D shape even if k=1
                    if indices.ndim == 1:
                        distances = distances[:, np.newaxis]
                        indices = indices[:, np.newaxis]
                except Exception as e:
                    print(f"  ✗ [Diag] tree.query failed: {e}, chunk shape: {chunk.shape}")
                    raise

                weights = 1.0 / (distances + 1e-10)
                weights /= weights.sum(axis=1, keepdims=True)

                try:
                    lai = np.sum(lut['params']['LAI'][indices] * weights, axis=1)
                    cab = np.sum(lut['params']['Cab'][indices] * weights, axis=1)
                    cw  = np.sum(lut['params']['Cw'][indices] * weights, axis=1)
                    unc = np.std(lut['params']['LAI'][indices], axis=1)
                except IndexError as e:
                    print(f"  ✗ [Diag] IndexError: {e}")
                    print(f"    indices shape: {indices.shape}, range: [{indices.min()}, {indices.max()}]")
                    print(f"    LAI shape: {lut['params']['LAI'].shape}")
                    print(f"    chunk shape: {chunk.shape}, distances shape: {distances.shape}")
                    raise

                idx = valid_idx[c0:c1]
                results['LAI'][idx] = lai
                results['Cab'][idx] = cab
                results['Cw'][idx] = cw
                results['CCC'][idx] = lai * cab / 100.0
                results['unc_LAI'][idx] = unc

                pct = 20 + int(70 * c1 / n_valid)
                active_jobs[job_id].update({
                    "progress": pct,
                    "message": f"Invirtiendo... {c1:,}/{n_valid:,} px ({100*c1/n_valid:.0f}%)"
                })

        del stack_2d, valid_mask  # free memory

        # 5. Save results as GeoTIFF (+ binary for web preview)
        active_jobs[job_id].update({"progress": 92, "message": "Guardando GeoTIFF..."})
        
        # Build fallback profile if none exists
        if profile is None and HAS_RASTERIO:
            profile = {
                'driver': 'GTiff', 'height': height, 'width': width,
                'count': 1, 'dtype': 'float32', 'compress': 'lzw', 'nodata': -9999
            }
        
        stats = {}
        geotiff_created = False
        for var, data in results.items():
            arr = data.reshape(height, width)
            # Binary for frontend canvas visualization
            bin_path = os.path.join(TEMP_DIR, f"{job_id}_{var}.bin")
            arr.tofile(bin_path)
            # GeoTIFF (always when rasterio available)
            if HAS_RASTERIO and profile is not None:
                tif_path = os.path.join(TEMP_DIR, f"{job_id}_{var}.tif")
                out_p = profile.copy()
                out_p.update(count=1, dtype='float32', compress='lzw', nodata=-9999)
                try:
                    with rasterio.open(tif_path, 'w', **out_p) as dst:
                        dst.write(arr, 1)
                    geotiff_created = True
                except Exception as e:
                    print(f"  Warning: GeoTIFF write failed for {var}: {e}")
            vd = data[data != -9999]
            if len(vd) > 0:
                stats[var] = {
                    'min': round(float(np.nanmin(vd)), 4),
                    'max': round(float(np.nanmax(vd)), 4),
                    'mean': round(float(np.nanmean(vd)), 4),
                    'std': round(float(np.nanstd(vd)), 4),
                    'valid_pct': round(float(100 * len(vd) / n_pixels), 1)
                }

        meta = {
            'width': width, 'height': height,
            'n_pixels': n_pixels, 'n_valid': n_valid,
            'bands': band_names, 'stats': stats,
            'variables': list(results.keys()),
            'has_geotiff': geotiff_created,
            'has_georef': profile is not None and 'crs' in profile and 'transform' in profile
        }
        with open(os.path.join(TEMP_DIR, f"{job_id}_meta.pkl"), 'wb') as f:
            pickle.dump(meta, f)

        active_jobs[job_id].update({
            "status": "completed", "progress": 100, "message": "¡Completado!",
            "meta": meta
        })
        shutil.rmtree(work_dir, ignore_errors=True)

    except Exception as e:
        import traceback; traceback.print_exc()
        active_jobs[job_id].update({"status": "error", "progress": 0, "message": str(e)})


# =============================================================================
# SERVE FRONTEND
# =============================================================================
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("  PROSAIL S2 Inversion Server v3.0")
    print("  PROSPECT-D + 4SAIL | Baret distributions | Gaussian noise")
    print("  → http://localhost:8000")
    print("  (HTTP polling — no WebSocket)")
    print("="*60)
    uvicorn.run(app, host="localhost", port=8000)
