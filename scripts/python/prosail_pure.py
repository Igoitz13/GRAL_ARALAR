"""
PROSAIL wrapper v3 for Sentinel-2 biophysical variable retrieval.
Uses the validated `prosail` library (Gómez-Dans) — PROSPECT-D + 4SAIL.

Improvements over v2.3:
  - ALA (Average Leaf Angle) as variable parameter
  - Soil brightness (rsoil) and moisture (psoil) as variable parameters
  - Gaussian noise injection on LUT spectra (regularization)
  - Baret et al. (2007) / SNAP-like truncated-Gaussian distributions
  - Band selection: 10 bands (all) or 8 bands (SNAP Biophysical Processor)

Ref: Jacquemoud et al. (2009), Verhoef et al. (2007), Féret et al. (2017)
     Baret et al. (2007) — LAI-2000 validation, SNAP S2 Toolbox
     Weiss & Baret (2016) — S2ToolBox Level 2 products, SNAP Biophysical
     Verrelst et al. (2015) — ARTMO toolbox (UV Valencia)
"""
import numpy as np
from typing import Union, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import prosail as _prosail

# =============================================================================
# Sentinel-2A Spectral Response Functions (box-filter approximation)
# =============================================================================
S2_BAND_INFO_ALL = [
    # idx, name, center_nm, fwhm_nm
    (0, 'B2',   490,  65),   # Blue
    (1, 'B3',   560,  35),   # Green
    (2, 'B4',   665,  30),   # Red
    (3, 'B5',   705,  15),   # Red Edge 1
    (4, 'B6',   740,  15),   # Red Edge 2
    (5, 'B7',   783,  20),   # Red Edge 3
    (6, 'B8',   842, 115),   # NIR broad
    (7, 'B8A',  865,  20),   # NIR narrow
    (8, 'B11', 1610,  90),   # SWIR 1
    (9, 'B12', 2190, 180),   # SWIR 2
]

# SNAP Biophysical Processor uses 8 bands (excludes B2 and B8):
# B2 (blue) — too sensitive to residual aerosol correction errors
# B8 (NIR broad, 115nm FWHM) — redundant with B8A and SRF too wide
SNAP8_BAND_INDICES = [1, 2, 3, 4, 5, 7, 8, 9]  # B3,B4,B5,B6,B7,B8A,B11,B12
SNAP8_BAND_NAMES = ['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

S2_WAVELENGTHS = np.array([b[2] for b in S2_BAND_INFO_ALL], dtype=np.float32)

# Precompute band integration slices (offset from 400nm)
_BAND_SLICES_10 = []
for _, _, center, fwhm in S2_BAND_INFO_ALL:
    lo = max(center - fwhm // 2, 400) - 400
    hi = min(center + fwhm // 2, 2500) - 400
    _BAND_SLICES_10.append((lo, hi))

# =============================================================================
# Baret et al. (2007) / SNAP distributions — truncated Gaussian
# Used by SNAP Biophysical Processor and referenced by ARTMO
# =============================================================================
BARET_DISTRIBUTIONS = {
    # param: (mean, std, min, max)
    'N':      (1.5,   0.3,   1.0,  3.0),
    'Cab':    (45.0,  30.0,  1.0,  90.0),
    'Car':    (10.0,  5.0,   0.0,  25.0),
    'Cbrown': (0.15,  0.2,   0.0,  2.0),
    'Cw':     (0.015, 0.008, 0.002, 0.05),
    'Cm':     (0.009, 0.005, 0.001, 0.025),
    'LAI':    (3.0,   3.0,   0.0,  15.0),  # Wide for agriculture
    'ALA':    (55.0,  15.0,  20.0, 80.0),
    'hot':    (0.1,   0.1,   0.01, 0.5),
    'psoil':  (0.5,   0.3,   0.0,  1.0),
    'rsoil':  (1.0,   0.5,   0.2,  3.5),
}


def _integrate_s2_10(spectrum_2101: np.ndarray) -> np.ndarray:
    """Integrate 1nm PROSAIL output over all 10 S2 bands."""
    result = np.empty(10, dtype=np.float32)
    for i, (lo, hi) in enumerate(_BAND_SLICES_10):
        result[i] = np.mean(spectrum_2101[lo:hi + 1])
    return result


def run(N: float, Cab: float, Car: float, Cbrown: float,
        Cw: float, Cm: float, LAI: float, hot: float,
        sza: float, vza: float, raa: float,
        ant: float = 0.0, ALA: float = 55.0,
        psoil: float = 1.0, rsoil: float = 1.0,
        factor: int = 0, varnames: bool = False) -> np.ndarray:
    """
    Run PROSAIL (PROSPECT-D + 4SAIL) and return S2-integrated reflectance.
    Returns 10-band array [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12].
    """
    spectrum = _prosail.run_prosail(
        n=float(N), cab=float(Cab), car=float(Car),
        cbrown=float(Cbrown), cw=float(Cw), cm=float(Cm),
        lai=float(LAI), lidfa=float(ALA), hspot=float(hot),
        tts=float(sza), tto=float(vza), psi=float(raa),
        ant=float(ant),
        prospect_version='D',
        typelidf=2,
        factor='SDR',
        rsoil=float(rsoil), psoil=float(psoil)
    )
    s2_refl = _integrate_s2_10(spectrum)
    if factor == 1:
        s2_refl *= np.float32(np.pi)
    return s2_refl


def prospect_d(N, Cab, Cw, Cdm, Car=None, Cbrown=0.0):
    """PROSPECT-D leaf optics. Returns (r_10, t_10) integrated over S2."""
    if Car is None:
        Car = Cab * 0.25
    N = float(np.asarray(N).ravel()[0]) if np.ndim(N) > 0 else float(N)
    Cab = float(np.asarray(Cab).ravel()[0]) if np.ndim(Cab) > 0 else float(Cab)
    Cw = float(np.asarray(Cw).ravel()[0]) if np.ndim(Cw) > 0 else float(Cw)
    Cdm = float(np.asarray(Cdm).ravel()[0]) if np.ndim(Cdm) > 0 else float(Cdm)
    Car = float(np.asarray(Car).ravel()[0]) if np.ndim(Car) > 0 else float(Car)
    Cbrown = float(np.asarray(Cbrown).ravel()[0]) if np.ndim(Cbrown) > 0 else float(Cbrown)
    wl, r_full, t_full = _prosail.run_prospect(
        n=N, cab=Cab, car=Car, cbrown=Cbrown,
        cw=Cw, cm=Cdm, ant=0.0, prospect_version='D'
    )
    return _integrate_s2_10(r_full), _integrate_s2_10(t_full)


def _sample_truncated_gaussian(mean, std, lo, hi, n):
    """Sample from truncated Gaussian via rejection sampling."""
    samples = np.empty(n, dtype=np.float32)
    filled = 0
    while filled < n:
        need = n - filled
        batch = np.random.normal(mean, std, size=int(need * 1.5 + 100)).astype(np.float32)
        good = batch[(batch >= lo) & (batch <= hi)]
        take = min(len(good), need)
        samples[filled:filled + take] = good[:take]
        filled += take
    return samples


def _sample_lhs_uniform(n, lo, hi):
    """Latin Hypercube Sampling — stratified uniform."""
    perm = np.random.permutation(n)
    return np.float32(lo + (perm + np.random.rand(n)) / n * (hi - lo))


def _sample_lhs_gaussian(n, mean, std, lo, hi):
    """LHS with truncated Gaussian CDF — best of both worlds."""
    from scipy.stats import truncnorm
    a_tn = (lo - mean) / max(std, 1e-10)
    b_tn = (hi - mean) / max(std, 1e-10)
    dist = truncnorm(a_tn, b_tn, loc=mean, scale=std)
    # Stratified quantiles
    perm = np.random.permutation(n)
    u = (perm + np.random.rand(n)) / n
    return dist.ppf(u).astype(np.float32)


def _resolve_geometry(geometry: Dict, n_samples: int) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Resolve a geometry spec into per-sample arrays for tts/tto/psi.

    Each of 'tts', 'tto', 'psi' can be:
      - scalar (float/int): fixed geometry, broadcast to all samples (legacy behaviour)
      - 2-tuple/list (lo, hi): sampled with LHS uniform over [lo, hi] (marginalisation)

    Defaults if missing: tts=30°, tto=0°, psi=0° (fixed).

    Returns:
        arrays: dict {'tts','tto','psi'} of float32 arrays of length n_samples
        sampled: True if at least one angle was sampled (for logging)
    """
    defaults = {'tts': 30.0, 'tto': 0.0, 'psi': 0.0}
    arrays = {}
    sampled_any = False
    for key in ('tts', 'tto', 'psi'):
        val = geometry.get(key, defaults[key])
        if isinstance(val, (tuple, list)) and len(val) == 2:
            lo, hi = float(val[0]), float(val[1])
            if hi < lo:
                lo, hi = hi, lo
            if hi > lo:
                arrays[key] = _sample_lhs_uniform(n_samples, lo, hi)
                sampled_any = True
            else:
                arrays[key] = np.full(n_samples, lo, dtype=np.float32)
        else:
            arrays[key] = np.full(n_samples, float(val), dtype=np.float32)
    return arrays, sampled_any


def generate_lut(n_samples: int, param_ranges: Dict,
                 geometry: Dict,
                 noise_sigma: float = 0.01,
                 distribution: str = 'baret',
                 band_selection: str = 'snap8'
                 ) -> Tuple[np.ndarray, Dict]:
    """
    Generate Look-Up Table using PROSAIL (PROSPECT-D + 4SAIL).

    Args:
        n_samples:      Number of LUT entries
        param_ranges:   Dict {param_name: (min, max)}
                        Supported: N, Cab, Car, Cbrown, Cw, Cm, LAI, hot, ALA, psoil, rsoil
        geometry:       Dict {tts, tto, psi}. Each value can be either:
                          - scalar float: fixed geometry (legacy)
                          - 2-tuple (lo, hi): sampled with LHS uniform — enables
                            geometric marginalisation so a single NN can be used
                            across scenes with different sun-sensor geometries.
                        Recommended ranges for Sentinel-2 over a 43°N site
                        (April–October): tts=(20,60), tto=(0,12), psi=(0,180).
        noise_sigma:    Gaussian noise std added to LUT spectra (0=none, 0.01=ARTMO default)
        distribution:   'uniform' = uniform sampling
                        'baret'   = Baret et al. (2007) truncated Gaussian (SNAP-like)
        band_selection: 'all10'  = all 10 S2 bands
                        'snap8'  = 8 bands like SNAP (B3,B4,B5,B6,B7,B8A,B11,B12)

    Returns:
        spectra: ndarray [n_valid, n_bands]  — n_bands = 10 or 8
        params:  Dict of parameter arrays (includes per-sample tts/tto/psi)
    """
    # Determine which bands to use
    if band_selection == 'snap8':
        band_idx = np.array(SNAP8_BAND_INDICES)
        n_bands_out = 8
    else:
        band_idx = np.arange(10)
        n_bands_out = 10

    # --- Sample parameters ---
    params = {}

    # Core PROSPECT-D + 4SAIL parameters with their ranges
    all_param_names = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'hot', 'ALA', 'psoil', 'rsoil']

    for pname in all_param_names:
        if pname in param_ranges:
            pmin, pmax = param_ranges[pname]
        elif pname in BARET_DISTRIBUTIONS:
            # Use Baret defaults if not specified
            _, _, pmin, pmax = BARET_DISTRIBUTIONS[pname]
        else:
            continue

        if distribution == 'baret' and pname in BARET_DISTRIBUTIONS:
            mean, std, d_min, d_max = BARET_DISTRIBUTIONS[pname]
            # Clip Baret range to user-specified range
            lo = max(pmin, d_min)
            hi = min(pmax, d_max)
            try:
                params[pname] = _sample_lhs_gaussian(n_samples, mean, std, lo, hi)
            except Exception:
                params[pname] = _sample_lhs_uniform(n_samples, lo, hi)
        else:
            params[pname] = _sample_lhs_uniform(n_samples, pmin, pmax)

    # Defaults for missing optional parameters
    if 'Car' not in params:
        params['Car'] = params['Cab'] * np.float32(0.25)
    if 'Cbrown' not in params:
        params['Cbrown'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    if 'ALA' not in params:
        params['ALA'] = np.full(n_samples, 55.0, dtype=np.float32)
    if 'psoil' not in params:
        params['psoil'] = np.full(n_samples, 1.0, dtype=np.float32)
    if 'rsoil' not in params:
        params['rsoil'] = np.full(n_samples, 1.0, dtype=np.float32)

    # --- Resolve geometry: scalar → fixed, (lo,hi) → LHS uniform ---
    geom_arrays, geom_sampled = _resolve_geometry(geometry, n_samples)
    if geom_sampled:
        rng_strs = []
        for k in ('tts', 'tto', 'psi'):
            arr = geom_arrays[k]
            if arr.min() < arr.max():
                rng_strs.append(f"{k}∈[{arr.min():.1f}°,{arr.max():.1f}°]")
            else:
                rng_strs.append(f"{k}={arr[0]:.1f}°")
        print(f"  Geometria muestreada: {', '.join(rng_strs)}")
    else:
        print(f"  Geometria fija: tts={geom_arrays['tts'][0]:.1f}° "
              f"tto={geom_arrays['tto'][0]:.1f}° psi={geom_arrays['psi'][0]:.1f}°")

    # Track geometry per sample so the training pipeline can use it as feature
    params['tts'] = geom_arrays['tts']
    params['tto'] = geom_arrays['tto']
    params['psi'] = geom_arrays['psi']

    spectra_10 = np.zeros((n_samples, 10), dtype=np.float32)

    for i in range(n_samples):
        try:
            spectra_10[i] = run(
                N=params['N'][i], Cab=params['Cab'][i], Car=params['Car'][i],
                Cbrown=params['Cbrown'][i], Cw=params['Cw'][i], Cm=params['Cm'][i],
                LAI=params['LAI'][i], hot=params['hot'][i],
                ALA=params['ALA'][i],
                psoil=params['psoil'][i], rsoil=params['rsoil'][i],
                sza=float(geom_arrays['tts'][i]),
                vza=float(geom_arrays['tto'][i]),
                raa=float(geom_arrays['psi'][i])
            )
        except Exception:
            spectra_10[i] = np.nan

        if (i + 1) % max(1, n_samples // 20) == 0:
            print(f"  LUT: {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")

    # --- Remove failed samples ---
    valid = np.all(np.isfinite(spectra_10) & (spectra_10 > 0), axis=1)
    if not np.all(valid):
        n_bad = n_samples - valid.sum()
        print(f"  Warning: {n_bad} samples removed (invalid spectra)")
        spectra_10 = spectra_10[valid]
        for k in params:
            params[k] = params[k][valid]

    # --- Add Gaussian noise (regularization) ---
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, spectra_10.shape).astype(np.float32)
        spectra_10 = np.clip(spectra_10 + noise, 0.001, 1.0)
        print(f"  Ruido gaussiano: σ = {noise_sigma:.3f}")

    # --- Band selection ---
    spectra_out = spectra_10[:, band_idx]

    n_valid = spectra_out.shape[0]
    band_names = [S2_BAND_INFO_ALL[i][1] for i in band_idx]
    print(f"  LUT final: {n_valid} muestras × {n_bands_out} bandas ({', '.join(band_names)})")

    return spectra_out, params


# Aliases for backward compatibility
run_vectorized = run
