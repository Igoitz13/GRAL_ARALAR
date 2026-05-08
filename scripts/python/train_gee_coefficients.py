#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  train_gee_coefficients.py                                       ║
║  Entrena modelos de regresión sobre la LUT PROSAIL y exporta     ║
║  los coeficientes como código JavaScript para GEE.               ║
╚══════════════════════════════════════════════════════════════════╝

Estrategia: Red neuronal de 1 capa oculta (5 neuronas, tanh)
            — misma arquitectura que el Biophysical Processor de SNAP
            (Weiss & Baret, 2016; S2ToolBox Level 2 products).

Entrada:  8 reflectancias S2 normalizadas + cos(SZA) + cos(VZA) + cos(RAA)
Salida:   1 variable biofísica por modelo (LAI, Cab, fCover, CCC, CWC)

Uso:
  python train_gee_coefficients.py temp_prosail/<uuid>.pkl
  python train_gee_coefficients.py temp_prosail/<uuid>.pkl --output coefs.js

Dependencias:
  pip install numpy scikit-learn
  (scipy ya está instalado si tienes la webapp)
"""

import pickle
import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# ── Configuración de bandas ──
SNAP8_NAMES = ['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
S2_10_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# Variables biofísicas a modelar
TARGET_VARS = {
    'LAI':    {'transform': 'none',  'desc': 'Leaf Area Index (m²/m²)'},
    'Cab':    {'transform': 'none',  'desc': 'Chlorophyll a+b (µg/cm²)'},
    'fCover': {'transform': 'none',  'desc': 'Fraction of vegetation cover'},
    'CCC':    {'transform': 'none',  'desc': 'Canopy Chlorophyll Content (µg/cm²)'},
    'CWC':    {'transform': 'none',  'desc': 'Canopy Water Content (g/m²)'},
}


def load_lut(pkl_path):
    """Carga el pickle de la LUT de la webapp."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_features(spectra, band_names, params):
    """
    Prepara features de entrada para el modelo (esquema canónico tipo SNAP).

    Input: 8 reflectancias S2 (snap8) + ángulos sol-sensor por muestra
    Features: 8 reflectancias normalizadas + cos(SZA) + cos(VZA) + cos(RAA)
    Total: 11 features

    Justificación del cambio respecto a la versión índices-NDVI/NDRE/NMDI:
    El estándar Weiss & Baret (2016) — y SNAP Biophysical Processor — codifica
    la geometría sol-sensor explícitamente como inputs adicionales. Con la LUT
    generada por muestreo LHS de geometrías (SZA ∈ [20°,60°], VZA ∈ [0°,12°],
    RAA ∈ [0°,180°]), la red puede aprender la dependencia angular de la BRDF
    en lugar de marginalizarla implícitamente. Los cosenos en lugar de los
    ángulos directos son la convención porque son las cantidades que aparecen
    en las ecuaciones de transferencia radiativa (Lambert, fórmula del coseno).
    """
    n = spectra.shape[0]

    # Validar que la LUT trae geometría per-sample (es decir, fue generada
    # con muestreo LHS de ángulos, no con geometría fija)
    for k in ('tts', 'tto', 'psi'):
        if k not in params:
            raise KeyError(
                f"La LUT no contiene 'params[{k!r}]'. Esta versión de "
                f"train_gee_coefficients.py requiere una LUT generada con "
                f"el patch de geometría per-sample en prosail_pure.py. "
                f"Regenera la LUT desde el frontend (recomendado en modo "
                f"'Muestreada (rangos)')."
            )
        if np.asarray(params[k]).shape != (n,):
            raise ValueError(
                f"params[{k!r}] tiene shape {np.asarray(params[k]).shape}, "
                f"esperado ({n},)."
            )

    # Aviso si los ángulos son constantes (LUT generada en modo Fija):
    # el modelo entrenado sólo será válido para esa geometría exacta y
    # no podrá generalizar a otras escenas.
    n_unique = {k: len(np.unique(np.asarray(params[k]))) for k in ('tts','tto','psi')}
    if all(v == 1 for v in n_unique.values()):
        print("  ⚠ ATENCIÓN: la LUT tiene geometría FIJA (un solo valor de "
              "SZA/VZA/RAA).")
        print("    Los cosenos serán constantes y la red no podrá aprender "
              "dependencia angular.")
        print("    El modelo resultante sólo será válido para escenas con "
              f"SZA={params['tts'][0]:.1f}°, VZA={params['tto'][0]:.1f}°, "
              f"RAA={params['psi'][0]:.1f}°.")
        print("    RECOMENDADO: regenera la LUT con geometría muestreada "
              "(modo 'Muestreada' en el frontend).")

    # Normalizar reflectancias a media=0, std=1
    refl_mean = spectra.mean(axis=0)
    refl_std = spectra.std(axis=0)
    refl_std[refl_std < 1e-8] = 1e-8
    refl_norm = (spectra - refl_mean) / refl_std

    # Cosenos de los ángulos (grados → radianes → cos)
    # No se normalizan: están naturalmente acotados en [-1, 1] como los
    # índices espectrales que sustituyen.
    sza_rad = np.deg2rad(np.asarray(params['tts'], dtype=np.float64))
    vza_rad = np.deg2rad(np.asarray(params['tto'], dtype=np.float64))
    raa_rad = np.deg2rad(np.asarray(params['psi'], dtype=np.float64))

    cos_sza = np.cos(sza_rad).astype(np.float32)
    cos_vza = np.cos(vza_rad).astype(np.float32)
    cos_raa = np.cos(raa_rad).astype(np.float32)

    # Apilar: 8 reflectancias normalizadas + 3 cosenos
    X = np.column_stack([refl_norm, cos_sza, cos_vza, cos_raa])

    return X, refl_mean, refl_std


def compute_derived(params):
    """Calcula variables derivadas."""
    LAI = params['LAI']
    Cab = params['Cab']
    Cw = params['Cw']
    Cm = params['Cm']
    
    return {
        'fCover': 1.0 - np.exp(-0.5 * LAI),
        'CCC': LAI * Cab,
        'CWC': LAI * Cw * 10000,
    }


class SimpleNN:
    """
    Red neuronal de 1 capa oculta (misma arquitectura que SNAP).
    Input → Dense(n_hidden, tanh) → Dense(1, linear) → Output
    
    Entrenamiento con gradiente descendente + regularización L2.
    """
    
    def __init__(self, n_input, n_hidden=5, learning_rate=0.001,
                 l2_reg=0.001, n_epochs=500, batch_size=1024):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.l2 = l2_reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Inicialización Xavier
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(1)
        
        # Normalización de salida
        self.y_mean = 0.0
        self.y_std = 1.0
    
    def forward(self, X):
        """Forward pass."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2.ravel()
    
    def fit(self, X, y):
        """Entrenar con mini-batch SGD."""
        n = len(y)
        
        # Normalizar target
        self.y_mean = y.mean()
        self.y_std = y.std()
        if self.y_std < 1e-8:
            self.y_std = 1.0
        y_norm = (y - self.y_mean) / self.y_std
        
        best_loss = np.inf
        best_params = None
        patience = 50
        no_improve = 0
        
        for epoch in range(self.n_epochs):
            # Shuffle
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y_norm[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n, self.batch_size):
                Xb = X_shuf[i:i+self.batch_size]
                yb = y_shuf[i:i+self.batch_size]
                nb = len(yb)
                
                # Forward
                pred = self.forward(Xb)
                error = pred - yb
                loss = np.mean(error**2) + self.l2 * (
                    np.sum(self.W1**2) + np.sum(self.W2**2))
                
                # Backward
                d_z2 = error.reshape(-1, 1) / nb  # [nb, 1]
                d_W2 = self.a1.T @ d_z2 + 2 * self.l2 * self.W2
                d_b2 = d_z2.sum(axis=0)
                
                d_a1 = d_z2 @ self.W2.T  # [nb, n_hidden]
                d_z1 = d_a1 * (1 - self.a1**2)  # tanh derivative
                d_W1 = Xb.T @ d_z1 + 2 * self.l2 * self.W1
                d_b1 = d_z1.sum(axis=0)
                
                # Update
                self.W1 -= self.lr * d_W1
                self.b1 -= self.lr * d_b1
                self.W2 -= self.lr * d_W2
                self.b2 -= self.lr * d_b2
                
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = (self.W1.copy(), self.b1.copy(),
                               self.W2.copy(), self.b2.copy())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        # Restore best
        if best_params:
            self.W1, self.b1, self.W2, self.b2 = best_params
        
        return best_loss
    
    def predict(self, X):
        """Predicción desnormalizada."""
        return self.forward(X) * self.y_std + self.y_mean
    
    def get_weights_dict(self):
        """Devuelve pesos como diccionario serializable."""
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.ravel().tolist(),
            'b2': float(self.b2[0]),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            'n_hidden': self.n_hidden,
        }


def evaluate_model_extended(nn, X_train, y_train, X_test, y_test,
                            n_bootstrap=500, n_kfolds=5, n_bins=5,
                            random_state=42):
    """
    Evaluación estadística extendida de un modelo NN ya entrenado.

    Devuelve un dict con:
      - Métricas puntuales: R², RMSE, nRMSE, bias (ya calculadas antes)
      - IC95% bootstrap: R2_ci, RMSE_ci
      - Skill score vs predictor trivial (media del train): SS y p-valor bootstrap
      - Test t pareado sobre residuos: t_bias, p_bias
      - K-fold CV sobre train: R2_kfold_mean, R2_kfold_std
      - Estratificación por bines de y_test: RMSE_por_bin, n_por_bin
    """
    rng = np.random.default_rng(random_state)
    n_test = len(y_test)

    pred_test = nn.predict(X_test)
    residuos = pred_test - y_test
    rmse_obs = float(np.sqrt(np.mean(residuos ** 2)))
    r_obs    = float(np.corrcoef(pred_test, y_test)[0, 1])
    r2_obs   = r_obs ** 2
    bias_obs = float(np.mean(residuos))

    # ── (1) Bootstrap sobre el test set: IC95% de R² y RMSE ──
    r2_boot   = np.empty(n_bootstrap)
    rmse_boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx_b = rng.integers(0, n_test, n_test)
        pb, yb = pred_test[idx_b], y_test[idx_b]
        if np.std(pb) > 1e-10 and np.std(yb) > 1e-10:
            r2_boot[b]   = np.corrcoef(pb, yb)[0, 1] ** 2
        else:
            r2_boot[b] = np.nan
        rmse_boot[b] = np.sqrt(np.mean((pb - yb) ** 2))
    r2_ci   = np.nanpercentile(r2_boot,   [2.5, 97.5]).tolist()
    rmse_ci = np.nanpercentile(rmse_boot, [2.5, 97.5]).tolist()

    # ── (2) Skill score vs predictor trivial (media de y_train) ──
    y_train_mean = float(np.mean(y_train))
    rmse_trivial = float(np.sqrt(np.mean((y_train_mean - y_test) ** 2)))
    # SS = 1 - RMSE_modelo² / RMSE_trivial² (Nash-Sutcliffe). SS>0 implica mejor que trivial.
    ss = 1.0 - (rmse_obs ** 2) / (rmse_trivial ** 2 + 1e-12)
    # p-valor bootstrap: fracción de muestras donde el modelo NO supera al trivial
    better_count = 0
    for b in range(n_bootstrap):
        idx_b = rng.integers(0, n_test, n_test)
        rm_mod = np.sqrt(np.mean((pred_test[idx_b] - y_test[idx_b]) ** 2))
        rm_tri = np.sqrt(np.mean((y_train_mean - y_test[idx_b]) ** 2))
        if rm_mod < rm_tri:
            better_count += 1
    p_vs_trivial = 1.0 - (better_count / n_bootstrap)

    # ── (3) Test t pareado para el sesgo ──
    t_stat, p_bias = scipy_stats.ttest_1samp(residuos, 0.0)
    bias_se = float(np.std(residuos, ddof=1) / np.sqrt(n_test))
    bias_ci = [float(bias_obs - 1.96 * bias_se), float(bias_obs + 1.96 * bias_se)]

    # ── (4) K-fold CV sobre el train set ──
    # Usamos la misma arquitectura y config del nn ya entrenado
    r2_folds = []
    rmse_folds = []
    n_train = len(y_train)
    fold_size = n_train // n_kfolds
    perm = rng.permutation(n_train)
    for k in range(n_kfolds):
        val_idx = perm[k * fold_size:(k + 1) * fold_size]
        tr_idx  = np.setdiff1d(perm, val_idx)
        nn_k = SimpleNN(
            n_input=X_train.shape[1], n_hidden=nn.n_hidden,
            learning_rate=nn.lr, l2_reg=nn.l2,
            n_epochs=max(nn.n_epochs // 2, 200),
            batch_size=nn.batch_size
        )
        nn_k.fit(X_train[tr_idx], y_train[tr_idx])
        pk = nn_k.predict(X_train[val_idx])
        yk = y_train[val_idx]
        if np.std(pk) > 1e-10:
            r2_folds.append(np.corrcoef(pk, yk)[0, 1] ** 2)
        rmse_folds.append(np.sqrt(np.mean((pk - yk) ** 2)))
    r2_kfold_mean   = float(np.mean(r2_folds)) if r2_folds else np.nan
    r2_kfold_std    = float(np.std(r2_folds, ddof=1)) if len(r2_folds) > 1 else np.nan
    rmse_kfold_mean = float(np.mean(rmse_folds))

    # ── (5) RMSE estratificado por bines de y_test (detección de saturación) ──
    bin_edges = np.quantile(y_test, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-6  # para no perder el mínimo
    bin_idx = np.clip(np.digitize(y_test, bin_edges[1:-1], right=False), 0, n_bins - 1)
    rmse_bins = []
    bias_bins = []
    n_bins_actual = []
    rango_bins = []
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() >= 3:
            r_ = residuos[mask]
            rmse_bins.append(float(np.sqrt(np.mean(r_ ** 2))))
            bias_bins.append(float(np.mean(r_)))
            n_bins_actual.append(int(mask.sum()))
            rango_bins.append([float(y_test[mask].min()), float(y_test[mask].max())])
        else:
            rmse_bins.append(np.nan)
            bias_bins.append(np.nan)
            n_bins_actual.append(int(mask.sum()))
            rango_bins.append([np.nan, np.nan])

    return {
        # Ya calculadas en train_models, se devuelven para coherencia
        'R2':         round(r2_obs, 4),
        'RMSE':       round(rmse_obs, 4),
        'bias':       round(bias_obs, 4),
        # Bootstrap IC95%
        'R2_ci95':    [round(v, 4) for v in r2_ci],
        'RMSE_ci95':  [round(v, 4) for v in rmse_ci],
        'bias_ci95':  [round(v, 4) for v in bias_ci],
        # Skill vs predictor trivial
        'RMSE_trivial': round(rmse_trivial, 4),
        'skill_score':  round(ss, 4),
        'p_vs_trivial': round(p_vs_trivial, 4),
        # Bias test
        't_bias':     round(float(t_stat), 3),
        'p_bias':     float(f"{p_bias:.3g}"),
        # K-fold
        'R2_kfold_mean':   round(r2_kfold_mean, 4),
        'R2_kfold_std':    round(r2_kfold_std, 4),
        'RMSE_kfold_mean': round(rmse_kfold_mean, 4),
        # Bins
        'rmse_por_bin':   [round(v, 4) if not np.isnan(v) else None for v in rmse_bins],
        'bias_por_bin':   [round(v, 4) if not np.isnan(v) else None for v in bias_bins],
        'n_por_bin':      n_bins_actual,
        'rango_por_bin':  rango_bins,
    }


def interpretar_metricas_extendidas(var, m):
    """Emite un diagnóstico en lenguaje natural para el log."""
    msgs = []
    # R² dentro del IC
    msgs.append(f"    R² = {m['R2']:.4f}  IC95% = [{m['R2_ci95'][0]:.4f}, {m['R2_ci95'][1]:.4f}]")
    msgs.append(f"    RMSE = {m['RMSE']:.4f}  IC95% = [{m['RMSE_ci95'][0]:.4f}, {m['RMSE_ci95'][1]:.4f}]")
    # Skill
    ss_sig = "✓ supera al trivial" if m['p_vs_trivial'] < 0.05 else "⚠ NO supera claramente al trivial"
    msgs.append(f"    Skill score = {m['skill_score']:.3f}  ({ss_sig}, p = {m['p_vs_trivial']:.3f})")
    # Bias
    bias_sig = "significativo" if m['p_bias'] < 0.05 else "no significativo"
    msgs.append(f"    Bias = {m['bias']:+.4f}  IC95% = [{m['bias_ci95'][0]:+.4f}, {m['bias_ci95'][1]:+.4f}]  (t={m['t_bias']}, p={m['p_bias']:.3g}, {bias_sig})")
    # K-fold consistency
    cv_stable = m['R2_kfold_std'] < 0.05
    flag_cv = "✓ CV estable" if cv_stable else "⚠ alta varianza entre folds"
    msgs.append(f"    R² (5-fold CV) = {m['R2_kfold_mean']:.3f} ± {m['R2_kfold_std']:.3f}  ({flag_cv})")
    # Bin RMSE — detectar saturación
    rmse_bins = [v for v in m['rmse_por_bin'] if v is not None]
    if len(rmse_bins) >= 3:
        max_rmse, min_rmse = max(rmse_bins), min(rmse_bins)
        ratio = max_rmse / (min_rmse + 1e-12)
        if ratio > 2.0:
            # identificar bin con peor rmse
            idx_peor = m['rmse_por_bin'].index(max_rmse)
            rango = m['rango_por_bin'][idx_peor]
            msgs.append(f"    ⚠ Heterocedasticidad: RMSE por bin varía ×{ratio:.1f}. Peor bin: {var}∈[{rango[0]:.2f}, {rango[1]:.2f}] con RMSE={max_rmse:.3f}")
        else:
            msgs.append(f"    ✓ RMSE homogéneo entre bines (ratio max/min = {ratio:.2f})")
    return msgs


def train_models(X, params, derived, extended_eval=True,
                 n_hidden=5, n_epochs=800, learning_rate=0.005,
                 l2_reg=0.0005, batch_size=2048):
    """Entrena un modelo NN por cada variable biofísica."""
    models = {}
    metrics = {}

    all_targets = {}
    all_targets['LAI'] = params['LAI']
    all_targets['Cab'] = params['Cab']
    all_targets['fCover'] = derived['fCover']
    all_targets['CCC'] = derived['CCC']
    all_targets['CWC'] = derived['CWC']

    for var_name, y in all_targets.items():
        print(f"\n  Entrenando {var_name}...")
        y = np.asarray(y, dtype=np.float64)

        # Train/test split (80/20)
        n = len(y)
        idx = np.random.permutation(n)
        n_train = int(0.8 * n)

        X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
        y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

        # Entrenar NN
        nn = SimpleNN(
            n_input=X.shape[1],
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            n_epochs=n_epochs,
            batch_size=batch_size
        )
        final_loss = nn.fit(X_train, y_train)

        # Evaluación puntual (se conserva para compatibilidad con generate_js_code)
        pred_test = nn.predict(X_test)
        rmse = np.sqrt(np.mean((pred_test - y_test) ** 2))
        r = np.corrcoef(pred_test, y_test)[0, 1]
        bias = np.mean(pred_test - y_test)
        y_range = y_test.max() - y_test.min()
        nrmse = rmse / y_range * 100 if y_range > 0 else np.nan

        print(f"    R² = {r**2:.4f}  RMSE = {rmse:.4f}  "
              f"nRMSE = {nrmse:.1f}%  bias = {bias:.4f}")

        metrics[var_name] = {
            'R2':      round(float(r**2), 4),
            'RMSE':    round(float(rmse), 4),
            'nRMSE':   round(float(nrmse), 1),
            'bias':    round(float(bias), 4),
            'n_train': n_train,
            'n_test':  n - n_train,
        }

        # Evaluación extendida (IC, tests, k-fold, bins)
        if extended_eval:
            print(f"  Evaluación estadística extendida ({var_name})...")
            ext = evaluate_model_extended(nn, X_train, y_train, X_test, y_test)
            for k, v in ext.items():
                if k not in metrics[var_name]:
                    metrics[var_name][k] = v
            for line in interpretar_metricas_extendidas(var_name, ext):
                print(line)

        models[var_name] = nn

    return models, metrics


def generate_js_code(models, refl_mean, refl_std, band_names, metrics):
    """Genera código JavaScript listo para pegar en GEE."""
    
    n_bands = len(band_names)
    
    js = []
    js.append("// ════════════════════════════════════════════════════════════")
    js.append("// COEFICIENTES DE INVERSIÓN PROSAIL (NN 1 capa, tipo SNAP)")
    js.append("// Generados por train_gee_coefficients.py")
    js.append("//")
    js.append("// Features (11): 8 reflectancias S2 normalizadas (B3, B4, B5, B6, B7,")
    js.append("//                B8A, B11, B12) + cos(SZA) + cos(VZA) + cos(RAA)")
    js.append("// Geometría:     leída del metadata de cada imagen Sentinel-2 vía")
    js.append("//                MEAN_SOLAR_ZENITH_ANGLE, MEAN_INCIDENCE_*_ANGLE_<band>")
    js.append("// Esquema:       Weiss & Baret (2016) — SNAP S2 Biophysical Processor")
    js.append("// ════════════════════════════════════════════════════════════")
    js.append("")
    js.append("// Normalización de las reflectancias de entrada")
    js.append(f"var REFL_MEAN = {json.dumps([round(float(x), 6) for x in refl_mean])};")
    js.append(f"var REFL_STD  = {json.dumps([round(float(x), 6) for x in refl_std])};")
    js.append(f"var BAND_NAMES = {json.dumps(band_names)};")
    js.append("")
    
    # Métricas como comentario
    js.append("// Métricas de entrenamiento (test set 20%):")
    for var, m in metrics.items():
        js.append(f"//   {var}: R²={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  nRMSE={m['nRMSE']:.1f}%")
    js.append("")
    
    js.append("// Pesos de las redes neuronales")
    js.append("var NN_WEIGHTS = {};")
    
    for var_name, model in models.items():
        w = model.get_weights_dict()
        js.append(f"")
        js.append(f"NN_WEIGHTS.{var_name} = {{")
        
        # W1: [n_input × n_hidden] → formato JS
        js.append(f"  W1: {json.dumps([[round(x, 6) for x in row] for row in w['W1']])},")
        js.append(f"  b1: {json.dumps([round(x, 6) for x in w['b1']])},")
        js.append(f"  W2: {json.dumps([round(x, 6) for x in w['W2']])},")
        js.append(f"  b2: {round(w['b2'], 6)},")
        js.append(f"  y_mean: {round(w['y_mean'], 6)},")
        js.append(f"  y_std: {round(w['y_std'], 6)}")
        js.append(f"}};")
    
    js.append("")
    js.append("// ════════════════════════════════════════════════════════════")
    js.append("// FUNCIÓN DE INVERSIÓN (aplicar NN sobre imagen S2)")
    js.append("// ════════════════════════════════════════════════════════════")
    js.append("")
    js.append("/**")
    js.append(" * Aplica la red neuronal de inversión PROSAIL sobre una imagen S2.")
    js.append(f" * @param {{ee.Image}} s2img — Imagen con bandas {band_names}")
    js.append(" * @returns {ee.Image} — Imagen con bandas LAI, Cab, fCover, CCC, CWC")
    js.append(" */")
    js.append("function invertPROSAIL(s2img) {")
    js.append("")
    js.append("  // 1. Normalizar reflectancias")
    
    for i, bname in enumerate(band_names):
        js.append(f"  var n{bname} = s2img.select('{bname}')"
                  f".subtract({round(float(refl_mean[i]), 6)})"
                  f".divide({round(float(refl_std[i]), 6)});")
    
    js.append("")
    js.append("  // 2. Geometría sol-sensor desde metadata de la imagen S2")
    js.append("  //    SZA: solar zenith   VZA: view zenith (promedio de bandas)")
    js.append("  //    RAA: relative azimuth = |SAA - VAA| (doblado a [0,180])")
    js.append("  var DEG2RAD = Math.PI / 180;")
    js.append("  var sza_deg = ee.Number(s2img.get('MEAN_SOLAR_ZENITH_ANGLE'));")
    js.append("  var saa_deg = ee.Number(s2img.get('MEAN_SOLAR_AZIMUTH_ANGLE'));")
    js.append("")
    js.append("  // VZA y VAA: media de las bandas usadas en la inversión")
    js.append(f"  var _vzBands = {json.dumps(band_names)};")
    js.append("  var _vza_sum = ee.Number(0);")
    js.append("  var _vaa_sum = ee.Number(0);")
    js.append("  for (var _b = 0; _b < _vzBands.length; _b++) {")
    js.append("    _vza_sum = _vza_sum.add(ee.Number(s2img.get('MEAN_INCIDENCE_ZENITH_ANGLE_' + _vzBands[_b])));")
    js.append("    _vaa_sum = _vaa_sum.add(ee.Number(s2img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_' + _vzBands[_b])));")
    js.append("  }")
    js.append("  var vza_deg = _vza_sum.divide(_vzBands.length);")
    js.append("  var vaa_deg = _vaa_sum.divide(_vzBands.length);")
    js.append("")
    js.append("  // RAA = |SAA - VAA|, doblado al rango [0,180]")
    js.append("  var _raa_raw = saa_deg.subtract(vaa_deg).abs();")
    js.append("  var raa_deg = ee.Number(ee.Algorithms.If(")
    js.append("    _raa_raw.gt(180), ee.Number(360).subtract(_raa_raw), _raa_raw));")
    js.append("")
    js.append("  // Convertir a cosenos y elevar a imágenes constantes (broadcast)")
    js.append("  var cos_sza = ee.Image.constant(sza_deg.multiply(DEG2RAD).cos()).rename('cos_sza');")
    js.append("  var cos_vza = ee.Image.constant(vza_deg.multiply(DEG2RAD).cos()).rename('cos_vza');")
    js.append("  var cos_raa = ee.Image.constant(raa_deg.multiply(DEG2RAD).cos()).rename('cos_raa');")
    js.append("")
    js.append("  // 3. Apilar features: [nB3, nB4, ..., nB12, cos_sza, cos_vza, cos_raa]")
    feature_list = ', '.join([f'n{b}' for b in band_names] + ['cos_sza', 'cos_vza', 'cos_raa'])
    js.append(f"  var features = ee.Image.cat([{feature_list}]);")
    js.append("  var nFeats = features.bandNames().length();")
    js.append("")
    js.append("  // 4. Aplicar cada NN")
    js.append("  function applyNN(weights) {")
    js.append("    // Capa oculta: tanh(X · W1 + b1)")
    js.append("    var hidden = ee.Image(0);")
    js.append("    var hiddenBands = [];")
    js.append("    for (var h = 0; h < weights.W1[0].length; h++) {")
    js.append("      var neuron = ee.Image(weights.b1[h]);")
    js.append("      for (var f = 0; f < weights.W1.length; f++) {")
    js.append("        neuron = neuron.add(features.select(f).multiply(weights.W1[f][h]));")
    js.append("      }")
    js.append("      hiddenBands.push(neuron.tanh());")
    js.append("    }")
    js.append("    var hiddenImg = ee.Image.cat(hiddenBands);")
    js.append("")
    js.append("    // Capa de salida: linear(hidden · W2 + b2)")
    js.append("    var output = ee.Image(weights.b2);")
    js.append("    for (var h = 0; h < weights.W2.length; h++) {")
    js.append("      output = output.add(hiddenImg.select(h).multiply(weights.W2[h]));")
    js.append("    }")
    js.append("")
    js.append("    // Desnormalizar")
    js.append("    return output.multiply(weights.y_std).add(weights.y_mean);")
    js.append("  }")
    js.append("")
    js.append("  var LAI    = applyNN(NN_WEIGHTS.LAI).max(0).rename('LAI');")
    js.append("  var Cab    = applyNN(NN_WEIGHTS.Cab).max(0).rename('Cab');")
    js.append("  var fCover = applyNN(NN_WEIGHTS.fCover).clamp(0, 1).rename('fCover');")
    js.append("  var CCC    = applyNN(NN_WEIGHTS.CCC).max(0).rename('CCC');")
    js.append("  var CWC    = applyNN(NN_WEIGHTS.CWC).max(0).rename('CWC');")
    js.append("")
    js.append("  return ee.Image.cat([LAI, Cab, fCover, CCC, CWC]);")
    js.append("}")
    
    return '\n'.join(js)


def main():
    parser = argparse.ArgumentParser(
        description='Entrena modelos NN sobre LUT PROSAIL y exporta coeficientes JS para GEE')
    parser.add_argument('pkl_file', help='Ruta al fichero .pkl de la LUT')
    parser.add_argument('--output', '-o', default=None,
                        help='Nombre del fichero JS de salida')
    parser.add_argument('--n-hidden', type=int, default=5,
                        help='Neuronas en la capa oculta (default: 5, como SNAP)')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Épocas de entrenamiento (default: 800)')
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("  ENTRENAMIENTO DE MODELOS NN PARA INVERSIÓN PROSAIL EN GEE")
    print("  Arquitectura: Input → Dense(tanh) → Dense(linear) → Output")
    print("=" * 65)
    
    # Cargar LUT
    print(f"\n▸ Cargando LUT: {args.pkl_file}")
    data = load_lut(args.pkl_file)
    
    spectra = data['spectra']
    params = data['params']
    n_samples, n_bands = spectra.shape
    band_selection = data.get('band_selection', 'snap8')
    
    if band_selection == 'snap8' or n_bands == 8:
        band_names = SNAP8_NAMES
    else:
        band_names = S2_10_NAMES
    
    print(f"  Entradas LUT: {n_samples:,}")
    print(f"  Bandas: {n_bands} ({', '.join(band_names)})")
    
    # Variables derivadas
    derived = compute_derived(params)
    
    # Preparar features
    print(f"\n▸ Preparando features ({n_bands} bandas norm. + cos(SZA,VZA,RAA))...")
    X, refl_mean, refl_std = prepare_features(spectra, band_names, params)
    print(f"  Feature matrix: {X.shape}")
    # Diagnóstico de la geometría leída de la LUT
    sza_arr = np.asarray(params['tts'])
    vza_arr = np.asarray(params['tto'])
    raa_arr = np.asarray(params['psi'])
    print(f"  SZA muestreado: [{sza_arr.min():.1f}°, {sza_arr.max():.1f}°]  "
          f"cos∈[{np.cos(np.deg2rad(sza_arr.max())):.3f}, "
          f"{np.cos(np.deg2rad(sza_arr.min())):.3f}]")
    print(f"  VZA muestreado: [{vza_arr.min():.1f}°, {vza_arr.max():.1f}°]")
    print(f"  RAA muestreado: [{raa_arr.min():.1f}°, {raa_arr.max():.1f}°]")
    
    # Entrenar modelos
    np.random.seed(42)
    print(f"\n▸ Entrenando 5 modelos NN ({args.n_hidden} neuronas ocultas)...")
    models, metrics = train_models(X, params, derived,
                                   n_hidden=args.n_hidden,
                                   n_epochs=args.epochs)
    
    # Generar código JS
    print(f"\n▸ Generando código JavaScript...")
    js_code = generate_js_code(models, refl_mean, refl_std, band_names, metrics)
    
    # Guardar
    if args.output:
        js_path = args.output
    else:
        stem = Path(args.pkl_file).stem
        js_path = f"{stem}_nn_coefficients.js"
    
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(js_code)
    
    print(f"\n  Código JS exportado: {js_path}")
    print(f"  Tamaño: {os.path.getsize(js_path) / 1024:.1f} KB")
    
    # También exportar como JSON (para otros usos)
    json_path = js_path.replace('.js', '.json')
    json_data = {
        'refl_mean': [float(x) for x in refl_mean],
        'refl_std': [float(x) for x in refl_std],
        'band_names': band_names,
        'metrics': metrics,
        'models': {var: model.get_weights_dict() for var, model in models.items()},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON exportado: {json_path}")

    # ── Exportar métricas extendidas a CSV (una fila por variable) ──
    csv_metrics_path = js_path.replace('.js', '_metrics.csv')
    # Columnas en orden fijo para que pueda pegarse directamente al TFG
    cols = ['variable', 'n_train', 'n_test',
            'R2', 'R2_ci_lo', 'R2_ci_hi',
            'RMSE', 'RMSE_ci_lo', 'RMSE_ci_hi',
            'nRMSE_pct', 'bias', 'bias_ci_lo', 'bias_ci_hi',
            'bias_p_value', 'skill_score', 'p_vs_trivial',
            'R2_kfold_mean', 'R2_kfold_std']
    lines = [','.join(cols)]
    for var, m in metrics.items():
        row = [var, str(m.get('n_train', '')), str(m.get('n_test', '')),
               f"{m.get('R2', ''):.4f}",
               f"{m.get('R2_ci95', [np.nan, np.nan])[0]:.4f}",
               f"{m.get('R2_ci95', [np.nan, np.nan])[1]:.4f}",
               f"{m.get('RMSE', ''):.4f}",
               f"{m.get('RMSE_ci95', [np.nan, np.nan])[0]:.4f}",
               f"{m.get('RMSE_ci95', [np.nan, np.nan])[1]:.4f}",
               f"{m.get('nRMSE', ''):.1f}",
               f"{m.get('bias', ''):.4f}",
               f"{m.get('bias_ci95', [np.nan, np.nan])[0]:.4f}",
               f"{m.get('bias_ci95', [np.nan, np.nan])[1]:.4f}",
               f"{m.get('p_bias', np.nan):.3g}",
               f"{m.get('skill_score', np.nan):.4f}",
               f"{m.get('p_vs_trivial', np.nan):.4f}",
               f"{m.get('R2_kfold_mean', np.nan):.4f}",
               f"{m.get('R2_kfold_std', np.nan):.4f}"]
        lines.append(','.join(row))
    with open(csv_metrics_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Métricas extendidas (CSV): {csv_metrics_path}")

    # ── Exportar métricas estratificadas por bin a CSV (una fila por bin) ──
    csv_bins_path = js_path.replace('.js', '_metrics_bins.csv')
    bin_lines = ['variable,bin,y_min,y_max,n,RMSE,bias']
    for var, m in metrics.items():
        if 'rmse_por_bin' not in m:
            continue
        for b, (rmse_b, bias_b, n_b, rango_b) in enumerate(zip(
                m['rmse_por_bin'], m['bias_por_bin'],
                m['n_por_bin'], m['rango_por_bin'])):
            if rmse_b is None:
                continue
            ymin, ymax = rango_b if rango_b else (np.nan, np.nan)
            bin_lines.append(f"{var},{b+1},{ymin:.3f},{ymax:.3f},{n_b},{rmse_b:.4f},{bias_b:+.4f}")
    with open(csv_bins_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bin_lines) + '\n')
    print(f"  Métricas por bin (CSV):    {csv_bins_path}")

    # Resumen
    print(f"\n{'='*65}")
    print(f"  RESUMEN CON INTERVALOS DE CONFIANZA")
    print(f"{'='*65}")
    print(f"  Arquitectura: {X.shape[1]} → {args.n_hidden} (tanh) → 1 (linear)")
    header = f"  {'Var':>8s}  {'R² [IC95%]':>20s}  {'RMSE [IC95%]':>24s}  {'Skill':>6s}  {'p(bias)':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for var, m in metrics.items():
        r2_str = f"{m['R2']:.3f} [{m.get('R2_ci95',[0,0])[0]:.3f},{m.get('R2_ci95',[0,0])[1]:.3f}]"
        rmse_str = f"{m['RMSE']:.3f} [{m.get('RMSE_ci95',[0,0])[0]:.3f},{m.get('RMSE_ci95',[0,0])[1]:.3f}]"
        print(f"  {var:>8s}  {r2_str:>20s}  {rmse_str:>24s}  {m.get('skill_score',0):>6.2f}  {m.get('p_bias',1):>8.2g}")

    print(f"""
  USO EN GEE:
  ───────────
  1. Copiar el contenido de {js_path} al inicio del script GEE
  2. Llamar:  var biophys = invertPROSAIL(s2_median_image);
  3. Extraer: var LAI = biophys.select('LAI');

  La función invertPROSAIL() espera una imagen con bandas
  {band_names} en reflectancia [0,1].
""")

    print("✅ Entrenamiento completado.")


if __name__ == '__main__':
    main()
