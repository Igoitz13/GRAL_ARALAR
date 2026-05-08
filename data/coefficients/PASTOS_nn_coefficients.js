// ════════════════════════════════════════════════════════════
// COEFICIENTES DE INVERSIÓN PROSAIL (NN 1 capa, tipo SNAP)
// Generados por train_gee_coefficients.py
//
// Features (11): 8 reflectancias S2 normalizadas (B3, B4, B5, B6, B7,
//                B8A, B11, B12) + cos(SZA) + cos(VZA) + cos(RAA)
// Geometría:     leída del metadata de cada imagen Sentinel-2 vía
//                MEAN_SOLAR_ZENITH_ANGLE, MEAN_INCIDENCE_*_ANGLE_<band>
// Esquema:       Weiss & Baret (2016) — SNAP S2 Biophysical Processor
// ════════════════════════════════════════════════════════════

// Normalización de las reflectancias de entrada
var REFL_MEAN = [0.068401, 0.029039, 0.096226, 0.281939, 0.35534, 0.358721, 0.169329, 0.069995];
var REFL_STD  = [0.03807, 0.026222, 0.048063, 0.072942, 0.09711, 0.094242, 0.048026, 0.041502];
var BAND_NAMES = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"];

// Métricas de entrenamiento (test set 20%):
//   LAI: R²=0.705  RMSE=1.083  nRMSE=13.7%
//   Cab: R²=0.811  RMSE=8.467  nRMSE=11.3%
//   fCover: R²=0.882  RMSE=0.079  nRMSE=8.4%
//   CCC: R²=0.817  RMSE=50.615  nRMSE=8.2%
//   CWC: R²=0.847  RMSE=169.031  nRMSE=4.6%

// Pesos de las redes neuronales
var NN_WEIGHTS = {};

NN_WEIGHTS.LAI = {
  W1: [[0.399612, 0.056577, -0.142049, 0.192911, 0.412468], [-0.231772, 0.367658, -0.092307, -0.138161, -0.570995], [-0.30038, 0.667243, 0.245357, 0.093845, 0.885965], [0.121252, -0.494443, -0.282288, -0.471417, 0.206746], [-0.03255, -0.26721, 0.48761, 0.781467, 0.810887], [0.180017, 0.35108, 0.517725, 0.115651, -0.358281], [0.18905, -0.140487, -0.170865, -0.091913, 0.601448], [-0.006255, -0.023559, -0.503651, -0.094168, -0.193086], [-0.033085, -0.22273, -0.083876, 0.448308, 0.060335], [-0.229191, 0.067459, 0.14854, -0.423424, 0.167974], [0.309545, 0.642982, 0.095426, -0.426455, 0.258214]],
  b1: [-0.026965, 0.033756, -0.066187, -0.063969, -0.001519],
  W2: [0.340739, -0.060005, 0.945652, 0.422958, -0.076349],
  b2: 0.019184,
  y_mean: 3.580161,
  y_std: 2.001362
};

NN_WEIGHTS.Cab = {
  W1: [[0.949556, -0.169414, 0.098316, -0.19206, 0.083745], [-0.312206, 0.257579, -0.219124, 0.039733, -0.041416], [0.045125, 0.7831, 0.774516, -0.50884, -0.15839], [0.376916, 0.042097, 0.610382, -0.26792, 0.581383], [-0.157021, 0.374497, -0.203234, 0.550994, 0.034359], [-0.048705, -0.452397, -0.383202, 0.249487, 0.119636], [-0.230354, 0.168535, -0.071175, 0.13537, 0.367346], [-0.101274, 0.219627, -0.220584, 0.139735, 0.07289], [-0.135729, 0.481412, 0.165383, 0.620826, -0.004996], [0.489883, 0.794561, 0.068593, 0.119928, -0.212438], [0.084945, 0.087354, -0.188901, -0.009505, 0.350751]],
  b1: [0.267324, -0.104276, 0.240959, 0.089767, 0.00271],
  W2: [-0.434675, -0.291665, -0.691805, 0.689655, -0.100607],
  b2: 0.153217,
  y_mean: 43.566156,
  y_std: 19.442663
};

NN_WEIGHTS.fCover = {
  W1: [[0.263204, 0.192863, -0.008151, 0.023981, 0.169806], [-0.563267, -0.760969, -0.181879, 0.128867, 0.152749], [0.11781, -0.060972, -0.258899, -0.195708, 0.128589], [-0.124378, -0.275177, 0.064026, 0.084009, -0.039411], [0.370074, 0.569875, 0.372757, 0.273453, -0.567079], [0.350204, 0.208888, -0.398451, 0.344515, 0.426178], [-0.282933, -0.147863, 0.179644, -0.177546, -0.389408], [-0.22017, -0.082575, 0.249952, -0.124292, 0.123466], [0.203455, 0.271043, 0.543373, 0.839695, -0.346089], [0.554709, 0.183537, 0.176215, -0.080259, -0.151079], [0.052948, -0.395832, -0.190185, 0.11462, 0.263421]],
  b1: [-0.03687, -0.025146, 0.395719, 0.264935, -0.027133],
  W2: [0.435148, 0.258804, -1.059842, 0.887675, -0.661989],
  b2: -0.245451,
  y_mean: 0.741174,
  y_std: 0.229636
};

NN_WEIGHTS.CCC = {
  W1: [[0.476657, -0.290327, -0.180029, 0.021705, 0.256028], [0.224549, -0.24845, -0.306817, -0.123675, -0.012191], [0.367998, 0.201574, -0.072719, -0.451945, 0.456641], [0.587677, 0.0193, 0.865045, 0.064103, 0.031433], [0.124359, 0.253965, -0.304283, 0.817187, -0.675033], [-1.053459, 0.365682, -0.07555, -0.18697, 0.435184], [0.005148, 0.264945, -0.235538, -0.179341, 0.322182], [0.260934, -0.576013, -0.116114, -0.14219, -0.066728], [0.271147, -0.039154, -0.893633, 0.177901, -0.129844], [0.736744, -0.056297, -0.152044, -0.195125, 0.043901], [-0.049579, -0.571946, 0.026821, 0.15758, 0.692081]],
  b1: [0.410192, 0.08484, 0.117483, -0.170501, 0.114837],
  W2: [-0.892956, 0.233818, -0.588943, 0.912238, -0.038021],
  b2: 0.322976,
  y_mean: 155.617879,
  y_std: 117.977319
};

NN_WEIGHTS.CWC = {
  W1: [[0.452908, 0.645614, 0.11287, -0.242507, 0.121551], [-0.100816, -0.232788, -0.395917, 0.226471, -0.043578], [0.12613, -0.856732, -0.877016, -0.67553, -0.032544], [-0.255137, -0.212072, -0.256144, 0.190965, -0.096373], [0.507227, -0.249453, 0.022589, 0.301826, 0.429784], [0.163542, 0.127476, 0.164345, 0.137529, 0.27571], [-0.284024, 0.550622, 0.091606, 0.311027, -0.611635], [-0.167966, -0.669346, -0.410973, -0.618263, -0.289889], [0.52133, 0.057674, -0.269505, -0.172838, 0.003683], [0.064311, -0.43893, 0.144234, 0.189601, -0.745574], [-0.366737, -0.150755, -0.278685, -0.074895, 0.023844]],
  b1: [0.081947, -0.067074, 0.085634, 0.120328, -0.268388],
  W2: [0.227134, -0.237048, 0.035401, 0.233531, 1.788331],
  b2: 0.786544,
  y_mean: 593.701437,
  y_std: 431.531365
};

// ════════════════════════════════════════════════════════════
// FUNCIÓN DE INVERSIÓN (aplicar NN sobre imagen S2)
// ════════════════════════════════════════════════════════════

/**
 * Aplica la red neuronal de inversión PROSAIL sobre una imagen S2.
 * @param {ee.Image} s2img — Imagen con bandas ['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
 * @returns {ee.Image} — Imagen con bandas LAI, Cab, fCover, CCC, CWC
 */
function invertPROSAIL(s2img) {

  // 1. Normalizar reflectancias
  var nB3 = s2img.select('B3').subtract(0.068401).divide(0.03807);
  var nB4 = s2img.select('B4').subtract(0.029039).divide(0.026222);
  var nB5 = s2img.select('B5').subtract(0.096226).divide(0.048063);
  var nB6 = s2img.select('B6').subtract(0.281939).divide(0.072942);
  var nB7 = s2img.select('B7').subtract(0.35534).divide(0.09711);
  var nB8A = s2img.select('B8A').subtract(0.358721).divide(0.094242);
  var nB11 = s2img.select('B11').subtract(0.169329).divide(0.048026);
  var nB12 = s2img.select('B12').subtract(0.069995).divide(0.041502);

  // 2. Geometría sol-sensor desde metadata de la imagen S2
  //    SZA: solar zenith   VZA: view zenith (promedio de bandas)
  //    RAA: relative azimuth = |SAA - VAA| (doblado a [0,180])
  var DEG2RAD = Math.PI / 180;
  var sza_deg = ee.Number(s2img.get('MEAN_SOLAR_ZENITH_ANGLE'));
  var saa_deg = ee.Number(s2img.get('MEAN_SOLAR_AZIMUTH_ANGLE'));

  // VZA y VAA: media de las bandas usadas en la inversión
  var _vzBands = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"];
  var _vza_sum = ee.Number(0);
  var _vaa_sum = ee.Number(0);
  for (var _b = 0; _b < _vzBands.length; _b++) {
    _vza_sum = _vza_sum.add(ee.Number(s2img.get('MEAN_INCIDENCE_ZENITH_ANGLE_' + _vzBands[_b])));
    _vaa_sum = _vaa_sum.add(ee.Number(s2img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_' + _vzBands[_b])));
  }
  var vza_deg = _vza_sum.divide(_vzBands.length);
  var vaa_deg = _vaa_sum.divide(_vzBands.length);

  // RAA = |SAA - VAA|, doblado al rango [0,180]
  var _raa_raw = saa_deg.subtract(vaa_deg).abs();
  var raa_deg = ee.Number(ee.Algorithms.If(
    _raa_raw.gt(180), ee.Number(360).subtract(_raa_raw), _raa_raw));

  // Convertir a cosenos y elevar a imágenes constantes (broadcast)
  var cos_sza = ee.Image.constant(sza_deg.multiply(DEG2RAD).cos()).rename('cos_sza');
  var cos_vza = ee.Image.constant(vza_deg.multiply(DEG2RAD).cos()).rename('cos_vza');
  var cos_raa = ee.Image.constant(raa_deg.multiply(DEG2RAD).cos()).rename('cos_raa');

  // 3. Apilar features: [nB3, nB4, ..., nB12, cos_sza, cos_vza, cos_raa]
  var features = ee.Image.cat([nB3, nB4, nB5, nB6, nB7, nB8A, nB11, nB12, cos_sza, cos_vza, cos_raa]);
  var nFeats = features.bandNames().length();

  // 4. Aplicar cada NN
  function applyNN(weights) {
    // Capa oculta: tanh(X · W1 + b1)
    var hidden = ee.Image(0);
    var hiddenBands = [];
    for (var h = 0; h < weights.W1[0].length; h++) {
      var neuron = ee.Image(weights.b1[h]);
      for (var f = 0; f < weights.W1.length; f++) {
        neuron = neuron.add(features.select(f).multiply(weights.W1[f][h]));
      }
      hiddenBands.push(neuron.tanh());
    }
    var hiddenImg = ee.Image.cat(hiddenBands);

    // Capa de salida: linear(hidden · W2 + b2)
    var output = ee.Image(weights.b2);
    for (var h = 0; h < weights.W2.length; h++) {
      output = output.add(hiddenImg.select(h).multiply(weights.W2[h]));
    }

    // Desnormalizar
    return output.multiply(weights.y_std).add(weights.y_mean);
  }

  var LAI    = applyNN(NN_WEIGHTS.LAI).max(0).rename('LAI');
  var Cab    = applyNN(NN_WEIGHTS.Cab).max(0).rename('Cab');
  var fCover = applyNN(NN_WEIGHTS.fCover).clamp(0, 1).rename('fCover');
  var CCC    = applyNN(NN_WEIGHTS.CCC).max(0).rename('CCC');
  var CWC    = applyNN(NN_WEIGHTS.CWC).max(0).rename('CWC');

  return ee.Image.cat([LAI, Cab, fCover, CCC, CWC]);
}