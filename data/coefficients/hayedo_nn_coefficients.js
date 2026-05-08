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
var REFL_MEAN = [0.059215, 0.019767, 0.09013, 0.304113, 0.402524, 0.431793, 0.192452, 0.069908];
var REFL_STD  = [0.019764, 0.010716, 0.02615, 0.048153, 0.064602, 0.065452, 0.040282, 0.024219];
var BAND_NAMES = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"];

// Métricas de entrenamiento (test set 20%):
//   LAI: R²=0.488  RMSE=1.198  nRMSE=18.4%
//   Cab: R²=0.750  RMSE=6.878  nRMSE=13.8%
//   fCover: R²=0.624  RMSE=0.074  nRMSE=16.3%
//   CCC: R²=0.631  RMSE=62.924  nRMSE=11.4%
//   CWC: R²=0.739  RMSE=203.294  nRMSE=7.9%

// Pesos de las redes neuronales
var NN_WEIGHTS = {};

NN_WEIGHTS.LAI = {
  W1: [[0.36622, 0.055244, -0.115018, 0.107134, 0.462087], [-0.150668, 0.366705, -0.054699, -0.080289, -0.541015], [-0.322102, 0.68487, 0.164852, 0.003715, 0.943677], [0.053735, -0.499334, -0.381077, -0.552616, 0.197941], [-0.196724, -0.249381, 0.150369, 0.696223, 0.817363], [0.303471, 0.347027, 0.90468, 0.291652, -0.37173], [0.212594, -0.192242, -0.061668, 0.036159, 0.678447], [-0.011714, -0.058609, -0.398961, -0.059491, -0.150713], [-0.092905, -0.235909, -0.118465, 0.437008, 0.057594], [-0.233387, 0.039848, 0.344214, -0.380245, 0.167836], [0.207583, 0.706579, 0.002619, -0.567001, 0.266938]],
  b1: [-0.022576, 0.002337, 0.132359, -0.001893, -0.006685],
  W2: [0.318261, 0.06718, 1.064551, 0.275261, -0.028283],
  b2: -0.179615,
  y_mean: 4.176762,
  y_std: 1.68581
};

NN_WEIGHTS.Cab = {
  W1: [[0.490532, -0.100276, -0.227743, -0.655459, -0.4105], [0.028939, 0.105773, -0.019384, -0.35007, 0.195214], [0.074848, -0.734363, -0.363988, 0.550871, 0.063976], [-0.219957, -0.580539, -0.03757, -0.170666, 0.167922], [-0.385654, 0.795397, 0.351189, -0.380603, -0.252995], [0.411297, -0.274804, -0.072209, -0.150992, -0.753677], [-0.079469, -0.045599, 0.112307, 0.360304, -0.490861], [-0.632755, 0.150111, 0.007208, 0.257852, -0.09201], [0.082085, -0.001734, 0.48273, -0.233277, 0.336825], [0.167731, -0.363041, -0.049236, -0.377062, -0.312876], [0.616711, 0.016486, 0.053105, 0.256464, 0.12337]],
  b1: [0.004327, -0.002428, -0.104142, -0.004415, -0.018324],
  W2: [0.035577, 0.75164, 0.999171, 0.018446, 0.044168],
  b2: -0.030532,
  y_mean: 48.945481,
  y_std: 13.746789
};

NN_WEIGHTS.fCover = {
  W1: [[0.062931, 0.38887, -0.745753, 0.006385, 0.399266], [-0.52236, 0.189251, 0.259197, 0.104502, -0.38237], [0.001081, 0.134285, -0.386286, -0.036878, 0.290988], [0.513431, 0.457396, -0.638888, 0.404816, 0.034871], [0.812201, 0.359955, -0.466464, -0.20289, 0.30687], [0.26148, -0.008986, -0.521741, -1.023188, 0.169311], [0.916592, 0.1156, 0.507214, -0.001729, -0.422953], [0.008636, -0.964981, 0.531825, 0.458197, -0.196485], [-0.033383, 0.003107, 0.153492, -0.043491, 0.107039], [-0.383019, 0.790145, 0.356969, -0.476394, -0.211627], [0.239798, 0.040034, 0.200703, 0.044014, 0.154221]],
  b1: [0.009055, 0.019179, -0.024201, -0.323367, 0.014307],
  W2: [0.036488, -0.069296, -0.044448, -1.485729, 0.005724],
  b2: -0.685967,
  y_mean: 0.831419,
  y_std: 0.121858
};

NN_WEIGHTS.CCC = {
  W1: [[0.027172, -0.394029, 0.052915, 0.182918, 0.025391], [0.082624, 0.096942, -0.322441, -0.004292, 0.185563], [0.053072, 0.092788, 0.549768, 0.296839, 0.467675], [0.336906, -0.489721, 0.489175, 0.682149, -0.496778], [-0.203523, 0.465868, -0.395581, -0.428393, -0.27518], [-0.644732, 0.108024, -0.070317, -0.542708, -0.191189], [0.084028, -1.12301, -0.798633, -0.086133, 0.164131], [0.219111, -0.298242, -0.082316, 0.194197, 0.244747], [-0.401684, 0.279607, -0.115261, -0.119785, 0.122193], [0.333004, 0.504524, 0.049719, 0.439026, -0.27448], [-0.19286, -0.491681, -0.445249, 0.067676, 0.45908]],
  b1: [-0.2603, -0.008969, 0.043591, 0.327045, -0.247296],
  W2: [-0.583275, -0.014028, -0.143024, -0.893042, -0.289305],
  b2: 0.206761,
  y_mean: 204.324902,
  y_std: 103.043354
};

NN_WEIGHTS.CWC = {
  W1: [[0.002939, 0.012007, 0.084435, -0.047519, -0.119237], [-0.088518, -0.061498, 0.073842, -0.173276, -0.020533], [0.11885, -0.059695, 0.067046, 0.374898, -0.439524], [-0.075259, 0.263144, -0.384173, -0.613935, 0.244876], [0.16565, -0.127338, 0.087026, 0.32102, -0.474865], [-0.530149, 0.277194, 0.70657, 0.669869, -0.247525], [0.337237, -0.596631, -0.574426, 0.223448, -0.047322], [-0.109582, -0.15236, -0.120106, -0.619494, 0.320732], [-0.08538, 0.086978, -0.018017, 0.082829, -0.147466], [-0.276522, -0.546111, -0.73788, 0.296525, -0.188107], [-0.01168, 0.074282, -0.054913, -0.005808, -0.057506]],
  b1: [-0.111155, -0.310708, -0.040056, -0.385426, -0.011598],
  W2: [-0.547637, 0.650052, 0.872997, 0.487759, 0.037321],
  b2: 0.61323,
  y_mean: 731.612806,
  y_std: 397.965085
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
  var nB3 = s2img.select('B3').subtract(0.059215).divide(0.019764);
  var nB4 = s2img.select('B4').subtract(0.019767).divide(0.010716);
  var nB5 = s2img.select('B5').subtract(0.09013).divide(0.02615);
  var nB6 = s2img.select('B6').subtract(0.304113).divide(0.048153);
  var nB7 = s2img.select('B7').subtract(0.402524).divide(0.064602);
  var nB8A = s2img.select('B8A').subtract(0.431793).divide(0.065452);
  var nB11 = s2img.select('B11').subtract(0.192452).divide(0.040282);
  var nB12 = s2img.select('B12').subtract(0.069908).divide(0.024219);

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