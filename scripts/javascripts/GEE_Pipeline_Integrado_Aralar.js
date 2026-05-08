// ================================================================
// PIPELINE INTEGRADO v3: PROSAIL-NN + ERA5-Land — Sierra de Aralar
// ZEC ES2120011 + HIC habitats + distancia a bordas
// Google Earth Engine
// ================================================================
//
// ZONIFICACIÓN:
//   Límite:  Polígono oficial ZEC / Parque Natural de Aralar
//   Hábitat: Cartografía HIC (Gobierno Vasco) rasterizada
//   Presión: Distancia euclidiana a bordas/majadas (3 anillos)
//   Zonas:   HIC_pastoral × anillo_distancia → código compuesto
//
// EXPORTS (3 independientes, join en R):
//   1. LAI zonal por escena S2
//   2. Clima ERA5-Land diario (serie continua)
//   3. ERA5-Land mensual 1981–presente (para SPEI)
//   + Validación MODIS + GeoTIFFs
//
// ================================================================


// ╔════════════════════════════════════════════════════════════════╗
// ║  0. CONFIGURACIÓN                                             ║
// ╚════════════════════════════════════════════════════════════════╝

var CONFIG = {
  years: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  mesInicio: 4,
  mesFin: 10,
  maxCloudPct: 25,
  Tbase: 5,
  scale: 20,
  era5scale: 11132,
  modisScale: 500,
  folder: 'GEE_Aralar_Pipeline',
  crs: 'EPSG:25830',
  
  // ┌──────────────────────────────────────────────────────────┐
  // │  ASSETS — EDITAR CON TUS RUTAS                           │
  // └──────────────────────────────────────────────────────────┘
   assets: {
    zec:    'projects/ee-letrak/assets/LIMITE_ARALAR',        // Límite ZEC (polígono)
    hic:    'projects/ee-letrak/assets/HIC_ARALAR',        // Hábitats HIC (polígonos con campo TIPO)
    bordas: 'projects/ee-letrak/assets/BORDAS_ARALAR',      // Bordas/majadas (puntos)
    mdt:    'projects/ee-letrak/assets/MDT_ARALAR_05M'     // MDT 5m (filtro altitudinal pastoreo estival)
   },

   // ── Filtro altitudinal del pastoreo estival (aplicado SOLO a TIPO=1, pasto) ──
   // El régimen de trashumancia vertical del sistema sasi en Aralar opera
   // efectivamente entre 700 y 1400 m. Por debajo: prados de fondo de valle
   // con gestión distinta. Por encima: superficies marginales con temporada
   // recortada por nieve persistente. El hayedo (TIPO=3) NO se filtra por
   // altitud porque sirve como referencia climática sin pastoreo.
   altPastoMin: 700,
   altPastoMax: 1400,
  
  // Anillos de distancia a bordas (metros)
  // Cercano: máxima presión ganadera
  // Intermedio: presión media
  // Remoto: mínima presión (pseudo-referencia)
  distRings: [500, 1000, 1500],  // umbrales: <500m, 500-1500m, 1000-1500m, >1500m
  
  // Paletas
  pal: {
    lai:     ['#ffffcc','#c2e699','#78c679','#31a354','#006837'],
    hic:     ['#b2df8a','#33a02c','#a6cee3','#1f78b4','#fb9a99',
              '#e31a1c','#fdbf6f','#ff7f00'],
    anillo:  ['#d73027','#fee08b','#1a9850'],  // cercano, medio, remoto
    zona:    ['#e41a1c','#ff7f00','#ffd700','#377eb8','#41ae76',
              '#984ea3','#a65628','#f781bf','#999999','#66c2a5',
              '#fc8d62','#8da0cb']
  }
};



// ╔════════════════════════════════════════════════════════════════╗
// ║  1. COEFICIENTES NN — PROSAIL DUAL (PASTO + HAYEDO)           ║
// ║                                                                ║
// ║  Dos modelos independientes con la misma arquitectura         ║
// ║  (11 features, 1 capa oculta tanh, 5 neuronas).               ║
// ║  - PASTO:  LUT Aralar — Pastos atlánticos HIC 6230*           ║
// ║  - HAYEDO: LUT Aralar — Hayedo HIC 9120/9150                   ║
// ║  Cada modelo se aplica solo a los píxeles de su tipo HIC.     ║
// ╚════════════════════════════════════════════════════════════════╝

var BAND_NAMES = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"];

// ─────────────────────────── MODELO PASTO ───────────────────────────
var REFL_MEAN_PASTO = [0.067984, 0.034397, 0.101759, 0.285153, 0.371599, 0.403881, 0.209006, 0.093786];
var REFL_STD_PASTO  = [0.037287, 0.036451, 0.050562, 0.07452, 0.096939, 0.102151, 0.077258, 0.065281];

// Métricas de entrenamiento (test set 20%):

var NN_WEIGHTS_PASTO = {};

NN_WEIGHTS_PASTO.LAI = {
  W1: [[0.397648, 0.052246, -0.105543, 0.124215, 0.430244], [-0.34444, 0.404414, -0.367712, -0.189712, -0.553369], [-0.295841, 0.662375, 0.304785, 0.078566, 0.91767], [0.007726, -0.470672, -0.522196, -0.597114, 0.1906], [-0.186179, -0.233756, 0.223976, 0.640357, 0.794669], [0.254502, 0.341894, 0.845924, 0.165808, -0.379126], [0.182744, -0.153597, -0.25935, 0.034043, 0.637438], [-0.010373, -0.039102, -0.562221, 0.060149, -0.14832], [0.021259, -0.231764, -0.003378, 0.492203, 0.08333], [-0.226349, 0.053104, 0.113751, -0.369632, 0.203087], [0.229829, 0.682611, 0.009593, -0.528639, 0.259046]],
  b1: [-0.020773, 0.017777, -0.101438, -0.000783, 0.033773],
  W2: [0.380975, 0.034688, 1.162109, 0.228515, -0.085929],
  b2: -0.0538,
  y_mean: 3.611106,
  y_std: 1.978065
};

NN_WEIGHTS_PASTO.Cab = {
  W1: [[-0.172613, -0.00124, -0.489844, 0.411581, 0.207921], [0.341769, 0.203489, 0.520759, 0.070081, -0.106998], [0.027049, -0.713397, -0.689471, -0.632723, -0.281458], [-0.207011, -0.068438, -0.440335, 0.201769, -0.115846], [-0.472996, 0.69872, 0.461269, -0.434009, -0.405637], [0.84119, -0.151677, -0.076386, -0.731488, -0.050534], [0.261511, -0.096602, 0.242437, -0.537093, -0.332716], [-0.401361, 0.330605, 0.230474, 0.225075, -0.381911], [0.09948, 0.435217, -0.253783, 0.240618, 0.272034], [0.733434, -0.111704, 0.180971, 0.057411, 0.018634], [0.340716, 0.025692, 0.031605, -0.525804, -0.027463]],
  b1: [0.106573, 0.119901, -0.28962, 0.017251, 0.036746],
  W2: [-0.027158, 0.96374, 0.935455, -0.013713, 0.134068],
  b2: -0.131646,
  y_mean: 45.003902,
  y_std: 18.439339
};

NN_WEIGHTS_PASTO.fCover = {
  W1: [[-0.088658, 0.569811, 0.006667, -0.288845, 0.352819], [0.00702, 0.03142, -0.640624, -0.050173, -0.095884], [-0.002134, 0.110293, 0.128313, 0.066274, 0.118519], [-0.157485, -0.447563, -0.247142, 0.32801, 0.496341], [-0.017632, -0.220463, 0.25077, -0.080723, -0.479701], [0.690679, 0.566608, 0.52624, -0.148929, 0.806897], [0.106981, 0.17891, -0.350532, -0.013882, 0.022476], [-0.421862, -0.330334, -0.264271, 0.047033, -0.274435], [0.63894, -0.30488, 0.037377, -0.381431, -0.033826], [0.202425, -0.12728, 0.141946, -0.623163, 0.102911], [-0.030044, 0.030723, -0.059608, 0.207204, -0.137658]],
  b1: [0.342032, -0.127891, 0.098404, -0.128257, 0.183337],
  W2: [1.263875, 0.286141, 0.905437, 0.210316, -0.220418],
  b2: -0.781171,
  y_mean: 0.748495,
  y_std: 0.219305
};

NN_WEIGHTS_PASTO.CCC = {
  W1: [[-0.044534, -0.204724, 0.485039, -0.09051, 0.217729], [0.188457, 0.372644, 0.210492, 0.081592, -0.230679], [0.239677, 0.086959, 0.607278, -0.438387, -0.151817], [-0.629687, 0.066564, 0.577241, 0.368582, -0.007722], [-0.077876, -0.332254, -0.950235, -0.047126, -0.471495], [0.158187, -0.173299, -0.056117, 0.166947, -0.021742], [0.042303, -0.064317, 0.096165, -0.033729, 0.157193], [0.200984, 0.523148, 0.10889, 0.023391, -0.374957], [-0.255677, -0.283462, 0.204296, 0.564012, 0.870066], [0.084701, -0.067375, 0.63711, -0.109246, -0.006983], [-0.143916, -0.075525, -0.053616, 0.076474, 0.790272]],
  b1: [0.045313, 0.369974, 0.430533, -0.030802, 0.02924],
  W2: [1.156558, -1.112376, -0.926273, 1.025456, 0.009704],
  b2: 0.290835,
  y_mean: 162.389314,
  y_std: 117.087622
};

NN_WEIGHTS_PASTO.CWC = {
  W1: [[0.074663, -0.20518, -0.096982, -0.46342, -0.191711], [-0.116023, -0.032436, 0.139173, 0.431033, -0.313818], [-0.15325, -0.058616, -0.077304, -0.217346, 0.397318], [0.226721, -0.04857, 0.250123, -0.036403, 0.601025], [-0.411805, 0.153394, 0.095608, 0.658914, -0.095908], [-0.279034, -0.737622, -0.784479, 0.18886, 0.70288], [0.34988, 0.219884, 0.94463, -0.502499, 0.501145], [0.282456, 0.388244, 0.42372, 0.2042, 0.437874], [-0.187467, 0.413539, -0.083644, 0.91038, 0.156629], [0.137669, 0.503127, 0.78051, -0.441964, 0.04272], [-0.112644, 0.310651, -0.038405, -0.149522, -0.1239]],
  b1: [0.045813, -0.151461, 0.590456, 0.047775, -0.003857],
  W2: [-0.413512, -0.237591, -1.617158, 0.177431, -7.7e-05],
  b2: 1.087048,
  y_mean: 600.155814,
  y_std: 428.524791
};


// ─────────────────────────── MODELO HAYEDO ──────────────────────────
var REFL_MEAN_HAYEDO = [0.059187, 0.019737, 0.090083, 0.30414, 0.402587, 0.431897, 0.192452, 0.069843];
var REFL_STD_HAYEDO  = [0.0198, 0.010686, 0.026195, 0.048453, 0.064906, 0.065761, 0.040282, 0.024244];

// Métricas de entrenamiento (test set 20%):

var NN_WEIGHTS_HAYEDO = {};

NN_WEIGHTS_HAYEDO.LAI = {
  W1: [[0.311232, 0.050837, -0.090592, 0.102759, 0.438156], [-0.073687, 0.34882, -0.094137, -0.029353, -0.513229], [-0.261077, 0.653507, 0.11578, 0.024509, 0.893171], [0.044533, -0.469263, -0.316148, -0.579592, 0.188964], [-0.167149, -0.230964, 0.095576, 0.594364, 0.778704], [0.367483, 0.344512, 0.908404, 0.275916, -0.353115], [0.214234, -0.175287, -0.047882, 0.013569, 0.645005], [0.00864, -0.04318, -0.40968, -0.057863, -0.144163], [-0.111267, -0.216339, -0.096327, 0.378536, 0.057312], [-0.232679, 0.048714, 0.349549, -0.397141, 0.163305], [0.190599, 0.668026, -0.003142, -0.530384, 0.253047]],
  b1: [-0.033534, 0.013364, 0.155903, -0.03919, -0.003012],
  W2: [0.311, 0.088016, 1.086188, 0.2998, -0.027903],
  b2: -0.189368,
  y_mean: 4.175644,
  y_std: 1.684576
};

NN_WEIGHTS_HAYEDO.Cab = {
  W1: [[-0.153012, -0.200926, -0.396734, 0.209273, -0.128475], [0.088144, -0.390937, -0.252178, -0.287682, -0.519997], [-0.576155, 0.042139, -0.166584, -0.226874, 0.040144], [-0.333089, 0.434347, 0.05642, 0.030845, 0.100623], [0.681837, 0.082972, 0.139671, -0.25866, 0.022902], [-0.31569, -0.297865, 0.158481, 0.465816, -0.108398], [0.068776, 0.49125, 0.08034, 0.314183, -0.34665], [0.042802, -0.039371, 0.024548, 0.034594, 0.148764], [0.259605, -0.200493, -0.203477, 0.502107, 0.017902], [-0.170609, -0.012245, 0.302049, -0.289351, -0.108453], [0.022679, 0.225883, 0.371517, -0.455309, 0.543962]],
  b1: [-0.140577, -0.010041, -0.000938, 0.002865, -0.002071],
  W2: [1.308065, -0.127618, 0.375785, 0.097577, -0.013684],
  b2: 0.007489,
  y_mean: 48.927083,
  y_std: 13.740878
};

NN_WEIGHTS_HAYEDO.fCover = {
  W1: [[0.045698, -0.446309, -0.228192, -0.00886, 0.332009], [0.101648, 0.20657, 0.080979, 0.069071, 0.202816], [-0.059769, 0.348539, -0.642493, 0.099869, -0.446158], [0.35736, 0.136094, -0.32896, 0.270177, 0.563248], [-0.272716, 0.162402, 0.103685, -0.593135, 1.258668], [-0.860487, -0.539686, 0.022722, 1.118641, 0.153149], [-0.029858, 0.554597, -0.53541, 0.195858, -0.195925], [0.45964, -0.482311, -0.331694, 0.142417, 0.369263], [-0.27621, 0.645827, -0.740824, 0.359185, 0.254766], [-0.510172, -0.313199, 0.046727, -0.668094, 0.043727], [0.018013, 0.446926, 0.137226, 0.238636, 0.267836]],
  b1: [-0.105414, 0.041735, 0.116687, -0.028412, 0.085545],
  W2: [-1.506622, -0.127141, 0.02758, 0.173228, -0.063301],
  b2: -0.677608,
  y_mean: 0.831346,
  y_std: 0.121685
};

NN_WEIGHTS_HAYEDO.CCC = {
  W1: [[0.013814, -0.212241, 0.401223, -0.043962, -0.130607], [0.155551, 0.057534, -0.313256, -0.012916, 0.270223], [0.034116, -0.282029, 0.245372, -0.252958, 0.334378], [0.102468, 0.18889, -0.177792, -0.742696, 0.237877], [-0.338361, -0.253632, -0.583791, 0.653775, -0.450348], [-0.461722, 0.601237, 0.110831, 0.247622, 0.481267], [-0.02635, -0.193815, -0.237096, 0.128779, 0.90866], [0.440499, 0.100407, -0.12012, -0.177142, 0.354599], [-0.667865, -0.101671, 0.436513, -0.096862, -0.450015], [0.071691, -0.27108, 0.200781, -0.641695, -0.050269], [-0.120176, -0.092384, 0.424213, -0.001594, -0.329968]],
  b1: [-0.032878, 0.312209, -0.093879, 0.187803, -0.031058],
  W2: [-0.500276, 0.440773, 0.018429, 1.086702, 0.049095],
  b2: 0.259585,
  y_mean: 204.591254,
  y_std: 102.840859
};

NN_WEIGHTS_HAYEDO.CWC = {
  W1: [[-0.09185, -0.021291, -0.038012, -0.008903, -0.10694], [0.189727, -0.154298, -0.004661, -0.345445, -0.002272], [-0.491707, 1.031056, 0.048272, 0.623791, -0.068242], [0.202288, -1.011684, -0.104036, 0.623318, 0.301451], [-0.154316, -0.531022, -0.289468, -0.274397, -0.400079], [0.081092, 0.082517, 0.922002, -0.644841, -0.239641], [0.22169, 0.321017, -0.446289, -0.27967, 0.358375], [0.188019, -0.215024, -0.01178, -0.10791, 0.623354], [-0.249762, 0.08407, 0.116348, 0.029779, 0.323845], [-0.12988, -0.180884, -0.095349, 0.125118, 0.364034], [-0.322245, -0.26956, -0.039158, 0.775251, -0.013812]],
  b1: [0.075507, -0.049659, -0.274844, -0.03222, 0.346984],
  W2: [-0.36269, -0.112436, 1.12896, -0.173422, -0.763824],
  b2: 0.60579,
  y_mean: 732.482992,
  y_std: 397.326436
};


// ╔════════════════════════════════════════════════════════════════╗
// ║  2. FUNCIONES DE INVERSIÓN                                    ║
// ║                                                                ║
// ║  invertPROSAILCon(s2img, REFL_MEAN, REFL_STD, NN_WEIGHTS):    ║
// ║    función genérica parametrizable; aplica una NN concreta.   ║
// ║                                                                ║
// ║  invertPROSAIL(s2img):                                         ║
// ║    wrapper que aplica los DOS modelos según hicTipo y los     ║
// ║    combina en una sola imagen. Añade banda 'modelo_inv'       ║
// ║    (1=pasto, 3=hayedo) para trazabilidad.                     ║
// ╚════════════════════════════════════════════════════════════════╝

function invertPROSAILCon(s2img, REFL_MEAN, REFL_STD, NN_WEIGHTS) {
  // 1. Normalización de reflectancias
  var normBands = [];
  for (var i = 0; i < BAND_NAMES.length; i++) {
    normBands.push(s2img.select(BAND_NAMES[i])
        .subtract(REFL_MEAN[i]).divide(REFL_STD[i]));
  }

  // 2. Geometría sol-sensor desde metadata S2
  var DEG2RAD = Math.PI / 180;
  var sza_deg = ee.Number(s2img.get('MEAN_SOLAR_ZENITH_ANGLE'));
  var saa_deg = ee.Number(s2img.get('MEAN_SOLAR_AZIMUTH_ANGLE'));

  var _vza_sum = ee.Number(0);
  var _vaa_sum = ee.Number(0);
  for (var _b = 0; _b < BAND_NAMES.length; _b++) {
    _vza_sum = _vza_sum.add(ee.Number(s2img.get('MEAN_INCIDENCE_ZENITH_ANGLE_' + BAND_NAMES[_b])));
    _vaa_sum = _vaa_sum.add(ee.Number(s2img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_' + BAND_NAMES[_b])));
  }
  var vza_deg = _vza_sum.divide(BAND_NAMES.length);
  var vaa_deg = _vaa_sum.divide(BAND_NAMES.length);

  var _raa_raw = saa_deg.subtract(vaa_deg).abs();
  var raa_deg = ee.Number(ee.Algorithms.If(
    _raa_raw.gt(180), ee.Number(360).subtract(_raa_raw), _raa_raw));

  var cos_sza = ee.Image.constant(sza_deg.multiply(DEG2RAD).cos()).rename('cos_sza');
  var cos_vza = ee.Image.constant(vza_deg.multiply(DEG2RAD).cos()).rename('cos_vza');
  var cos_raa = ee.Image.constant(raa_deg.multiply(DEG2RAD).cos()).rename('cos_raa');

  // 3. Feature matrix: 8 reflectancias norm + 3 cosenos = 11 features
  var features = ee.Image.cat(normBands.concat([cos_sza, cos_vza, cos_raa]));

  function applyNN(weights) {
    var hiddenBands = [];
    for (var h = 0; h < weights.W1[0].length; h++) {
      var neuron = ee.Image(weights.b1[h]);
      for (var f = 0; f < weights.W1.length; f++) {
        neuron = neuron.add(features.select(f).multiply(weights.W1[f][h]));
      }
      hiddenBands.push(neuron.tanh());
    }
    var hiddenImg = ee.Image.cat(hiddenBands);
    var output = ee.Image(weights.b2);
    for (var h = 0; h < weights.W2.length; h++) {
      output = output.add(hiddenImg.select(h).multiply(weights.W2[h]));
    }
    return output.multiply(weights.y_std).add(weights.y_mean);
  }

  return ee.Image.cat([
    applyNN(NN_WEIGHTS.LAI).max(0).rename('LAI'),
    applyNN(NN_WEIGHTS.Cab).max(0).rename('Cab'),
    applyNN(NN_WEIGHTS.fCover).clamp(0, 1).rename('fCover'),
    applyNN(NN_WEIGHTS.CCC).max(0).rename('CCC'),
    applyNN(NN_WEIGHTS.CWC).max(0).rename('CWC')
  ]);
}

function invertPROSAIL(s2img) {
  // Máscaras por tipo HIC (hicTipo se define en la sección 4 del pipeline)
  var mask_pasto  = hicTipo.eq(1);  // HIC 6230* y afines
  var mask_hayedo = hicTipo.eq(3);  // HIC 9120/9150

  // Invertir cada modelo sobre la imagen COMPLETA (sin enmascarar previamente).
  // Esto evita que updateMask upstream rompa el cómputo de features angulares
  // (cos_sza es ee.Image.constant() sin máscara; su combinación con
  // ee.Image.cat de bandas enmascaradas puede producir imágenes vacías según
  // el orden de evaluación de GEE).
  var biophys_pasto  = invertPROSAILCon(s2img, REFL_MEAN_PASTO,
                                        REFL_STD_PASTO,
                                        NN_WEIGHTS_PASTO);
  var biophys_hayedo = invertPROSAILCon(s2img, REFL_MEAN_HAYEDO,
                                        REFL_STD_HAYEDO,
                                        NN_WEIGHTS_HAYEDO);

  // Combinación atómica: donde mask_hayedo sea verdadera, sustituir TODAS las
  // bandas de biophys_pasto por las de biophys_hayedo de una sola vez.
  // `.where()` aplicado a una imagen multibanda sustituye banda a banda
  // manteniendo nombres y orden. Es más robusto que blend() o mosaic().
  var combined = biophys_pasto.where(mask_hayedo, biophys_hayedo);

  // Enmascarar: solo píxeles de pasto o hayedo tienen valor válido
  combined = combined.updateMask(mask_pasto.or(mask_hayedo));

  // Banda de trazabilidad: 1=pasto, 3=hayedo, 0=sin modelo (enmascarado)
  var modelo = mask_pasto.multiply(1)
                .add(mask_hayedo.multiply(3))
                .rename('modelo_inv');

  return combined.addBands(modelo);
}



function maskS2(image) {
  var scl = image.select('SCL');
  return image.updateMask(scl.eq(4).or(scl.eq(5)))
              .divide(10000)
              .copyProperties(image, image.propertyNames());
}


// ╔════════════════════════════════════════════════════════════════╗
// ║  3. AOI: LÍMITE ZEC / PARQUE NATURAL                          ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== CARGANDO ASSETS ===');

var zecFC = ee.FeatureCollection(CONFIG.assets.zec);
var aralar = zecFC.geometry();
Map.centerObject(aralar, 12);
Map.addLayer(ee.Image().paint(aralar, 0, 2), {palette: ['yellow']},
             'Limite ZEC Aralar', true, 0.6);

print('AOI:', CONFIG.assets.zec);
print('Area ZEC (ha):', aralar.area().divide(10000).round());


// ╔════════════════════════════════════════════════════════════════╗
// ║  4. ZONIFICACIÓN: HIC × DISTANCIA A BORDAS                   ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== ZONIFICACION: HIC x distancia bordas ===');

// ── 4a. Cartografía HIC ──
// El asset debe tener un campo numérico TIPO:
//   1 = pasto (6170, 6210, 6230*, 6510)
//   2 = brezal (4030, 4090)
//   3 = hayedo (9120)
//   4 = encinar (9340)
// Y un campo HIC1 con el código original

var hicFC = ee.FeatureCollection(CONFIG.assets.hic);

// Rasterizar por TIPO
var hicTipo = hicFC.reduceToImage({
  properties: ['TIPO'],
  reducer: ee.Reducer.first()
}).rename('hic_tipo').clip(aralar).toInt();

// ╔════════════════════════════════════════════════════════════════╗
// ║  FILTRO ALTITUDINAL DEL PASTOREO ESTIVAL (NUEVO v2)           ║
// ║                                                                ║
// ║  Aplicado SOLO al pasto (TIPO=1): se eliminan los píxeles     ║
// ║  fuera del rango altPastoMin..altPastoMax (por defecto        ║
// ║  700-1400 m, régimen sasi de trashumancia vertical).          ║
// ║                                                                ║
// ║  El hayedo (TIPO=3) se deja intacto: cubre todo el rango      ║
// ║  altitudinal porque sirve como referencia climática sin       ║
// ║  pastoreo (Estrategia D del IPP).                             ║
// ║                                                                ║
// ║  Brezales (TIPO=2) y encinares (TIPO=4) tampoco se filtran    ║
// ║  porque no entran en el cálculo de IPP en esta versión.       ║
// ╚════════════════════════════════════════════════════════════════╝
var mdtAralar = ee.Image(CONFIG.assets.mdt).clip(aralar);

var maskAltitudPasto = mdtAralar.gte(CONFIG.altPastoMin)
                                .and(mdtAralar.lte(CONFIG.altPastoMax));

// Donde hicTipo==1 y el píxel está fuera del rango altitudinal, se pone a 0
// (sin hábitat asignado) para que se propague el filtro a todo lo demás
var fueraDeRangoPasto = hicTipo.eq(1).and(maskAltitudPasto.not());
var hicTipoOriginal = hicTipo;  // Guardamos para diagnóstico de superficies
hicTipo = hicTipo.where(fueraDeRangoPasto, 0);

// Diagnóstico en consola
print('=== FILTRO ALTITUDINAL PASTO (' + CONFIG.altPastoMin + '-' + CONFIG.altPastoMax + ' m) ===');
var nPastoOrig = hicTipoOriginal.eq(1).selfMask().reduceRegion({
  reducer: ee.Reducer.count(), geometry: aralar,
  scale: CONFIG.scale, maxPixels: 1e9}).get('hic_tipo');
var nPastoFiltrado = hicTipo.eq(1).selfMask().reduceRegion({
  reducer: ee.Reducer.count(), geometry: aralar,
  scale: CONFIG.scale, maxPixels: 1e9}).get('hic_tipo');
print('Pixeles pasto antes del filtro altitudinal:', nPastoOrig);
print('Pixeles pasto despues del filtro altitudinal:', nPastoFiltrado);

// Visualización opcional de los píxeles pasto descartados por altitud
Map.addLayer(fueraDeRangoPasto.selfMask(),
  {palette: ['#ff0000']},
  'Pasto excluido por altitud (<' + CONFIG.altPastoMin + ' o >' + CONFIG.altPastoMax + ' m)', false);

// Rasterizar por código HIC numérico (para detalle)
// Convertir HIC1 texto a numérico limpiando asteriscos
var hicCode = hicFC.map(function(f) {
  var raw = ee.String(f.get('HIC1'));
  // Eliminar asterisco: "6230*" → "6230"
  var clean = raw.replace('[^0-9]', '', 'g');
  return f.set('HIC_NUM', ee.Number.parse(clean));
}).reduceToImage({
  properties: ['HIC_NUM'],
  reducer: ee.Reducer.first()
}).rename('hic_code').clip(aralar).toInt();

// Máscaras por tipo de hábitat
var pastoMask  = hicTipo.eq(1);  // 6170, 6210, 6230*, 6510
var brezalMask = hicTipo.eq(2);  // 4030, 4090
var hayedoMask = hicTipo.eq(3);  // 9120
var encinarMask = hicTipo.eq(4); // 9340

// Superficie por tipo (los pixeles pasto ya reflejan el filtro altitudinal)
print('=== SUPERFICIES POR TIPO HIC (post-filtro altitudinal en pasto) ===');
print('Pixeles pasto (HIC TIPO=1, filtrado 700-1400m):', pastoMask.selfMask().reduceRegion({
  reducer: ee.Reducer.count(), geometry: aralar,
  scale: CONFIG.scale, maxPixels: 1e9}).get('hic_tipo'));
print('Pixeles hayedo (HIC TIPO=3, sin filtrar):', hayedoMask.selfMask().reduceRegion({
  reducer: ee.Reducer.count(), geometry: aralar,
  scale: CONFIG.scale, maxPixels: 1e9}).get('hic_tipo'));

Map.addLayer(hicTipo.selfMask(), {min: 1, max: 4, palette: ['#33a02c','#e31a1c','#1f78b4','#ff7f00']},
             'HIC tipos (1=pasto 2=brezal 3=hayedo 4=encinar)', false);

// ── 4b. Distancia a bordas ──
var bordasFC = ee.FeatureCollection(CONFIG.assets.bordas);
print('Bordas cargadas:', bordasFC.size());

// Distancia euclidiana a la borda más cercana (metros)
var distBorda = bordasFC.distance(5000).clip(aralar).rename('dist_borda');

// Clasificar en 3 anillos
var anillo = ee.Image(1)  // 1 = cercano (<500m)
    .where(distBorda.gte(CONFIG.distRings[0]).and(distBorda.lt(CONFIG.distRings[1])), 2)  // intermedio
    .where(distBorda.gte(CONFIG.distRings[1]), 3)  // remoto
    .rename('anillo');

Map.addLayer(distBorda, {min: 0, max: 3000, palette: ['red','yellow','green']},
             'Distancia a bordas (m)', false);
Map.addLayer(anillo.updateMask(pastoMask), {min: 1, max: 3, palette: CONFIG.pal.anillo},
             'Anillos presion (pasto)', false);

// ── 4c. Zona compuesta: HIC × anillo ──
// Codificación: TIPO * 10 + anillo
//   11 = pasto cercano     12 = pasto intermedio   13 = pasto remoto
//   21 = brezal cercano    22 = brezal intermedio  23 = brezal remoto
//   31 = hayedo cercano    32 = hayedo intermedio  33 = hayedo remoto
//   41 = encinar cercano   42 = encinar intermedio 43 = encinar remoto

var zonaImg = hicTipo.multiply(10).add(anillo)
    .rename('zona')
    .clip(aralar)
    .updateMask(hicTipo.gt(0));  // Solo píxeles con HIC asignado

// Tabla de nombres de zona para el panel CSV
var ZONA_NAMES = {
  11: 'pasto_cercano',    12: 'pasto_intermedio',   13: 'pasto_remoto',
  21: 'brezal_cercano',   22: 'brezal_intermedio',  23: 'brezal_remoto',
  31: 'hayedo_cercano',   32: 'hayedo_intermedio',  33: 'hayedo_remoto',
  41: 'encinar_cercano',  42: 'encinar_intermedio', 43: 'encinar_remoto'
};

// Lista de IDs de zona que realmente existen (para el export)
var ZONA_IDS = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43];

Map.addLayer(zonaImg.updateMask(pastoMask),
  {min: 11, max: 13, palette: CONFIG.pal.anillo},
  'Zonas pastorales (HIC x dist)', true);

// ── 4d. Auxiliares: DEM y pendiente ──
// Se usa el MDT5 local (mismo asset del filtro altitudinal) para asegurar
// consistencia entre filtro y productos derivados (pendiente, exposición)
var dem = mdtAralar;
var slope = ee.Terrain.slope(dem);

// Estadísticas por zona → se exportan en zonas_caracterizacion (sección 11)
print('Zonificacion completada. Detalle en export zonas_caracterizacion.');


// ╔════════════════════════════════════════════════════════════════╗
// ║  5. INVERSIÓN PROSAIL-NN MULTITEMPORAL                        ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== MODULO A: PROSAIL-NN ===');

var yearStart = Math.min.apply(null, CONFIG.years);
var yearEnd   = Math.max.apply(null, CONFIG.years);

var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aralar)
    .filterDate(yearStart + '-01-01', (yearEnd + 1) + '-01-01')
    .filter(ee.Filter.calendarRange(CONFIG.mesInicio, CONFIG.mesFin, 'month'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
    .map(maskS2);

print('Escenas S2:', s2col.size());

// Aplicar inversión a cada escena
var s2inv = s2col.select(BAND_NAMES).map(function(img) {
  return invertPROSAIL(img).clip(aralar)
         .updateMask(hicTipo.gt(0))  // Solo sobre hábitats HIC
         .copyProperties(img, ['system:time_start',
           'MEAN_SOLAR_ZENITH_ANGLE', 'CLOUDY_PIXEL_PERCENTAGE']);
});

print('Escenas invertidas:', s2inv.size());



// ╔════════════════════════════════════════════════════════════════╗
// ║  6. EXPORT 1 — LAI ZONAL POR ESCENA                          ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== EXPORT 1: LAI zonal + pixel count ===');

var laiZonal = s2inv.map(function(img) {
  var laiWithZona = img.select('LAI').addBands(zonaImg);
  
  // Media de LAI por zona
  var groupedMean = laiWithZona.reduceRegion({
    reducer: ee.Reducer.mean().group({groupField: 1, groupName: 'zona'}),
    geometry: aralar,
    scale: CONFIG.scale,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 4         // reparte el cálculo en más tiles paralelos
  });
  
  // Conteo de píxeles válidos por zona
  var groupedCount = laiWithZona.reduceRegion({
    reducer: ee.Reducer.count().group({groupField: 1, groupName: 'zona'}),
    geometry: aralar,
    scale: CONFIG.scale,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 4
  });
  
  // Protección contra escenas sin píxeles válidos:
  // Si la escena está completamente nublada sobre el AOI, 'groups' no existe
  var rawMean  = groupedMean.get('groups');
  var rawCount = groupedCount.get('groups');
  var groupsMean  = ee.List(ee.Algorithms.If(rawMean,  rawMean,  ee.List([])));
  var groupsCount = ee.List(ee.Algorithms.If(rawCount, rawCount, ee.List([])));
  
  // Propiedades base
  var props = ee.Dictionary({
    'date': img.date().format('YYYY-MM-dd'),
    'year': img.date().get('year'),
    'doy':  img.date().getRelative('day', 'year'),
    'cloud_pct': img.get('CLOUDY_PIXEL_PERCENTAGE')
  });
  
  // Aplanar LAI medias (null-safe con combine para defaults)
  // ee.Dictionary.combine() inyecta el default SIN evaluar .get() condicional
  var propsWithLAI = groupsMean.iterate(function(g, acc) {
    g = ee.Dictionary(g).combine(ee.Dictionary({mean: -9999}), false);
    var zId = ee.Number(g.get('zona')).int();
    var zMean = g.get('mean');
    return ee.Dictionary(acc).set(ee.String('LAI_z').cat(zId.format('%d')), zMean);
  }, props);
  
  // Aplanar conteos (null-safe con combine)
  var propsWithAll = groupsCount.iterate(function(g, acc) {
    g = ee.Dictionary(g).combine(ee.Dictionary({count: 0}), false);
    var zId = ee.Number(g.get('zona')).int();
    var zCount = g.get('count');
    return ee.Dictionary(acc).set(ee.String('N_z').cat(zId.format('%d')), zCount);
  }, propsWithLAI);
  
  return ee.Feature(null, ee.Dictionary(propsWithAll));
});

// Construir lista de selectores
var laiSelectors = ['date', 'year', 'doy', 'cloud_pct'];
ZONA_IDS.forEach(function(zid) {
  laiSelectors.push('LAI_z' + zid);
});
ZONA_IDS.forEach(function(zid) {
  laiSelectors.push('N_z' + zid);
});

Export.table.toDrive({
  collection: laiZonal,
  description: 'LAI_zonal_HIC_aralar',
  folder: CONFIG.folder,
  fileFormat: 'CSV',
  selectors: laiSelectors
});


// ╔════════════════════════════════════════════════════════════════╗
// ║  7. EXPORT 2 — CLIMA ERA5-Land DIARIO                        ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== EXPORT 2: Clima diario ===');

var climDailyFeats = ee.FeatureCollection([]);

CONFIG.years.forEach(function(yr) {
  var start = yr + '-0' + CONFIG.mesInicio + '-01';
  var end   = yr + '-' + (CONFIG.mesFin + 1) + '-01';
  
  var era5d = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
      .filterDate(start, end)
      .filterBounds(aralar);
  
  var dailyFeats = era5d.map(function(img) {
    var d = img.date();
    var P_mm = img.select('total_precipitation_sum').multiply(1000);
    var T_C = img.select('temperature_2m').subtract(273.15);
    var SM1 = img.select('volumetric_soil_water_layer_1');
    var SM2 = img.select('volumetric_soil_water_layer_2');
    var SM_root = SM1.multiply(7).add(SM2.multiply(21)).divide(28);
    // Radiación solar descendente (J/m² → MJ/m²/día)
    var Rad = img.select('surface_solar_radiation_downwards_sum')
                 .divide(1e6);
    
    var stats = ee.Image.cat([
      P_mm.rename('P_mm'), T_C.rename('T2m_C'),
      SM1.rename('SM_sup'), SM_root.rename('SM_root'),
      Rad.rename('Rad_MJ')
    ]).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: aralar,
      scale: CONFIG.era5scale, bestEffort: true
    });
    
    return ee.Feature(null, stats)
        .set('date', d.format('YYYY-MM-dd'))
        .set('year', d.get('year'))
        .set('doy', d.getRelative('day', 'year'));
  });
  
  climDailyFeats = climDailyFeats.merge(dailyFeats);
});

Export.table.toDrive({
  collection: climDailyFeats,
  description: 'clima_diario_aralar',
  folder: CONFIG.folder,
  fileFormat: 'CSV',
  selectors: ['date', 'year', 'doy', 'P_mm', 'T2m_C', 'SM_sup', 'SM_root', 'Rad_MJ']
});


// ╔════════════════════════════════════════════════════════════════╗
// ║  8. EXPORT 3 — ERA5-Land MENSUAL 1981–presente (SPEI)        ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== EXPORT 3: ERA5 mensual (serie larga) ===');

var era5m = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    .filterDate('1981-01-01', '2026-01-01')
    .filterBounds(aralar);

var monthlyFeats = era5m.map(function(img) {
  var d = img.date();
  var stats = img.select([
    'temperature_2m', 'total_precipitation_sum',
    'volumetric_soil_water_layer_1'
  ]).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: aralar,
    scale: CONFIG.era5scale, bestEffort: true
  });
  
  return ee.Feature(null, {
    'date':    d.format('YYYY-MM-dd'),
    'year':    d.get('year'),
    'month':   d.get('month'),
    'Tmean_K': stats.get('temperature_2m'),
    'P_m':     stats.get('total_precipitation_sum'),
    'SM_sup':  stats.get('volumetric_soil_water_layer_1')
  });
});

Export.table.toDrive({
  collection: monthlyFeats,
  description: 'ERA5Land_monthly_1981_present',
  folder: CONFIG.folder,
  fileFormat: 'CSV',
  selectors: ['date', 'year', 'month', 'Tmean_K', 'P_m', 'SM_sup']
});


// ╔════════════════════════════════════════════════════════════════╗
// ║  9. VALIDACIÓN MODIS                                          ║
// ╚════════════════════════════════════════════════════════════════╝

print('=== Validacion MODIS ===');

var lai2024 = s2inv.filter(ee.Filter.calendarRange(2024, 2024, 'year'))
    .select('LAI').median().clip(aralar);

var modis = ee.ImageCollection('MODIS/061/MCD15A3H')
    .filterBounds(aralar)
    .filterDate('2024-06-01', '2024-09-30')
    .map(function(img) {
      return img.updateMask(img.select('FparLai_QC').bitwiseAnd(1).eq(0))
                .copyProperties(img, ['system:time_start']);
    });

var modisLAI = modis.select('Lai').median().multiply(0.1).clip(aralar).rename('MODIS_LAI');

var s2proj = ee.Projection('EPSG:32630').atScale(CONFIG.scale);
var prosailLAI_500 = lai2024
    .setDefaultProjection(s2proj)
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
    .reproject({crs: modisLAI.projection(), scale: CONFIG.modisScale});

var valSamples = prosailLAI_500.rename('PROSAIL_LAI').addBands(modisLAI)
    .sample({region: aralar, scale: CONFIG.modisScale,
             numPixels: 400, seed: 42, geometries: true})
    .filter(ee.Filter.notNull(['PROSAIL_LAI', 'MODIS_LAI']));

print('Puntos validacion:', valSamples.size());

print(ui.Chart.feature.byFeature(valSamples, 'PROSAIL_LAI', 'MODIS_LAI')
    .setChartType('ScatterChart')
    .setOptions({
      title: 'LAI: PROSAIL-NN vs MODIS — Aralar 2024',
      hAxis: {title: 'LAI PROSAIL'}, vAxis: {title: 'LAI MODIS'},
      pointSize: 4, dataOpacity: 0.5,
      trendlines: {0: {type: 'linear', color: 'red', showR2: true}}
    }));

print('Pearson r:', valSamples.reduceColumns({
  reducer: ee.Reducer.pearsonsCorrelation(),
  selectors: ['PROSAIL_LAI', 'MODIS_LAI']
}).get('correlation'));

Export.table.toDrive({
  collection: valSamples,
  description: 'Validacion_PROSAIL_vs_MODIS',
  folder: CONFIG.folder, fileFormat: 'CSV'
});


// ╔════════════════════════════════════════════════════════════════╗
// ║  10. GeoTIFFs Y VISUALIZACIÓN                                 ║
// ╚════════════════════════════════════════════════════════════════╝

CONFIG.years.forEach(function(yr) {
  var yrInv = s2inv.filter(ee.Filter.calendarRange(yr, yr, 'year'));
  
  // Composito mediano (para cartografía descriptiva)
  var med = yrInv.select(['LAI', 'Cab', 'fCover', 'CCC']).median().clip(aralar);
  Export.image.toDrive({
    image: med.toFloat().unmask(-9999),
    description: 'PROSAIL_NN_Aralar_' + yr,
    folder: CONFIG.folder,
    region: aralar, scale: CONFIG.scale,
    crs: CONFIG.crs, maxPixels: 1e9,
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    }
  });
  
  // Composito máximo LAI (para normalización pixel a pixel)
  var laiMax = yrInv.select('LAI').max().clip(aralar);
  Export.image.toDrive({
    image: laiMax.toFloat().unmask(-9999),
    description: 'PROSAIL_NN_Aralar_LAImax_' + yr,
    folder: CONFIG.folder,
    region: aralar, scale: CONFIG.scale,
    crs: CONFIG.crs, maxPixels: 1e9,
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    }
  });
});

// Composito 2024 para visualización
Map.addLayer(lai2024, {min: 0, max: 4, palette: CONFIG.pal.lai},
             'LAI PROSAIL 2024 (mediana)', true);
Map.addLayer(modisLAI, {min: 0, max: 4, palette: CONFIG.pal.lai},
             'MODIS LAI 2024', false);
Map.addLayer(dem, {min: 400, max: 1500, palette: ['green','yellow','brown']},
             'DEM SRTM', false);

// Series temporales por año
CONFIG.years.forEach(function(yr) {
  print(ui.Chart.image.series({
    imageCollection: s2inv.filter(ee.Filter.calendarRange(yr, yr, 'year')).select('LAI'),
    region: aralar, reducer: ee.Reducer.mean(),
    scale: CONFIG.scale, xProperty: 'system:time_start'
  }).setOptions({
    title: 'LAI medio Aralar ' + yr,
    vAxis: {title: 'LAI'}, lineWidth: 2, pointSize: 4, colors: ['#006837']
  }));
});


// ╔════════════════════════════════════════════════════════════════╗
// ║  10b. IPP ESPACIALIZADO — ESTRATEGIA A (corrección climática) ║
// ║  Requiere: gam_predicciones_fecha como asset (CSV de R)       ║
// ╚════════════════════════════════════════════════════════════════╝
//
// INSTRUCCIONES:
// 1. Ejecutar el script R para obtener gam_predicciones_fecha.csv
// 2. Subir a GEE como asset:
//    earthengine upload table --asset_id users/TU_USUARIO/gam_preds gam_predicciones_fecha.csv
// 3. Descomentar el bloque siguiente y editar la ruta del asset


/*var GAM_PREDS_ASSET = 'projects/ee-letrak/assets/gam_predicciones_fecha';  // ← EDITAR

print('=== IPP ESPACIALIZADO — Estrategia A ===');
var gamPreds = ee.FeatureCollection(GAM_PREDS_ASSET);
print('Predicciones GAM cargadas:', gamPreds.size());

CONFIG.years.forEach(function(yr) {
  var yrInv = s2inv.filter(ee.Filter.calendarRange(yr, yr, 'year'));
  
  // LAI máximo del año para normalizar
  var laiMaxYr = yrInv.select('LAI').max().clip(aralar);
  // Proteger divisiones por cero
  var laiMaxSafe = laiMaxYr.where(laiMaxYr.lte(0), 1);
  
  // Para cada escena: LAI_norm - predicción GAM
  var residuos = yrInv.map(function(img) {
    var dateStr = img.date().format('YYYY-MM-dd');
    
    // Buscar la predicción GAM para esta fecha
    var match = gamPreds.filter(ee.Filter.eq('date', dateStr));
    var nMatch = match.size();
    
    // Si no hay match para esta fecha, devolver imagen vacía
    return ee.Algorithms.If(nMatch.gt(0),
      // Hay match: calcular residuo
      (function() {
        var pred = ee.Number.parse(match.first().get('LAI_norm_pred'));
        var laiNorm = img.select('LAI').divide(laiMaxSafe);
        return laiNorm.subtract(pred).rename('IPP_A')
            .copyProperties(img, ['system:time_start']);
      })(),
      // No hay match: null (se filtrará)
      null
    );
  }, true);  // dropNulls = true
  
  var residuosIC = ee.ImageCollection(residuos);
  
  // Verificar que hay escenas
  var nRes = residuosIC.size();
  print('IPP_A ' + yr + ': escenas con match GAM =', nRes);
  
  // Media de residuos = IPP_A anual espacializado
  var ippA = residuosIC.mean().clip(aralar)
      .updateMask(hicTipo.gt(0));  // Solo sobre hábitats HIC
  
  Export.image.toDrive({
    image: ippA.toFloat().unmask(-9999),
    description: 'IPP_estrategia_A_' + yr,
    folder: CONFIG.folder,
    region: aralar, scale: CONFIG.scale,
    crs: CONFIG.crs, maxPixels: 1e9,
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    }
  });
  
  // Visualizar último año
  if (yr === CONFIG.years[CONFIG.years.length - 1]) {
    Map.addLayer(ippA, {min: -0.3, max: 0.3, palette: ['#d73027','#fee08b','#1a9850']},
                 'IPP Estrategia A ' + yr, true);
  }
});

print('IPP Estrategia A: exports lanzados');
*/


// ╔════════════════════════════════════════════════════════════════╗
// ║  11. EXPORT AUXILIAR: Distancia media a borda por zona        ║
// ╚════════════════════════════════════════════════════════════════╝

// Exportar tabla de caracterización de zonas (para R)
var zonaChars = zonaImg.addBands(dem.rename('alt'))
    .addBands(slope.rename('pend'))
    .addBands(distBorda);

var charFeats = ee.FeatureCollection(ZONA_IDS.map(function(zid) {
  var mask = zonaImg.eq(zid);
  var stats = zonaChars.updateMask(mask).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: aralar,
    scale: 30,
    maxPixels: 1e8
  });
  var count = zonaImg.updateMask(mask).reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: aralar,
    scale: CONFIG.scale,
    maxPixels: 1e8
  });
  return ee.Feature(null, {
    'zona_id': zid,
    'zona_name': ZONA_NAMES[zid] || 'unknown',
    'alt_mean': stats.get('alt'),
    'pend_mean': stats.get('pend'),
    'dist_borda_mean': stats.get('dist_borda'),
    'n_pixels': count.get('zona')
  });
}));

Export.table.toDrive({
  collection: charFeats,
  description: 'zonas_caracterizacion',
  folder: CONFIG.folder,
  fileFormat: 'CSV',
  selectors: ['zona_id', 'zona_name', 'alt_mean', 'pend_mean', 'dist_borda_mean', 'n_pixels']
});


// ╔════════════════════════════════════════════════════════════════╗
// ║  12. INTERFAZ                                                 ║
// ╚════════════════════════════════════════════════════════════════╝

var panel = ui.Panel({
  style: {position: 'bottom-left', padding: '8px 12px',
          backgroundColor: 'rgba(255,255,255,0.92)'}
});
panel.add(ui.Label({
  value: '🏔 Pipeline v3 — Aralar ZEC',
  style: {fontWeight: 'bold', fontSize: '14px'}
}));
panel.add(ui.Label('PROSAIL-NN | HIC habitats | Dist. bordas'));
panel.add(ui.Label('Anillos: <' + CONFIG.distRings[0] + 'm | ' +
  CONFIG.distRings[0] + '-' + CONFIG.distRings[1] + 'm | >' + CONFIG.distRings[1] + 'm'));
panel.add(ui.Label('Periodo: ' + CONFIG.years.join(', ')));
Map.add(panel);

// Inspector
Map.onClick(function(coords) {
  var pt = ee.Geometry.Point(coords.lon, coords.lat);
  lai2024.addBands(zonaImg).addBands(hicCode.rename('hic'))
    .addBands(anillo).addBands(dem.rename('alt'))
    .addBands(distBorda)
    .reduceRegion({reducer: ee.Reducer.first(), geometry: pt, scale: CONFIG.scale})
    .evaluate(function(r) {
      if (!r) return;
      var zn = ZONA_NAMES[r.zona] || 'sin HIC';
      print(coords.lon.toFixed(4) + ', ' + coords.lat.toFixed(4) +
            ' | HIC: ' + (r.hic || '?') +
            ' | Zona: ' + zn +
            ' | Dist borda: ' + (r.dist_borda ? r.dist_borda.toFixed(0) : '?') + 'm' +
            ' | Alt: ' + (r.alt ? r.alt.toFixed(0) : '?') + 'm' +
            ' | LAI: ' + (r.LAI ? r.LAI.toFixed(2) : 'N/A'));
    });
});

print('==========================================================');
print('Tasks:');
print('  LAI_zonal_HIC_aralar         (LAI por zona HIC x dist)');
print('  clima_diario_aralar          (ERA5-Land diario)');
print('  ERA5Land_monthly_1981_present (serie larga para SPEI)');
print('  Validacion_PROSAIL_vs_MODIS  (scatter MODIS)');
print('  zonas_caracterizacion        (alt, pend, dist por zona)');
print('  PROSAIL_NN_Aralar_20XX       (GeoTIFFs x' + CONFIG.years.length + ')');
print('Carpeta Drive: ' + CONFIG.folder);
print('==========================================================');