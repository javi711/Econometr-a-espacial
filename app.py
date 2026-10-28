# -*- coding: utf-8 -*-
"""
🛰️ Observatorio SpVAR Colombia · app.py (ARCHIVO ÚNICO · Google Colab + Streamlit Cloud)
Clima satelital (CHIRPS) → Producción agropecuaria → Pobreza (IPM) · 33 departamentos · 2018–2025

CÓMO USARLO
  • Google Colab : pega TODO este archivo en una sola celda y ejecútala (Entorno de ejecución → Ejecutar todas).
                   Se instala lo que falte, la interfaz Gradio aparece dentro de Colab y se imprime un enlace
                   público https://….gradio.live (72 h).
  • Streamlit    : sube este app.py a GitHub junto con requirements.txt y despliega en Streamlit Cloud
                   (Main file path = app.py). En local:  streamlit run app.py
  • requirements : `python app.py --requirements` escribe el requirements.txt de abajo junto a este archivo.

REQUIREMENTS.TXT (Streamlit Cloud · versiones verificadas juntas · ruedas binarias: GEOS/PROJ/GDAL vienen
dentro de shapely, pyproj y pyogrio, así que no se compila nada; Python 3.12 o 3.13)
  streamlit==1.66.1
  numpy==2.5.3
  pandas==3.0.5
  scipy==1.18.1
  geopandas==1.2.0
  shapely==2.2.0
  pyproj==3.8.0
  pyogrio==0.13.0
  libpysal==4.15.1
  esda==2.10.0
  spreg==1.9.1
  scikit-learn==1.9.1
  plotly==7.1.0
  requests==2.34.2
  (Gradio solo se usa en Colab y se instala allí automáticamente; no va en requirements.txt.)

INFORME DE AUDITORÍA (cambios frente al script original)
  A. Simultaneidad y endogeneidad. El original estimaba cada ecuación por MCO/ML univariada tratando Agro(t) como
     exógeno en la ecuación de Pobreza. Ahora W·y(t) y Agro(t) [+ W·Agro(t) en SDM] son endógenas y se
     instrumentan con W·X y W²·X de los regresores exógenos (W²·X y W³·X si W·X ya es regresor), según
     Kelejian-Prucha (1998) y Anselin: spreg.TSLS (S2SLS, errores White) sin error espacial y
     spreg.GM_Endog_Error_Het (GS2SLS / GMM de Kelejian-Prucha robusto a heterocedasticidad) con error. La
     estimación VI se acepta solo si es admisible: F de 1ª etapa ≥ 10 (Staiger-Stock), |ρ|, |λ| < 0.98, sin
     el recorte interno de spreg a ±0.9 y con bloque dinámico propio estable. Si los instrumentos son débiles
     (frecuente con N = 33: W·X explica poco a W·y), se usa la QML de Lee-Yu, que identifica ρ por el
     Jacobiano (Lee 2004), con función de control para Agro(t) (residuo de la 1ª etapa como regresor; su p
     es la prueba de Hausman). La lluvia se trata como estrictamente exógena. Se reportan F, Sargan, AK y
     Hausman por ecuación. La correlación contemporánea entre innovaciones (tipo SUR) entra en la inferencia:
     covarianza conjunta entre ecuaciones por sándwich de funciones de influencia y Monte Carlo
     (Krinsky-Robb) conjunto. Opción avanzada SPVAR_NICKELL_IV=1: instrumenta además los rezagos con niveles
     t−1 y t−2 (Arellano-Bover); viene apagada porque con T = 7 esos instrumentos son débiles.
  B. A0 y multiplicador. A0 (3N×3N) = ρ_k W en la diagonal, β_kj I + θ_kj W bajo la diagonal (sistema
     recursivo clima→agro→pobreza). Se verifica max|eig(A0)| < 1 y se comprueba numéricamente que la serie
     de Neumann Σ A0^m converge a S = (I − A0)⁻¹ (error máximo reportado en la pestaña Modelo).
  C. Impactos LeSage-Pace (2009) por bloque S_kj: Directo = tr(S_kj)/N, Total = 1′S_kj1/N,
     Indirecto = Total − Directo, con IC 95 % de la distribución Monte Carlo conjunta.
  D. Selección en panel. El original aplicaba LM/AIC/BIC sobre datos con media restada (Within), cuyos errores
     quedan correlacionados en el tiempo y con grados de libertad mal contados. Ahora se usa la
     transformación ortonormal de Lee-Yu (2010) (desviaciones ortogonales hacia adelante): N(T−1)
     observaciones con errores i.i.d. y W_NT = I_(T−1) ⊗ W exacta; los efectos de año entran como dummies
     del periodo transformado (exacto, sin doble demeaning). AIC/BIC con n = N(T−1) y k que incluye
     dummies; árbol LM robusto de Anselin et al. (1996); prueba AK sobre los residuos VI: si rechaza, se
     añade el componente de error (MCO→SEM, SLX→SDEM, SAR→SAC). Si una estimación VI da |ρ| o |λ| ≥ 0.98,
     se pasa a la siguiente estructura según BIC.
  E. Sp-IRF. Ψ_h = Φ^h S B (L̃ ⊗ I_N), con Φ = S A1, B = diag[(I − λ_k W)⁻¹] y L̃ la factorización de
     Cholesky (orden clima→agro→pobreza) de la covarianza de innovaciones, normalizada a diagonal unitaria;
     un choque de 1 d.e. es la desviación estándar del choque ortogonal. Se exige max|eig(Φ)| < 1.
  Robustez. GeoJSON: ruta local / caché / requests.Session con reintentos / respaldo embebido (geometría
  simplificada + topología Queen/Rook + centroides exactos, así W no cambia aunque falle la red).
  Simplificación con coverage_simplify (respeta fronteras compartidas) + make_valid. Semillas fijas.
  Datos: series reales embebidas sin cambios (CHIRPS v3.0, DANE VA agropecuario, DANE IPM).
"""

# =============================================================================
# 0 · ENTORNO E INSTALACIÓN
# =============================================================================
import os, io, re, sys, json, zlib, base64, zipfile, warnings, contextlib, tempfile, importlib.util, subprocess

REQUIREMENTS = """streamlit==1.66.1
numpy==2.5.3
pandas==3.0.5
scipy==1.18.1
geopandas==1.2.0
shapely==2.2.0
pyproj==3.8.0
pyogrio==0.13.0
libpysal==4.15.1
esda==2.10.0
spreg==1.9.1
scikit-learn==1.9.1
plotly==7.1.0
requests==2.34.2
"""


def _detectar_entorno():
    """streamlit (streamlit run app.py) · colab · jupyter · script (python app.py)."""
    if "streamlit" in sys.modules:
        try:
            from streamlit import runtime
            if runtime.exists():
                return "streamlit"
        except Exception:
            pass
    if "google.colab" in sys.modules:
        return "colab"
    try:
        get_ipython  # noqa: F821
        return "jupyter"
    except NameError:
        return "script"


ENTORNO = _detectar_entorno()
_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

if "--requirements" in sys.argv:
    with open(os.path.join(_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS)
    print("✓ requirements.txt escrito en", _DIR)
    sys.exit(0)


def _version(mod):
    try:
        from importlib.metadata import version
        return tuple(int(x) for x in re.findall(r"\d+", version(mod))[:2])
    except Exception:
        return None


if ENTORNO != "streamlit":   # en Streamlit Cloud las dependencias vienen de requirements.txt
    requeridos = {"gradio": ("gradio>=5.0,<7", (5, 0)), "plotly": ("plotly>=5.24", (5, 24)),
                  "geopandas": ("geopandas", None), "libpysal": ("libpysal", None),
                  "esda": ("esda", None), "spreg": ("spreg>=1.8", (1, 8))}
    faltan = [pip for mod, (pip, minimo) in requeridos.items()
              if importlib.util.find_spec(mod) is None or (minimo and (_version(mod) or (0, 0)) < minimo)]
    if faltan:
        print("Instalando:", ", ".join(faltan), "…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltan])
    print("✓ Dependencias listas")

# =============================================================================
# MÓDULO 1 · MOTOR ECONOMÉTRICO (puramente algorítmico: NumPy, Pandas, SciPy, PySAL, GeoPandas)
# =============================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import sparse, stats
from libpysal.weights import W as PesosW, Queen, Rook, KNN, DistanceBand, WSP
from libpysal.weights.util import attach_islands, min_threshold_distance
from esda.moran import Moran, Moran_Local
from spreg import OLS, TSLS, ML_Lag, ML_Error, GM_Error_Het, GM_Endog_Error_Het

warnings.filterwarnings("ignore")

# Caché: en Streamlit persiste entre re-ejecuciones del script y entre sesiones; en Colab es lru_cache.
if ENTORNO == "streamlit":
    import streamlit as st

    def _cache(f):
        return st.cache_resource(show_spinner=False)(f)
else:
    from functools import lru_cache

    def _cache(f):
        return lru_cache(maxsize=None)(f)

# -----------------------------------------------------------------------------
# 1. DATOS (reales, embebidos para arranque inmediato; mismas fuentes del cuaderno SpVAR)
# -----------------------------------------------------------------------------
# --- Respaldo embebido: valores reales extraídos el 20-oct-2026 con el mismo procedimiento del cuaderno ---
# CHIRPS v3.0 (UCSB-CHC): precipitación anual media sobre el polígono MGN de cada departamento, mm, 1991–2025
CHIRPS_TXT = """05:2959,2928,3419,3232,3201,3617,2746,3407,3904,3553,3189,2957,3350,3131,3329,3509,3550,3573,3183,3743,3643,3221,3209,3029,2861,3061,3258,3152,3061,2968,3403,3764,2849,2840,3511
08:806,976,1125,877,1342,1202,791,1294,1531,1034,973,851,1403,1138,1268,1207,1415,1363,898,1905,1695,981,1146,951,688,1253,1301,1039,995,1244,988,1875,1091,1451,1098
11:1200,943,1179,1115,1074,1248,901,1159,1242,1103,954,1097,1046,1207,1110,1258,1085,1281,1004,1625,1595,1104,1159,1030,704,1030,1140,1051,1016,1144,1149,1486,1118,1123,1285
13:2059,2110,2308,2202,2435,2664,1824,2609,2809,2264,2129,1932,2399,2228,2443,2536,2539,2471,2099,3153,2899,2362,2422,2078,1925,2382,2543,2295,2135,2210,2391,3087,2088,2358,2512
15:1767,1759,2057,2017,1920,2037,1610,2111,2150,1913,1732,1825,1967,2038,1938,2075,1831,2079,1770,2337,2403,1985,1854,1683,1657,1989,2037,1958,1845,1809,1878,2368,1979,2003,2119
17:2548,2553,3005,2872,2823,2925,2259,2978,3449,3004,2549,2532,2702,2587,2981,2951,3021,3518,2696,3423,3515,2619,2845,2526,1967,2439,3064,2912,2632,2471,2972,3704,2689,2598,3103
18:3221,3233,3694,3518,2967,3304,3092,3505,3558,3519,3356,3372,3265,3393,3212,3133,3097,3204,3465,3268,3340,3169,3515,3445,3375,3349,3239,3184,3601,3108,3367,3515,3288,3283,3365
19:3822,3503,4085,3902,3909,4096,3883,4029,4481,3834,3474,3518,3736,3535,3647,3942,3822,4218,3827,4017,4014,3495,3937,3869,3406,3690,3936,3573,3831,3487,4218,4514,3929,3437,4204
20:1323,1618,1683,1570,1906,2000,1216,1994,2256,1762,1581,1429,1867,1569,1872,1803,1908,1828,1462,2645,2303,1716,1691,1559,1250,1736,1930,1603,1581,1671,1560,2479,1725,1815,1806
23:2154,2060,2261,2189,2163,2567,1975,2312,2578,2099,2030,1986,2211,2077,2411,2332,2461,2349,2051,2634,2473,2338,2082,1944,1899,2077,2395,2122,2137,1960,2380,2671,2013,2258,2389
25:1727,1537,1898,1854,1709,1892,1478,1853,2015,1744,1598,1757,1689,1877,1809,1944,1710,1995,1672,2103,2247,1740,1832,1700,1414,1809,1921,1797,1751,1614,1833,2317,1877,1763,2046
27:5216,5265,5631,5845,5314,6557,4778,5473,6656,6234,6060,6012,5785,5538,6094,6145,6152,5951,6118,6120,6361,6084,5889,5715,5209,5368,5724,5417,5545,5137,5952,6142,5392,5541,6048
41:1619,1336,1810,1690,1569,1804,1385,1579,2067,1630,1352,1456,1439,1565,1701,1766,1619,1928,1595,1990,2264,1576,1633,1556,1072,1476,1906,1462,1515,1499,1950,2120,1607,1649,1834
44:562,736,757,635,1006,792,502,904,1226,617,715,594,969,1015,1126,775,1134,1030,635,1623,1333,852,695,654,470,1037,900,661,632,884,686,1239,771,912,729
47:1464,1610,1647,1455,1823,1829,1222,1885,2162,1604,1499,1382,1760,1634,1716,1716,1949,1766,1302,2540,2250,1606,1630,1381,1131,1610,1773,1492,1390,1548,1393,2268,1473,1638,1545
50:2925,2637,3030,3001,2674,2948,2651,2999,2962,3036,2775,2856,3023,3078,2908,2890,2694,2837,2878,3080,3117,2787,3056,2891,2791,2936,2745,3002,2914,2643,2865,3311,2792,2890,3212
52:3528,2966,3510,3259,3430,3249,3970,3942,4227,3490,2664,3393,3660,3266,3048,3775,3517,4009,3273,3865,3517,3304,3275,3992,3601,3284,3513,3369,3610,3450,4331,4045,4108,3277,4225
54:2166,2188,2289,2360,2604,2716,1881,2574,2831,2284,2133,2149,2478,2498,2716,2547,2417,2471,2111,3316,2942,2356,2096,1901,1804,2151,2646,2439,2163,2296,1712,3160,2372,2347,2366
63:2113,1947,2556,2330,2166,2685,2114,2396,3030,2579,2022,2024,2129,2106,2270,2369,2265,2709,2083,2812,2884,2231,2361,2104,1769,2206,2713,2564,2406,1987,2502,3023,2553,2170,2526
66:2637,2524,3110,2922,2872,2900,2364,2817,3344,2972,2555,2445,2626,2498,2858,3137,2900,3297,2824,3446,3444,2649,2957,2578,2172,2516,2986,2891,2780,2597,3019,3262,2666,2559,2931
68:2165,2152,2505,2487,2592,2624,2117,2564,2809,2393,2313,2192,2538,2314,2461,2584,2467,2572,2353,2910,2927,2283,2323,2207,2130,2326,2443,2379,2367,2244,2460,2961,2188,2258,2532
70:1587,1741,1830,1739,1909,2233,1405,2162,2270,1797,1656,1531,1888,1787,1942,2047,2113,2159,1705,2735,2517,2016,1911,1519,1442,1790,2157,1843,1749,1742,1925,2539,1643,2051,2086
73:1885,1589,2016,1935,1854,2113,1546,2040,2445,1948,1621,1750,1770,1745,2003,2061,2034,2429,1824,2435,2488,1737,1872,1717,1315,1773,2160,1925,1816,1739,2260,2675,1983,1816,2229
76:3513,3415,3642,3590,3671,3930,3192,3507,3976,3395,3568,3438,3646,3254,3639,3811,3684,3612,3670,3914,3789,3528,3470,3491,3253,3562,3663,3526,3396,2858,3682,3625,3121,2949,3585
81:2094,2415,2224,2299,2257,2525,1933,2401,2510,2177,2055,2168,2083,2420,2154,2248,2235,2168,2045,2451,2625,2304,2253,2073,2035,2324,2241,2539,2088,2016,2140,2572,2052,2223,2529
85:2283,2341,2620,2577,2504,2302,2029,2445,2478,2536,2224,2353,2463,2633,2487,2468,2208,2400,2311,2774,2795,2631,2159,2266,2147,2535,2256,2607,2246,2139,2248,2640,2244,2470,2585
86:3279,3310,3806,3494,3015,3273,3205,3610,4037,3561,3321,3377,3396,3392,3539,3608,3595,3750,3741,3269,3638,3370,3793,3733,3612,3671,3633,3334,3917,3519,3960,3905,3838,3525,3727
88:1613,1484,1940,1821,2417,2295,1644,2406,2696,1856,2173,2016,2158,2223,2605,1828,1984,2842,1695,2585,2387,1904,1602,1878,1364,1495,2094,1568,1389,2301,1293,2317,1768,2216,1994
91:3391,3440,3739,3838,3342,3749,3437,3536,3606,3343,3405,3639,3678,3395,3340,3586,3611,3366,3777,3312,3571,3543,3588,3657,3643,3577,3385,3353,3986,3396,3379,3635,3312,3486,3373
94:3252,3333,3651,3274,3297,3538,3181,3533,3486,3512,3274,3703,3409,3436,3452,3694,3391,3342,3372,3211,3794,3585,3287,3330,3130,3452,3192,3395,3571,3081,3289,3373,3089,3082,3257
95:3154,3104,3294,3267,2886,3301,3150,3415,3083,3452,3079,3224,3486,3300,3133,3108,2939,3014,3223,3007,3424,3203,3566,3232,3349,3251,2988,3148,3513,2848,3048,3391,3067,3048,3434
97:3532,3557,3841,3780,3633,3897,3729,3917,3803,3820,3546,3947,3817,3459,3608,3600,3398,3461,3781,3288,3679,3893,3840,4037,3947,3941,3608,3891,4134,3342,3572,3847,3389,3709,3454
99:2805,2717,3028,2846,2716,2904,2584,3035,2918,3089,2689,3122,2923,3031,2764,2914,2750,2728,2804,2942,3294,2914,2643,2835,2830,2889,2734,2928,2822,2473,2656,2937,2518,2624,2932"""
# DANE: VA agropecuario, series encadenadas de volumen (base 2015), miles de millones de pesos, 2005–2025pr
AGRO_TXT = """05:4856.14,4883.98,5266.85,5106.74,4981.44,5049.66,5106.74,5368.48,5733.25,5756.92,6153.06,6362.5,6567.34,6677.55,6897.45,7000.75,7437.48,7319.4,7524.26,8018.14,8291.15
08:280.73,288.25,276.96,292.77,298.79,293.52,289.76,288.25,307.82,323.63,329.65,352.85,374.74,392.04,405.4,416.77,443.51,453.11,447.54,468.13,499.06
11:11.01,11.01,11.03,11.33,11.41,11.23,11.52,11.53,11.57,12.24,12.47,11.72,12.8,12.96,13.05,13.07,13.49,13.61,13.8,14.22,14.37
13:883.15,884.28,905.85,1032.99,1068.18,970.55,995.53,1064.77,1127.21,1157.85,1196.23,1218.04,1280.08,1335.33,1374.02,1448.18,1497.95,1570.23,1586.38,1662.85,1753.55
15:1815.31,1764.38,1859.44,1926.22,1933.01,1888.87,1921.69,2065.42,2235.18,2186.52,2142.92,2186.42,2404.56,2449.41,2527.67,2565.21,2602.61,2529.49,2541.46,2694.95,2716.02
17:957.79,1026.4,1035.82,1031.78,999.49,1066.76,1015.64,986.04,1100.39,1224.15,1274,1211.3,1261.8,1280.99,1310.64,1327.74,1324.21,1294.4,1314.47,1424.95,1454.02
18:397.46,382.44,409.75,415.22,404.29,381.07,423.41,456.19,445.27,443.9,459.77,497.93,509.45,503.71,525.75,545.48,565.35,564.29,568.38,593.54,603.79
19:1412.28,1451.92,1470.81,1397.17,1100.74,1176.27,1200.81,1248.01,1459.48,1565.21,1692.62,1766.23,1851.87,1826.48,1883.73,1894.35,2013.2,2036.72,2084.66,2242.59,2243.57
20:1231.86,1287.48,1277.75,1233.25,1233.25,1133.15,1160.95,1215.18,1217.96,1254.11,1255.98,1168.83,1242.67,1275.82,1325.79,1348.14,1401.45,1373.9,1402.53,1448.14,1497.96
23:1281.65,1429.83,1712.92,1550.36,1514.98,1499.5,1501.71,1453.05,1474.06,1527.14,1542.7,1478.46,1495.78,1528.91,1556.61,1598.12,1631.24,1673.66,1674.27,1742.14,1838.92
25:5589.25,5601.55,5751,5793.2,5700.01,5849.46,5854.73,5872.31,6215.16,6273.18,6299.94,6591.91,7112.33,7277.04,7470.68,7591.36,7973.7,7723.93,7773.96,8264.95,8503.2
27:551.34,486.79,478.06,504.23,483.3,444.91,493.77,499,521.68,553.09,581.68,619.3,651.32,637.12,667.71,685.75,720.53,725.48,738.86,765.36,776.99
41:1611.27,1676.42,1749.52,1636.7,1530.23,1674.83,1678.01,1623.99,1840.09,2016.47,2112.37,2265.54,2275.41,2318.75,2367.69,2425.18,2461.44,2539.8,2609.51,2802.68,2880.76
44:425.99,448.57,475.66,409.43,377.82,358.25,362.77,380.83,350.72,331.16,326.32,331.61,373.95,374.65,387.06,391.62,402.25,408,411.43,437.3,448.15
47:1445.19,1424.86,1399.83,1470.21,1498.37,1429.55,1438.93,1520.26,1507.75,1543.72,1576.1,1590.56,1705.75,1706.39,1743.96,1728.65,1748.33,1699.02,1683.41,1725.69,1747.2
50:1058.31,1064.79,1072.35,1358.79,1564.85,1664.83,1800.49,1873.08,2064.67,2045.63,2361.32,2463.63,3050.17,3052.72,3122.56,3158.66,3441.72,3416.3,3428.43,3526.55,3691.45
52:1481.29,1657.59,1569.44,1415.17,1336.46,1358.5,1448.23,1571.01,1726.85,1726.85,1767.48,1729.65,1841.11,1887.15,1915.78,1996.15,2045.09,2000.67,2000.92,2084.72,2155.95
54:1150.89,1135.46,1055.24,1158.6,1260.43,1191,1167.86,1135.46,1243.46,1271.22,1198.66,1220.45,1247.82,1271.43,1299.24,1317.09,1332.49,1308.18,1339.32,1433.25,1432.72
63:718.47,859.63,815.22,861.22,870.73,900.87,842.18,815.22,851.7,885.01,934.17,972.94,1017.44,1027.34,1056.17,1057.8,1140.56,1107.19,1117.11,1207.93,1227.13
66:677.06,795.86,865.07,730.12,683.98,688.59,668.98,682.83,752.03,800.48,845.11,828.09,839.16,849.32,887.15,894.84,929.03,908.23,926.79,1005.88,1011.25
68:3021.58,2962.84,2862.64,3147.69,2936.93,3128.69,3213.34,3211.61,3318.73,3475.94,3793,3928.34,4190.56,4301.32,4452.34,4518.57,4697.43,4603.42,4732.77,5090.34,5291.42
70:456.69,499.88,513.91,495.56,549.54,470.73,503.12,539.83,558.18,554.94,589.47,655.94,694.79,693.96,701.64,734.62,776.22,761.47,755.74,793.46,816.29
73:2299.26,2425.66,2425.66,2204.85,2140.85,2161.65,2252.85,2187.25,2449.66,2601.66,2736.9,2744.16,2775.75,2733.5,2783.46,2823.26,2930.5,2841.45,2928.25,3158.36,3253.39
76:3439.32,3465.02,4001.56,3590.32,3803.97,3688.31,3795.94,3752.57,3951.76,4253.77,4329.24,4393.95,4416.19,4557.65,4673.91,4871.21,5069,5243.65,5370.49,5704.64,5987.6
81:469.03,460.82,491.31,571.05,609.74,615.6,670.72,780.94,770.39,763.35,748.31,766.69,839.8,866.27,885.16,906.62,960.28,948.52,969.43,1025.88,1063.02
85:484.76,507.58,581.49,682.57,704.31,736.92,817.35,946.69,1021.69,1009.73,1219.71,1427.21,1525.38,1527.98,1555.17,1578.62,1685.19,1648.97,1687.46,1835.47,1887.55
86:193.27,183.51,173.74,144.46,144.46,152.27,160.08,167.89,160.08,171.79,191.48,191.04,181.22,185.57,187.05,192.83,205.16,204.55,206.72,215.71,222.01
88:12.27,12.27,13.63,16.36,14.99,14.99,14.99,14.99,16.36,14.99,14.65,14.78,13.85,14.96,15.3,16.8,17.5,18.02,18.02,18.25,19.57
91:114.17,92.43,78.83,78.83,97.86,106.02,100.58,97.86,97.86,92.43,95.84,93.92,97.63,102.32,103.96,113.13,119.5,123.4,122.86,122.77,126.43
94:18.21,13.01,13.01,15.61,23.41,26.02,26.02,26.02,26.02,26.02,24.87,26.15,28.65,29.57,30,31.65,32.41,31.42,31.39,32.39,33.29
95:155.02,143.34,134.84,123.74,109.95,112.83,112.93,113.27,119.88,121,129.03,139.94,144.22,145.82,147.17,150.64,161.97,158.31,159.59,168.79,169.99
97:7,14,18.67,21,18.67,14,14,14,14,16.34,16.76,18.25,18.26,18.67,18.76,18.88,18.85,18.85,19.17,19.76,19.87
99:156.16,156.5,160.68,161.86,159.26,163.43,156.57,149.02,147.13,148.51,172.18,172.66,172.67,177.33,182.5,185.65,193.15,187.27,187.89,195.55,193.85"""
# DANE: incidencia de pobreza multidimensional (IPM, % de personas), total departamental, 2018–2025
IPM_TXT = """05:15.3,15.7,14.9,14.3,10.7,9.5,10.9,9.5
08:21.1,14.9,14.1,15.5,10.2,12,9.5,8.2
11:4.1,7.1,7.5,5.7,3.8,3.6,5.4,2.2
13:31.9,26.9,28.1,26.8,19.4,18.4,15.6,17
15:16.5,12.8,11.7,10.9,9.6,9.9,6.9,6.4
17:13.8,14.3,14.5,11.5,10.5,7.4,9.2,7.9
18:26.9,25.7,26.1,23.1,19.6,17.2,13.9,15
19:27.7,24,28.2,18.6,18.3,15.8,14,13
20:31.7,25.5,27.2,25.3,19.1,17.7,13.4,14.8
23:34.4,34.7,31.8,26.9,26.9,21.4,25.7,19.8
25:10.5,12.3,11.4,9.4,7.3,7.6,7.4,4.1
27:46.3,42.3,49,36,36.8,37.4,33.9,30.8
41:18,18.3,23.4,17.5,13.3,11.9,10.9,9.7
44:53.3,48.8,51.7,48.7,42.9,42.6,39.3,40.1
47:36.4,31.6,33.4,32.4,23,21.4,19.2,17.6
50:15.6,19.1,14.1,14.9,11.3,12.9,10.4,12.3
52:33.1,23.2,27.3,22.1,17.6,16.6,18.1,13.2
54:29.5,24.2,26.1,24.7,18.5,20.5,15.2,13.3
63:14.2,10.2,12.9,10.9,10,7.5,7.4,6.7
66:11.6,11.1,13.1,10.7,10,11.8,9.5,4.9
68:12.6,12.4,12.5,13.3,10.4,9.8,6.8,6.9
70:41.7,33.3,38.1,30.3,26,23.1,21.8,25.4
73:22.2,15.2,19,16.6,10.5,12.9,12.6,12.2
76:14.1,10.8,11.1,8.6,9.7,7.2,6.2,4.9
81:27.6,23.3,26.1,26.8,22.6,22.8,17.7,12.7
85:19,18.3,19.6,19.5,13.3,15,11.3,8.9
86:24.1,25.4,27.8,22.8,20.8,13.2,11.8,13.6
88:8.5,8.2,11.9,7,8.8,5,6,4.3
91:35.4,35.6,39,25.7,27.9,25.4,15.6,18.1
94:60.6,67,65.9,57.3,46.5,52.1,49,51.8
95:31.4,30.9,34.6,31,26.9,30.6,21.3,20
97:68.5,66.5,65.6,52.7,47.1,55.7,37.4,41.7
99:63.5,72.2,75.6,64.8,75.4,65.4,70.2,55.2"""

# --- Respaldo geográfico embebido (solo se usa si no hay red ni caché) ---
# Geometría MGN-DANE de los 33 departamentos simplificada con coverage_simplify(0.03) (fronteras compartidas
# intactas), comprimida zlib + base64. TOPOLOGIA: vecinos Queen/Rook y puntos representativos EPSG:3116 calculados
# sobre la geometría completa, para que las matrices W sean idénticas aunque se use este respaldo.
GEO_RESPALDO = (
    "eNqtfc2uJreR5asIWlcVkv9k766rNHINSlXqUpUBj2EMGj2ahgFPy9BoFobhhZe96EVjHkkvNjyHycPM4NVivvbm4saXf0wmGYw4cSL4ly9/+vOfvv/y"
    "H778L9//00//58fvX//wxz9+/88//eGHf/3yxZf/c/z2v7/8h9/9xZzXj/7pxx/+9P2PP/0Bx//y5ZtvP334769fv/n6Qz/pSP34+OX9N7/62H95ev/p"
    "7Yd//Pz26cu/vvjyX77/4X99/9OPf8Z1522//eGPf/4XPvSff/jhx//xh3/9p5/43N/97mWJr2p4UV/537+gUGrqkguSPWVXTjkfkI9QTzm59qK8aoeb"
    "8lG7XHyccnZdzjFLxvF+1ZQbro95Pi82nB9imLKLkFuTHHj+KQYPaT4stMqLq07uoj+8TsbZh57tY4Z8jGeFV81B9i1NOaBtrvpTrhW3P4qbcisv8qtW"
    "Z2MO5yC72XbnUpdLnsd9b22XZ1f5lLuYS5rNSw1y0KsfkJNuH/i4mNRVCY8L+lTJ47gvszeyw+OcX10P+Sj6dDlAVvfg/PSqqbm9Oam/o24XKmQdzjl1"
    "ueR5u9K/FOTZu4XiKdSjXKWcIcUptxYgu3Hr9OpoeHQ+BwFk1+XZE+mVi7hbmoe95+FDcsHt4/md06tQcHo+O6YfKbx9nnLi40ooU+bd/bxdxtGkuxW0"
    "NenimtEtcR5tNVzORvfy6lwksy3nzbtcKVedn9kVXuc39nqaIttWWz5ld2CIHa5OOfOTxik2DJEjzbt7iE538/2T9r/HvDoUyK7NxkUOoDwflg/KzZ9y"
    "cRhw7hzQkHk7N29fMH6D2lZ496hXrW2M71NsKV6mQ5cLZ0ubx2vBu9WkpzWcX/M8v/rI2TzlXDi757ukyMl7zMsTpnaebUmBqiLpYofjwc1XzXkoGrX9"
    "wPlJr94CHpbq7PiWqfUo9ed4KK3STvHwOKp+aH1eV6kl3KtCzuoYiF1ZTJEa+fDqJ2q15or6oVIj6yvxxftZUw4R16+v1rVc1/Al6jPhuNfjS8Lzg746"
    "+rn2Ab/ag+anqOM1d3mN8Bxx/6zn5d7VtT9V3wW3K1ljzmE5qmoOW5vjvNpn3j1cRnTVXO0iGxOqTq9svJ+nh+Txcse6O447fffo0dg1YaLD+X3STrmi"
    "s6uf948Ng269fH8ylys/tUPkOIpTtZSI67Obx4vzHDdu6pojUU5SVbh/KfP6EPgty0XzYahIMTZ03qHHQV0sOUK5Q54qunmONK1ApyXw+9//9a8v/j9t"
    "krrZJJ/e/fy3bpe8/vCQUdL6WuqOPuyjmp7wQ/BaJPqn6D9E9WVf6XhGm7rYR94jefVWgJy1TvCOxa91JPKHNnv/iPyh6o5du+CHNte82oeTc9DCc9HL"
    "vKI1/eDZhnbM9yiBrUzJaWlsfI8yF1ZMYL7YPOPSFw98Gufsp/nVh68/fPr5by++ePPq9auHvo7rpkfUOtNNmT5tur0kWyH0l+z2kZep4iDLMundGqYG"
    "6CfjWHO6to/Rfu9QZLWlLjtfZLVRPublvq9C/W9Zsn8BY0lWWF92+iNlRrm+gMNamb17JJxfZUZhxY6a4pHfnHasrDo8L+qTY4Xuz4tTdrzf1NdoKe7n"
    "ZBetvnvkY4b9Y777+d9/8/Txge/4zIjfp4SZMmZKbXPOzMpt3j43syPPcF6zio06iizJbiPhh0Mn9E/Y+jqZbnKV3V/76t3gymiK4XjOcwgVnp81fOuR"
    "u5xS0BwtXY4avrlbKpCLLGfcP+gdUzd9W7emLuO7XQzxyMf7NcL7IoDT5QhE3N7lqBGK2y+17A7c7/BrhFKebkstWKNalluSodW7RSIZi0Ct5ZRLwyJQ"
    "T7sMMhbg07TBrMUCm6fT0610LPd53q0UOoQx3e5+FIm0Rop8JhpWLeh2dNHK5TjdxXo7nJNuRxdtrq9QEwV22Lq8Yb2MwT3rwlkXb3cBrYdoHMi7e2l8"
    "z801Na6rdW0319e6xtZ1Nq61db2ta25dd+PXw1Lwcv6GVRY0JhONOj/7LdHsWmtSPtoy2s6zq1RgpgnZSrmsYH2ASmXWjAHr1K3dOOWAD9IAkENaCx7k"
    "uPzBQDlPs8YFTKjUshw+TKAs/xBLRpd9kgNIOS0rCvO9ynLoiwVkWWH9SdAvx1Rq+LJQQGfXw0OkWjz0xDQUqStTUcLZhpaTT3nwinDMZyTHE1JZjaAq"
    "zWoF3FgoY5d2++SRpSPtS8dvn17//LcHVg7fTSO6U2ePuP4JGyfWaHw/TnBiogP+FaVzqPejDb5djjpKRzSdo9F3Kx/nx3Pa+/7B6Erq+lAIfczDcSAd"
    "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"
)
TOPOLOGIA = json.loads("".join("""
    {"puntos":{"05":[849062.9,1279417.1],"08":[900862.9,1672738.6],"11":[989996.2,968262.3],"13":[982754.7,1477218.0],"1
    5":[1139289.8,1138869.5],"17":[859193.5,1076213.6],"18":[920414.2,616044.2],"19":[689744.1,730769.4],"20":[1050306.8
    ,1518043.3],"23":[820651.7,1420794.7],"25":[1044969.4,1018554.7],"27":[667827.1,1186415.2],"41":[846071.3,785719.7],
    "44":[1163121.5,1758309.8],"47":[953216.6,1614023.2],"50":[1132729.9,846062.0],"52":[575359.5,662295.0],"54":[113353
    6.0,1384572.2],"63":[818341.1,980008.0],"66":[789269.9,1056845.4],"68":[1092403.4,1260284.8],"70":[878650.2,1509420.
    1],"73":[885157.9,944652.2],"76":[772589.8,942175.4],"81":[1346334.8,1217909.8],"85":[1267413.0,1080269.3],"86":[778
    812.6,542245.4],"88":[167709.1,1892947.7],"91":[1305698.4,258949.7],"94":[1545885.0,782163.6],"95":[1190221.9,689966
    .0],"97":[1348861.6,535745.5],"99":[1512603.6,1001773.6]},"Queen":{"05":["13","15","17","23","27","66","68"],"08":["
    13","47"],"11":["25","41","50"],"13":["05","08","20","23","47","68","70"],"15":["05","17","25","50","54","68","81","
    85"],"17":["05","15","25","66","73"],"18":["19","41","50","86","91","95","97"],"19":["18","41","52","73","76","86"],
    "20":["13","44","47","54","68"],"23":["05","13","70"],"25":["11","15","17","41","50","73","85"],"27":["05","66","76"
    ],"41":["11","18","19","25","50","73"],"44":["20","47"],"47":["08","13","20","44"],"50":["11","15","18","25","41","8
    5","95","99"],"52":["19","86"],"54":["15","20","68"],"63":["66","73","76"],"66":["05","17","27","63","73","76"],"68"
    :["05","13","15","20","54"],"70":["13","23"],"73":["17","19","25","41","63","66","76"],"76":["19","27","63","66","73
    "],"81":["15","85","99"],"85":["15","25","50","81","99"],"86":["18","19","52","91"],"88":[],"91":["18","86","97"],"9
    4":["95","97","99"],"95":["18","50","94","97","99"],"97":["18","91","94","95"],"99":["50","81","85","94","95"]},"Roo
    k":{"05":["13","15","17","23","27","66","68"],"08":["13","47"],"11":["25","41","50"],"13":["05","08","20","23","47",
    "68","70"],"15":["05","17","25","54","68","81","85"],"17":["05","15","25","66","73"],"18":["19","41","50","86","91",
    "95","97"],"19":["18","41","52","73","76","86"],"20":["13","44","47","54","68"],"23":["05","13","70"],"25":["11","15
    ","17","41","50","73"],"27":["05","66","76"],"41":["11","18","19","25","50","73"],"44":["20","47"],"47":["08","13","
    20","44"],"50":["11","18","25","41","85","95","99"],"52":["19","86"],"54":["15","20","68"],"63":["66","73","76"],"66
    ":["05","17","27","63","73","76"],"68":["05","13","15","20","54"],"70":["13","23"],"73":["17","19","25","41","63","6
    6","76"],"76":["19","27","63","66","73"],"81":["15","85","99"],"85":["15","50","81","99"],"86":["18","19","52","91"]
    ,"88":[],"91":["18","86","97"],"94":["95","97","99"],"95":["18","50","94","97","99"],"97":["18","91","94","95"],"99"
    :["50","81","85","94","95"]}}
""".split()))


def embebido(txt, anio0, nombre):
    filas = []
    for linea in txt.strip().splitlines():
        cod, vals = linea.split(":")
        filas += [(cod, anio0 + k, float(v)) for k, v in enumerate(vals.split(","))]
    return pd.DataFrame(filas, columns=["cod", "anio", nombre])


URL_GEO = os.environ.get("GEO_PATH", "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson")
VARS = ["clima", "agro", "ipm"]
CORTO = {"clima": "Clima", "agro": "Agro", "ipm": "Pobreza"}
ETQ = {"clima": "Anomalía de lluvia (%)", "agro": "Crecimiento agropecuario (%)", "ipm": "Pobreza multidimensional (%)"}
ESCALA = {"clima": "BrBG", "agro": "Greens", "ipm": "Reds"}
COLOR = {"clima": "#38bdf8", "agro": "#4ade80", "ipm": "#f87171"}
MATRICES = ["Queen", "Rook", "KNN3", "KNN4", "KNN5", "InvDist"]
ALPHA, H = 0.05, 6
NORMAL = (1991, 2020)
N_MC = 400                     # extracciones Monte Carlo (Krinsky-Robb) para IC de impactos
CACHE_GEO = [os.path.join(_DIR, "co_dptos.geojson"), os.path.join(tempfile.gettempdir(), "spvar_co_dptos.geojson")]


# -----------------------------------------------------------------------------
# 1b. GEOMETRÍA ROBUSTA (red con reintentos → caché → respaldo embebido)
# -----------------------------------------------------------------------------
def _sesion_http():
    s = requests.Session()
    reintentos = Retry(total=3, connect=3, read=3, backoff_factor=1.0,
                       status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=reintentos))
    s.mount("http://", HTTPAdapter(max_retries=reintentos))
    s.headers["User-Agent"] = "ObservatorioSpVAR/2.0 (+econometria espacial)"
    return s


def _geo_valida(g):
    return g is not None and len(g) == 33 and {"DPTO_CCDGO", "DPTO_CNMBR"} <= set(g.columns)


@_cache
def _cargar_geometria():
    """(GeoDataFrame, fuente). Nunca lanza excepción por red: cae al respaldo embebido."""
    if os.path.exists(URL_GEO):                                    # ruta local explícita (GEO_PATH)
        with contextlib.suppress(Exception):
            g = gpd.read_file(URL_GEO)
            if _geo_valida(g):
                return g, "local"
    for ruta in CACHE_GEO:                                         # caché local de una descarga previa
        with contextlib.suppress(Exception):
            if os.path.exists(ruta):
                g = gpd.read_file(ruta)
                if _geo_valida(g):
                    return g, "caché"
    if URL_GEO.startswith("http"):                                 # descarga con reintentos y timeout
        try:
            r = _sesion_http().get(URL_GEO, timeout=(10, 60))
            r.raise_for_status()
            g = gpd.read_file(io.BytesIO(r.content))
            if _geo_valida(g):
                for ruta in CACHE_GEO:
                    with contextlib.suppress(Exception):
                        with open(ruta, "wb") as f:
                            f.write(r.content)
                        break
                return g, "red"
        except Exception as e:
            print(f"⚠ GeoJSON no disponible por red ({type(e).__name__}); se usa el respaldo embebido.")
    g = gpd.read_file(io.BytesIO(zlib.decompress(base64.b64decode("".join(GEO_RESPALDO)))))
    return g, "respaldo"


def _solo_poligonos(geom):
    """make_valid puede devolver GeometryCollection con líneas; se conservan solo las partes poligonales."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    partes = [p for p in getattr(geom, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(partes) if partes else geom


def simplificar_seguro(geoms, tol=0.02):
    """Simplificación topológica sin polígonos inválidos: coverage_simplify mantiene las fronteras compartidas
    (no abre huecos entre departamentos); si no está disponible, simplify(preserve_topology=True). Después,
    make_valid y, si algo queda vacío o inválido, se vuelve a la geometría original de ese departamento."""
    geoms = np.asarray(geoms, dtype=object)
    try:
        s = shapely.coverage_simplify(geoms, tol)
    except Exception:
        s = shapely.simplify(geoms, tol, preserve_topology=True)
    s = np.array([_solo_poligonos(g) for g in shapely.make_valid(s)], dtype=object)
    malos = shapely.is_empty(s) | ~shapely.is_valid(s)
    s[malos] = geoms[malos]
    return s


_GEO_RAW, GEO_FUENTE = _cargar_geometria()
GDF = _GEO_RAW.rename(columns={"DPTO_CCDGO": "cod", "DPTO_CNMBR": "nombre"})
GDF["cod"] = GDF["cod"].astype(str).str.zfill(2)
GDF["departamento"] = GDF["nombre"].str.title().replace({
    "Bogotá, D.C.": "Bogotá", "Archipiélago De San Andrés, Providencia Y Santa Catalina": "San Andrés",
    "Quindio": "Quindío", "Valle Del Cauca": "Valle del Cauca", "Norte De Santander": "Norte de Santander"})
GDF = GDF[["cod", "departamento", "geometry"]]
GDF = (GDF.set_crs(4326) if GDF.crs is None else GDF).to_crs(4326)
GDF = GDF.sort_values("cod").reset_index(drop=True)
GDF["geometry"] = [_solo_poligonos(g) for g in GDF.geometry.make_valid()]
N = len(GDF)
COD = GDF["cod"].tolist()
NOMBRE = dict(zip(GDF["cod"], GDF["departamento"]))
assert N == 33, "Se esperan 33 departamentos"
assert COD == sorted(TOPOLOGIA["puntos"]), "Los códigos DANE no coinciden con la topología embebida"
GJ = json.loads(gpd.GeoDataFrame(GDF[["cod", "departamento"]], geometry=simplificar_seguro(GDF.geometry.values, 0.02),
                                 crs=4326).to_json(drop_id=True))

# -----------------------------------------------------------------------------
# 1c. PANEL (transformaciones idénticas al cuaderno: anomalía vs. normal 1991–2020 y crecimiento logarítmico)
# -----------------------------------------------------------------------------
LLUVIA = embebido(CHIRPS_TXT, 1991, "lluvia")
NORM = LLUVIA[LLUVIA["anio"].between(*NORMAL)].groupby("cod")["lluvia"].mean().rename("normal")
LLUVIA = LLUVIA.merge(NORM, on="cod")
LLUVIA["clima"] = 100 * (LLUVIA["lluvia"] / LLUVIA["normal"] - 1)
AGRO = embebido(AGRO_TXT, 2005, "va").sort_values(["cod", "anio"])
AGRO["agro"] = 100 * AGRO.groupby("cod")["va"].transform(lambda s: np.log(s).diff())
IPM = embebido(IPM_TXT, 2018, "ipm")
PANEL = (IPM.merge(AGRO[["cod", "anio", "agro"]], on=["cod", "anio"])
         .merge(LLUVIA[["cod", "anio", "clima"]], on=["cod", "anio"]).sort_values(["anio", "cod"]).reset_index(drop=True))
PANEL["departamento"] = PANEL["cod"].map(NOMBRE)
ANIOS = [int(a) for a in sorted(PANEL["anio"].unique())]
assert len(PANEL) == N * len(ANIOS) and PANEL[VARS].notna().all().all()

# -----------------------------------------------------------------------------
# 2. MATRICES DE PESOS W (estandarizadas por filas)
# -----------------------------------------------------------------------------
_G = GDF.to_crs(3116)
if GEO_FUENTE == "respaldo":   # puntos exactos calculados sobre la geometría completa
    COORDS = np.array([TOPOLOGIA["puntos"][c] for c in COD], dtype=float)
else:
    _P = _G.geometry.representative_point()
    COORDS = np.column_stack([_P.x, _P.y])


@_cache
def matriz(nombre):
    """(objeto libpysal W, matriz densa N×N). Queen/Rook desde la geometría (o la topología embebida);
    San Andrés (isla) se enlaza con su vecino más cercano. Todas se estandarizan por filas: W·1 = 1,
    radio espectral(W) = 1, lo que hace que |ρ| < 1 sea condición suficiente de invertibilidad de I − ρW."""
    if nombre in ("Queen", "Rook"):
        if GEO_FUENTE == "respaldo":
            idx = {c: i for i, c in enumerate(COD)}
            vec = {idx[c]: [idx[v] for v in vs] for c, vs in TOPOLOGIA[nombre].items()}
            w = PesosW(vec, silence_warnings=True)
        else:
            w = (Queen if nombre == "Queen" else Rook).from_dataframe(_G, use_index=False, silence_warnings=True)
        if w.islands:
            w = attach_islands(w, KNN.from_array(COORDS, k=1))
    elif nombre.startswith("KNN"):
        w = KNN.from_array(COORDS, k=int(nombre[3:]))
    else:
        w = DistanceBand(COORDS, threshold=min_threshold_distance(COORDS) * 1.001, binary=False, alpha=-1.0,
                         silence_warnings=True)
    w.transform = "r"
    W = np.zeros((N, N))
    for i, vec in w.neighbors.items():
        W[i, vec] = w.weights[i]
    assert np.isfinite(W).all() and np.allclose(W.sum(1), 1)
    return w, W


@_cache
def tabla_moran():
    np.random.seed(42)                       # permutaciones reproducibles (esda usa el RNG global)
    filas = []
    for m in MATRICES:
        w, _ = matriz(m)
        for v in VARS:
            res = [Moran(PANEL[PANEL["anio"] == a].sort_values("cod")[v].values, w, permutations=199) for a in ANIOS]
            filas.append({"Matriz": m, "Variable": CORTO[v], "I de Moran": np.mean([r.I for r in res]),
                          "% años significativos": 100 * np.mean([r.p_sim < ALPHA for r in res])})
    return pd.DataFrame(filas)


def mejor_W():
    t = tabla_moran()
    res = t.groupby("Matriz").agg(I=("I de Moran", "mean"), sig=("% años significativos", lambda s: (s >= 50).sum()))
    eleg = res[res["sig"] >= 2]
    return (eleg if len(eleg) else res)["I"].idxmax()


# -----------------------------------------------------------------------------
# 3. ESPECIFICACIÓN DEL SpVAR ESTRUCTURAL (recursivo: clima → agro → pobreza)
# -----------------------------------------------------------------------------
#   y_k,t = ρ_k W y_k,t + Σ_{j<k} (β_kj y_j,t + θ_kj W y_j,t) + Σ_j (γ_kj y_j,t−1 + δ_kj W y_j,t−1) + μ_k,i + τ_k,t + v_k,t
#   v_k,t = λ_k W v_k,t + ε_k,t          (componente de error espacial en SEM/SDEM/SAC)
ESPEC = {"clima": ["clima_L1"], "agro": ["agro_L1", "clima", "clima_L1", "ipm_L1"],
         "ipm": ["ipm_L1", "agro", "clima", "agro_L1", "clima_L1"]}
# Endogeneidad contemporánea: en Pobreza, Agro(t) (y su W·Agro(t) en SDM) se determina en el mismo año que
# el IPM, con innovaciones posiblemente correlacionadas → se instrumenta. La lluvia la determina la atmósfera:
# es estrictamente exógena. Los rezagos t−1 son predeterminados (sesgo de Nickell de orden 1/T, ver nota).
ENDOG_CONTEMP = {"ipm": ("agro",)}
# Opción avanzada: instrumentar también los rezagos con niveles fechados t−1 y t−2 (Arellano-Bover). Con
# T = 7 y N = 33 esos instrumentos resultan débiles (F de 1ª etapa < 10) y la varianza explota, por eso viene
# apagada; actívala solo para un análisis de sensibilidad.
INSTRUMENTAR_REZAGOS = os.environ.get("SPVAR_NICKELL_IV", "0") == "1"
NIVELES = {"clima_L1": ["clima_L1", "clima_L2"], "agro_L1": ["agro_L1", "agro_L2"], "ipm_L1": ["ipm_L1"]}


def es_exogena(r, eq):
    if r in ENDOG_CONTEMP.get(eq, ()):
        return False
    if INSTRUMENTAR_REZAGOS and r.endswith("_L1") and r != "clima_L1":
        return False
    return True


CON_LAG, CON_ERR, CON_SLX = ("SAR", "SDM", "SAC"), ("SEM", "SDEM", "SAC"), ("SLX", "SDM", "SDEM")
ESTIMADOR = {False: "S2SLS espacial (White)", True: "GMM-KP het. (GS2SLS)"}


def _fod(df, cols):
    """Desviaciones ortogonales hacia adelante (Arellano-Bover 1995), x*_t = c_t [x_t − media(x_{t+1..T})],
    c_t = √((T−t)/(T−t+1)). Es una transformación ortonormal F′ con F′F = I_{T−1} y FF′ = J_T (matriz de
    demeaning): elimina el efecto fijo y, a diferencia del Within, deja errores i.i.d. con N(T−1) datos
    (Lee y Yu, 2010). Como actúa solo en el tiempo, W_NT = I_{T−1} ⊗ W se conserva exactamente."""
    partes = []
    for c, g in df.sort_values(["cod", "anio"]).groupby("cod", sort=False):
        X = g[cols].to_numpy(float)
        T = len(X)
        Z = np.array([np.sqrt((T - t - 1) / (T - t)) * (X[t] - X[t + 1:].mean(0)) for t in range(T - 1)])
        partes.append(pd.DataFrame(Z, columns=cols).assign(cod=c, anio=g["anio"].to_numpy()[:-1]))
    return pd.concat(partes, ignore_index=True)


@_cache
def datos_panel(efectos):
    """Panel transformado (Lee-Yu) listo para estimar, con instrumentos en niveles y dummies de periodo."""
    base = PANEL[["cod", "anio"] + VARS].copy()
    for df, col in ((LLUVIA, "clima"), (AGRO, "agro"), (IPM, "ipm")):
        for k in (1, 2):
            base = base.merge(df[["cod", "anio", col]].assign(anio=df["anio"] + k).rename(columns={col: f"{col}_L{k}"}),
                              on=["cod", "anio"], how="left")
    L1 = [f"{v}_L1" for v in VARS]
    D = base.dropna(subset=VARS + L1).sort_values(["cod", "anio"]).reset_index(drop=True)
    T = D["anio"].nunique()
    F = _fod(D, VARS + L1)
    niv = sorted({c for v in NIVELES.values() for c in v})
    F = F.merge(D[["cod", "anio"] + niv].rename(columns={c: f"{c}_niv" for c in niv}), on=["cod", "anio"])
    F = F.sort_values(["anio", "cod"]).reset_index(drop=True)
    periodos = sorted(F["anio"].unique())
    assert F[[f"{c}_niv" for c in niv]].notna().all().all()
    assert all(F.loc[F["anio"] == a, "cod"].tolist() == COD for a in periodos)
    dums = []
    if efectos == "Departamento + año":
        # Efectos de año en el espacio transformado: FOD(τ_t ⊗ 1_N) = τ*_t ⊗ 1_N → una dummy por periodo
        # transformado (se omite una por la constante). Exacto, sin el doble demeaning del original.
        for a in periodos[1:]:
            F[f"T_{a}"] = (F["anio"] == a).astype(float)
            dums.append(f"T_{a}")
    return F, T, len(periodos), tuple(dums)


def _rango(M):
    return np.linalg.matrix_rank(M, tol=1e-8 * max(1.0, np.abs(M).max()) * max(M.shape)) if M.size else 0


def _depurar_instrumentos(X, Q, nombres):
    """Quita instrumentos colineales con [1, X, Q_ya_elegidos] (evita H′H singular)."""
    n = X.shape[0]
    Hs = np.column_stack([np.ones(n), X])
    r0, sel = _rango(Hs), []
    for j in range(Q.shape[1]):
        cand = np.column_stack([Hs, Q[:, j]])
        r = _rango(cand)
        if r > r0:
            Hs, r0 = cand, r
            sel.append(j)
    return Q[:, sel], [nombres[j] for j in sel]


def _seleccion(F, eq, w_nt, dums, extra=()):
    """Árbol LM robusto (Anselin et al. 1996) + LR del bloque SLX + AIC/BIC de la verosimilitud transformada de
    Lee-Yu (n = N(T−1), k incluye dummies). Es una etapa de cribado: la estimación final es por VI."""
    xn = list(ESPEC[eq]) + list(extra) + list(dums)
    y, X = F[[eq]].values, F[xn].values
    mascara = [n not in dums for n in xn]           # W·dummy = dummy: nunca se rezaga espacialmente
    # (los residuos de control FC_* sí se rezagan: W·v̂ controla la endogeneidad de W·Agro en el SDM)
    kw = dict(w=w_nt, name_y=eq, name_x=xn)
    m = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for nom, f in {"MCO": lambda: OLS(y, X, spat_diag=True, **kw),
                       "SLX": lambda: OLS(y, X, slx_lags=1, slx_vars=mascara, **kw),
                       "SAR": lambda: ML_Lag(y, X, method="full", spat_impacts=None, **kw),
                       "SEM": lambda: ML_Error(y, X, method="full", **kw),
                       "SDM": lambda: ML_Lag(y, X, slx_lags=1, slx_vars=mascara, method="full", spat_impacts=None, **kw),
                       "SDEM": lambda: ML_Error(y, X, slx_lags=1, slx_vars=mascara, method="full", **kw)}.items():
            try:
                m[nom] = f()
            except Exception:
                pass
    n = len(y)
    ic = {k: (-2 * float(mod.logll) + 2 * (len(np.ravel(mod.betas)) + 1),
              -2 * float(mod.logll) + (len(np.ravel(mod.betas)) + 1) * np.log(n))
          for k, mod in m.items() if hasattr(mod, "logll")}
    o = m["MCO"]
    s = lambda t: t[1] < ALPHA
    if not (s(o.lm_lag) or s(o.lm_error)): ruta = "MCO"
    elif s(o.lm_lag) and not s(o.lm_error): ruta = "SAR"
    elif s(o.lm_error) and not s(o.lm_lag): ruta = "SEM"
    elif s(o.rlm_lag) and s(o.rlm_error): ruta = "SAC"
    else: ruta = "SAR" if o.rlm_lag[0] > o.rlm_error[0] else "SEM"
    p_slx = stats.chi2.sf(2 * (float(m["SLX"].logll) - float(o.logll)), len(ESPEC[eq])) if "SLX" in m else 1.0
    if p_slx < ALPHA and ruta != "SAC":
        ruta = {"MCO": "SLX", "SAR": "SDM", "SEM": "SDEM"}[ruta]
    aic = min(ic, key=lambda k: ic[k][0]); bic = min(ic, key=lambda k: ic[k][1])
    eleg = ruta if (ruta in (aic, bic) or ruta == "SAC") else bic
    orden = [eleg] + [k for k in sorted(ic, key=lambda k: ic[k][1]) if k != eleg]
    return dict(ruta=ruta, aic=aic, bic=bic, eleg=eleg, orden=orden, modelos=m)


def _estimar_vi(F, eq, est, WNT, w_nt, dums):
    """Estimación de una ecuación con la estructura `est` por variables instrumentales espaciales.
      · Endógenas: W·y(t) (SAR/SDM/SAC) y Agro(t) [+ W·Agro(t)] en la ecuación de Pobreza.
      · Instrumentos excluidos (Kelejian-Prucha 1998; Anselin): W·X y W²·X de los regresores exógenos
        (W²·X y W³·X cuando W·X ya es regresor, SDM/SDEM), depurados de colinealidad.
      · Sin error espacial → spreg.TSLS (S2SLS, errores White); con error → spreg.GM_Endog_Error_Het
        (GS2SLS / GMM de Kelejian-Prucha robusto a heterocedasticidad). Sin endógenas → OLS (White) o
        spreg.GM_Error_Het."""
    lag, err, slx = est in CON_LAG, est in CON_ERR, est in CON_SLX
    R = ESPEC[eq]
    Xc, Xn, Ec, En = [], [], [], []

    def agregar(nombre, vec, exog):
        (Xc if exog else Ec).append(np.asarray(vec, float)); (Xn if exog else En).append(nombre)

    for r in R:
        agregar(r, F[r].values, es_exogena(r, eq))
    if slx:
        for r in R:
            agregar(f"W_{r}", WNT @ F[r].values, es_exogena(r, eq))
    for d in dums:
        agregar(d, F[d].values, True)
    if lag:
        agregar(f"W_{eq}", WNT @ F[eq].values, False)
    n = len(F)
    X = np.column_stack(Xc) if Xc else np.empty((n, 0))
    y = F[[eq]].values
    kw = dict(name_y=eq, name_x=Xn, name_w="W_NT")
    f1 = sargan = ak = np.nan
    kq = 0
    with contextlib.redirect_stdout(io.StringIO()):
        if Ec:                                                   # ---- VI espacial ----
            E = np.column_stack(Ec)
            base = [(r, F[r].values) for r in R if es_exogena(r, eq)]
            base += [(f"{c}_niv", F[f"{c}_niv"].values) for c in dict.fromkeys(
                c for r in R if not es_exogena(r, eq) for c in NIVELES.get(r, []))]
            Qc, Qn = [v for k, v in base if k.endswith("_niv")], [k for k in dict(base) if k.endswith("_niv")]
            for p in range(1, 3 + int(slx)):
                for nom, v in base:
                    if slx and p == 1 and not nom.endswith("_niv"):
                        continue                                 # W·X exógena ya es regresor
                    Wv = v
                    for _ in range(p):
                        Wv = WNT @ Wv
                    Qc.append(Wv); Qn.append(f"W{p}_{nom}")
            Q, Qn = _depurar_instrumentos(X, np.column_stack(Qc), Qn)
            if Q.shape[1] < E.shape[1]:
                raise ValueError("Ecuación no identificada: menos instrumentos que endógenas")
            kw.update(name_yend=En, name_q=Qn)
            if err:
                m = GM_Endog_Error_Het(y, X, yend=E, q=Q, w=w_nt, **kw)
                e = np.ravel(m.e_filtered)
            else:
                m = TSLS(y, X, yend=E, q=Q, w=w_nt, robust="white", spat_diag=True, **kw)
                e = np.ravel(m.u)
                ak = float(m.ak_test[1])
            z, h = np.asarray(m.z, float), np.asarray(m.h, float)
            kq = Q.shape[1]
        else:                                                    # ---- sin endógenas ----
            if err:
                m = GM_Error_Het(y, X, w=w_nt, **kw)
                e = np.ravel(m.e_filtered)
            else:
                m = OLS(y, X, w=w_nt, robust="white", spat_diag=True, **kw)
                e = np.ravel(m.u)
                ak = float(m.lm_error[1])                        # LM-error sobre residuos MCO
            z = h = np.asarray(m.x, float)
    nombres = list(getattr(m, "name_z", None) or m.name_x)
    if err and "lambda" not in nombres:
        nombres.append("lambda")
    nombres = ["rho" if k == f"W_{eq}" else k for k in nombres]
    b = np.ravel(m.betas).astype(float)
    V = np.asarray(m.vm, float)
    if not (len(nombres) == len(b) and np.isfinite(b).all() and np.isfinite(V).all()):
        raise ValueError("estimación no finita")
    # Influencia de β̂ (covarianza conjunta entre ecuaciones, tipo SUR): β̂ − β ≈ (Ẑ′Z_s)⁻¹ Ẑ′ε,
    # con Z_s = (I − λW) Z en modelos con error espacial y Ẑ = P_H Z_s.
    lam = b[-1] if err else 0.0
    zs = z - lam * (WNT @ z)
    zh = h @ np.linalg.lstsq(h, zs, rcond=None)[0]
    psi = (e[:, None] * zh) @ np.linalg.inv(zh.T @ zs).T
    if err:
        psi = np.column_stack([psi, np.zeros(n)])
    if Ec:   # F de primera etapa (Staiger-Stock, mínimo entre endógenas) y Sargan de sobreidentificación
        H0 = np.column_stack([np.ones(n), X])
        rss = lambda A, v: float(((v - A @ np.linalg.lstsq(A, v, rcond=None)[0]) ** 2).sum())
        f1 = min(((rss(H0, E[:, j]) - rss(h, E[:, j])) / kq) / (rss(h, E[:, j]) / (n - h.shape[1]))
                 for j in range(E.shape[1]))
        gl = kq - E.shape[1]
        r2 = 1 - rss(h, e) / float(((e - e.mean()) ** 2).sum())
        sargan = float(stats.chi2.sf(n * r2, gl)) if gl > 0 else np.nan
    return dict(m=m, est=est, metodo="VI", b=pd.Series(b, index=nombres), V=pd.DataFrame(V, index=nombres, columns=nombres),
                e=e, psi=psi, sig2=float(e @ e / n), f1=float(f1), sargan=sargan, ak=ak, hausman=np.nan,
                n_q=kq, n_en=len(Ec), recortado=bool(err and lag and abs(abs(b[nombres.index("rho")]) - 0.9) < 1e-12))


def _instrumentos_espaciales(F, cols, WNT, potencias=(1, 2)):
    out = []
    for c in cols:
        v = F[c].values
        for p in range(1, max(potencias) + 1):
            v = WNT @ v
            if p in potencias:
                out.append(v)
    return np.column_stack(out)


def _funcion_control(F, eq, WNT, dums):
    """Primera etapa de la función de control (Hausman / Rivers-Vuong): Agro*(t) sobre [1, X, dummies, W·X, W²·X]
    con X = regresores exógenos de la ecuación de Pobreza. Devuelve los residuos v̂ (que se añaden como
    regresor a la QML: su coeficiente corrige la endogeneidad y su t es la prueba de Hausman) y el F de
    los instrumentos excluidos."""
    res = {}
    for v in ENDOG_CONTEMP.get(eq, ()):
        exo = [r for r in ESPEC[eq] if es_exogena(r, eq)]
        n = len(F)
        H0 = np.column_stack([np.ones(n)] + [F[c].values for c in exo + list(dums)])
        Q, _ = _depurar_instrumentos(H0[:, 1:], _instrumentos_espaciales(F, exo, WNT), list(range(2 * len(exo))))
        H = np.column_stack([H0, Q])
        x = F[v].values
        uh = x - H @ np.linalg.lstsq(H, x, rcond=None)[0]
        u0 = x - H0 @ np.linalg.lstsq(H0, x, rcond=None)[0]
        f1 = ((u0 @ u0 - uh @ uh) / Q.shape[1]) / (uh @ uh / (n - H.shape[1]))
        res[f"FC_{v}"] = (uh, float(f1))
    return res


def _desde_ml(m, est, eq, F, WNT, f1=np.nan):
    """Resultado estándar a partir de la QML de Lee-Yu (spreg.ML_Lag / ML_Error / OLS) del paso de selección."""
    lag, err = est in CON_LAG, est in CON_ERR
    nombres = ["rho" if k == f"W_{eq}" else k for k in m.name_x]
    b = np.ravel(m.betas).astype(float)
    V = np.asarray(m.vm, float)[:len(b), :len(b)]
    e = np.ravel(m.e_filtered if err else m.u)
    x = np.asarray(m.x, float)
    z = np.column_stack([x, WNT @ F[eq].values]) if lag else x
    h = np.column_stack([x, _depurar_instrumentos(x[:, 1:], np.column_stack([WNT @ x[:, 1:], WNT @ (WNT @ x[:, 1:])]),
                                                  list(range(2 * (x.shape[1] - 1))))[0]]) if lag else x
    lam = b[-1] if err else 0.0
    zs = z - lam * (WNT @ z)
    zh = h @ np.linalg.lstsq(h, zs, rcond=None)[0]
    psi = (e[:, None] * zh) @ np.linalg.pinv(zh.T @ zs).T
    if err:
        psi = np.column_stack([psi, np.zeros(len(e))])
    fc = [k for k in nombres if k.startswith("FC_")]
    haus = (float(2 * stats.norm.sf(abs(b[nombres.index(fc[0])] / np.sqrt(V[nombres.index(fc[0])] [nombres.index(fc[0])]))))
            if fc else np.nan)
    ak = float(m.lm_error[1]) if hasattr(m, "lm_error") else np.nan
    return dict(m=m, est=est, metodo="QML", b=pd.Series(b, index=nombres), V=pd.DataFrame(V, index=nombres, columns=nombres),
                e=e, psi=psi, sig2=float(e @ e / len(e)), f1=f1, sargan=np.nan, ak=ak, hausman=haus, n_q=0, n_en=0,
                recortado=False)


def _sistema(P, W):
    """A0, A1, B y S = (I − A0)⁻¹ del sistema 3N × 3N; Φ = S A1 (dinámica del VAR en forma reducida)."""
    I, Z = np.eye(N), np.zeros((N, N))
    g = lambda eq, nom: float(P[eq].get(nom, 0.0))
    A0 = [[Z] * 3 for _ in range(3)]; A1 = [[Z] * 3 for _ in range(3)]
    for k, ek in enumerate(VARS):
        for j, ej in enumerate(VARS):
            if j == k:   A0[k][j] = g(ek, "rho") * W                               # autocorrelación espacial
            elif j < k:  A0[k][j] = g(ek, ej) * I + g(ek, f"W_{ej}") * W           # cruce local + derrame
            A1[k][j] = g(ek, f"{ej}_L1") * I + g(ek, f"W_{ej}_L1") * W
    A0, A1 = np.block(A0), np.block(A1)
    B = np.block([[np.linalg.inv(I - g(ek, "lambda") * W) if j == k else Z for j in range(3)] for k, ek in enumerate(VARS)])
    S = np.linalg.inv(np.eye(3 * N) - A0)
    Phi = S @ A1
    return dict(A0=A0, A1=A1, S=S, B=B, Phi=Phi, rA0=float(np.abs(np.linalg.eigvals(A0)).max()),
                rPhi=float(np.abs(np.linalg.eigvals(Phi)).max()))


def verificar_neumann(A0, S, tol=1e-12, max_m=2000):
    """Con max|eig(A0)| < 1 la serie de Neumann Σ_{m≥0} A0^m converge a (I − A0)⁻¹. Devuelve (error máx, m)."""
    suma, Pm = np.eye(len(A0)), np.eye(len(A0))
    for m in range(1, max_m + 1):
        Pm = Pm @ A0
        suma += Pm
        if np.abs(Pm).max() < tol:
            break
    return float(np.abs(S - suma).max()), m


def impactos_lesage_pace(S_kj):
    """LeSage y Pace (2009) sobre el bloque S_kj = ∂y_k/∂ε_j′ del multiplicador:
    Directo = tr(S_kj)/N · Total = 1′S_kj1/N · Indirecto (derrame) = Total − Directo."""
    d, t = np.trace(S_kj) / N, S_kj.sum() / N
    return np.array([d, t - d, t])


def _resumen(M):          # alias retrocompatible
    return impactos_lesage_pace(M)


def _psd(V):
    w_, U = np.linalg.eigh((V + V.T) / 2)
    return (U * np.clip(w_, 0, None)) @ U.T


@_cache
def modelo(W_nombre, efectos):
    """Selecciona la estructura espacial de cada ecuación (Lee-Yu + LM robusto + AIC/BIC + AK) y la estima:
       1º VI espacial (S2SLS / GMM-KP); se acepta si es admisible: |ρ|, |λ| < 0.98, sin recorte de spreg y con
          instrumentos fuertes (F de 1ª etapa ≥ 10, regla de Staiger-Stock).
       2º Si los instrumentos son débiles, QML de Lee-Yu (identifica ρ por el Jacobiano log|I − ρW|, sin depender de
          la fuerza de X; Lee 2004) con función de control para Agro(t): la endogeneidad sigue corregida.
    Luego arma el sistema 3N × 3N, impactos LeSage-Pace con IC Monte Carlo conjuntos y Sp-IRF ortogonalizadas."""
    w, W = matriz(W_nombre)
    F, T, Tt, dums = datos_panel(efectos)
    WNT = sparse.kron(sparse.identity(Tt), sparse.csr_matrix(W), format="csr")
    w_nt = WSP(WNT).to_W(silence_warnings=True)
    w_nt.transform = "r"

    I_N = np.eye(N)

    def radio_propio(b, eq):
        """Estabilidad dinámica del bloque propio (Yu, de Jong y Lee 2008): max|eig((I − ρW)⁻¹(γI + δW))| < 1."""
        Bk = np.linalg.solve(I_N - b.get("rho", 0.0) * W, b.get(f"{eq}_L1", 0.0) * I_N + b.get(f"W_{eq}_L1", 0.0) * W)
        return float(np.abs(np.linalg.eigvals(Bk)).max())

    def admisible(r, eq):
        return (r is not None and np.isfinite(r["b"].values).all() and abs(r["b"].get("rho", 0)) < 0.98
                and abs(r["b"].get("lambda", 0)) < 0.98 and not r["recortado"]
                and (r["metodo"] == "QML" or r["n_en"] == 0 or r["f1"] >= 10) and radio_propio(r["b"], eq) < 1)

    sel, RES = [], {}
    for eq in VARS:
        FC = _funcion_control(F, eq, WNT, dums)
        Fq = F.assign(**{k: v for k, (v, _) in FC.items()}) if FC else F
        f1_fc = min((f for _, f in FC.values()), default=np.nan)
        sc = _seleccion(Fq, eq, w_nt, dums, extra=tuple(FC))
        r, nota = None, ""
        for cand in sc["orden"]:
            opciones = []
            with contextlib.suppress(Exception):
                iv = _estimar_vi(F, eq, cand, WNT, w_nt, dums)
                # Prueba AK (Anselin-Kelejian 1997) sobre residuos VI: si persiste correlación espacial, se añade λ.
                sube = {"MCO": "SEM", "SLX": "SDEM", "SAR": "SAC"}.get(cand)
                if sube and iv["ak"] == iv["ak"] and iv["ak"] < ALPHA:
                    with contextlib.suppress(Exception):
                        opciones.append((_estimar_vi(F, eq, sube, WNT, w_nt, dums), f"AK p={iv['ak']:.3f} → {sube}"))
                opciones.append((iv, ""))
            for o, txt in opciones:
                if admisible(o, eq):
                    r, nota = o, txt
                    break
            if r is not None:
                break
            if cand in sc["modelos"]:
                with contextlib.suppress(Exception):
                    q = _desde_ml(sc["modelos"][cand], cand, eq, Fq, WNT, f1_fc)
                    if admisible(q, eq):
                        iv0 = opciones[-1][0] if opciones else None
                        motivo = (f"VI débil (F={iv0['f1']:.1f})" if iv0 is not None and iv0["f1"] < 10
                                  else "VI fuera de la región estable")
                        r, nota = q, f"{motivo} → QML Lee-Yu" + (" + FC" if FC else "")
                        break
        if r is None:                                   # último recurso: MCO de Lee-Yu (siempre existe)
            r, nota = _desde_ml(sc["modelos"]["MCO"], "MCO", eq, Fq, WNT, f1_fc), "sin estructura admisible → MCO"
        RES[eq] = r
        b = r["b"]
        if r["metodo"] == "QML":
            estim = {"MCO": "MCO Lee-Yu (White)", "SLX": "MCO Lee-Yu (White)"}.get(r["est"], "QML Lee-Yu") + (" + FC" if FC else "")
        elif r["n_en"] == 0:
            estim = "GMM-KP het." if r["est"] in CON_ERR else "MCO (White)"
        else:
            estim = ESTIMADOR[r["est"] in CON_ERR]
        haus = r["hausman"]
        if r["metodo"] == "VI" and hasattr(r["m"], "dwh"):
            with contextlib.suppress(Exception):
                haus = float(np.ravel(r["m"].dwh)[1])
        rnd = lambda v, k=3: round(float(v), k) if v == v else None
        sel.append({"Ecuación": CORTO[eq], "Árbol LM": sc["ruta"], "AIC": sc["aic"], "BIC": sc["bic"], "Elegido": r["est"],
                    "Estimador": estim, "ρ": round(float(b.get("rho", 0)), 3), "λ": round(float(b.get("lambda", 0)), 3),
                    "F 1ª etapa": rnd(r["f1"], 1), "Sargan p": rnd(r["sargan"]), "AK p": rnd(r["ak"]),
                    "Hausman p": rnd(haus), "Nota": nota})
    PAR = {eq: (RES[eq]["b"], RES[eq]["V"]) for eq in VARS}
    P0 = {eq: PAR[eq][0] for eq in VARS}
    SIS = _sistema(P0, W)
    SIS["neumann"] = verificar_neumann(SIS["A0"], SIS["S"]) if SIS["rA0"] < 1 else (np.inf, 0)
    assert SIS["S"].shape == (3 * N, 3 * N) and np.isfinite(SIS["S"]).all()
    # --- Correlación contemporánea de innovaciones (tipo SUR) y ortogonalización de Cholesky ---
    Ein = np.column_stack([RES[eq]["e"] for eq in VARS])
    SIG = Ein.T @ Ein / len(Ein)
    CORR = SIG / np.sqrt(np.outer(np.diag(SIG), np.diag(SIG)))
    Lc = np.linalg.cholesky(SIG)
    Lt = Lc / np.diag(Lc)                      # diagonal unitaria: columna j = choque ortogonal j por unidad
    # --- Covarianza conjunta de los parámetros: bloques propios = vm de spreg; cruzados = sándwich ---
    idx = np.cumsum([0] + [len(PAR[eq][0]) for eq in VARS])
    VJ = np.zeros((idx[-1], idx[-1]))
    for a, ea in enumerate(VARS):
        for c, ec in enumerate(VARS):
            VJ[idx[a]:idx[a + 1], idx[c]:idx[c + 1]] = (PAR[ea][1].values if a == c
                                                        else RES[ea]["psi"].T @ RES[ec]["psi"])
    VJ = _psd(VJ)
    bJ = np.concatenate([PAR[eq][0].values for eq in VARS])

    def imp(P):
        s_ = _sistema(P, W)
        return {(ej, ek): impactos_lesage_pace(s_["S"][k * N:(k + 1) * N, j * N:(j + 1) * N])
                for k, ek in enumerate(VARS) for j, ej in enumerate(VARS)}, s_
    base, _ = imp(P0)
    rng = np.random.default_rng(zlib.crc32(f"{W_nombre}|{efectos}".encode()))   # reproducible por especificación
    draws = []
    for vec in rng.multivariate_normal(bJ, VJ, size=N_MC, method="eigh"):
        Pd = {eq: pd.Series(vec[idx[a]:idx[a + 1]], index=PAR[eq][0].index) for a, eq in enumerate(VARS)}
        try:
            d_, s_ = imp(Pd)
            if s_["rA0"] < 1 and s_["rPhi"] < 1:
                draws.append(d_)
        except np.linalg.LinAlgError:
            pass
    filas = []
    for (ej, ek), val in base.items():
        arr = np.array([d[(ej, ek)] for d in draws]) if draws else np.full((1, 3), np.nan)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        for i_, ef in enumerate(("Directo", "Indirecto", "Total")):
            filas.append({"Choque": CORTO[ej], "Respuesta": CORTO[ek], "Efecto": ef, "Estimado": val[i_],
                          "IC_inf": lo[i_], "IC_sup": hi[i_], "Sig": bool(not (lo[i_] <= 0 <= hi[i_]) and abs(val[i_]) > 1e-12)})
    IMP = pd.DataFrame(filas)
    # --- Mecanismos (coeficientes estructurales) ---
    mec = []
    for eq in VARS:
        b, V = PAR[eq]
        for nom in b.index:
            if nom == "CONSTANT" or nom.startswith("T_") or "FC_" in nom: continue   # constante, años, controles
            base_ = nom[2:] if nom.startswith("W_") else nom
            var = base_.replace("_L1", ""); t = "(t-1)" if base_.endswith("_L1") else "(t)"
            if nom in ("rho", "lambda") or var == eq:
                grupo = "Autodinámica"
                lab = {"rho": f"ρ · W·{CORTO[eq]}(t)", "lambda": "λ · W·v(t)"}.get(nom, ("W·" if nom.startswith("W_") else "") + CORTO[eq] + t)
            elif nom.startswith("W_"):
                grupo, lab = "Espacial cruzada", f"W·{CORTO[var]}{t}"
            else:
                grupo, lab = "Cruce local", f"{CORTO[var]}{t}"
            se = np.sqrt(V.loc[nom, nom]) if V.loc[nom, nom] > 0 else np.nan
            mec.append({"Ecuación": CORTO[eq], "Mecanismo": grupo, "Término": lab, "coef": float(b[nom]), "ee": se,
                        "p": 2 * stats.norm.sf(abs(b[nom] / se)) if se == se else np.nan})
    MEC = pd.DataFrame(mec)
    # --- Sp-IRF ortogonalizadas: R[h] = Φ^h · S · B · (L̃ ⊗ I_N) ---
    sd = {eq: float(Lc[i, i]) for i, eq in enumerate(VARS)}       # d.e. del choque ortogonal
    Psi, R, Pm = SIS["S"] @ SIS["B"] @ np.kron(Lt, np.eye(N)), [], np.eye(3 * N)
    for _ in range(H + 1):
        R.append(Pm @ Psi); Pm = SIS["Phi"] @ Pm
    SB = SIS["S"] @ SIS["B"]
    feed = {k: np.trace(SB[i * N:(i + 1) * N, i * N:(i + 1) * N]) / N - 1 for i, k in enumerate(VARS)}
    spill = {k: 1 - np.trace(SB[i * N:(i + 1) * N, i * N:(i + 1) * N]) / SB[i * N:(i + 1) * N, i * N:(i + 1) * N].sum()
             for i, k in enumerate(VARS)}
    return dict(W_nombre=W_nombre, efectos=efectos, T=T, n=len(F), SEL=pd.DataFrame(sel), PAR=PAR, SIS=SIS, IMP=IMP,
                MEC=MEC, R=np.array(R), sd=sd, feed=feed, spill=spill, n_draws=len(draws), CORR=CORR,
                DIAG={eq: {k: RES[eq][k] for k in ("f1", "sargan", "ak", "n_q", "n_en")} for eq in VARS})


@_cache
def lisa(W_nombre, var):
    w, _ = matriz(W_nombre)
    medias = PANEL.groupby("cod")[var].mean().reindex(COD)
    l = Moran_Local(medias.values, w, permutations=999, seed=42)
    lab = {1: "Alto-Alto", 2: "Bajo-Alto", 3: "Bajo-Bajo", 4: "Alto-Bajo"}
    return pd.DataFrame({"cod": COD, "departamento": [NOMBRE[c] for c in COD], "valor": medias.values,
                         "cluster": [lab[q] if p < ALPHA else "No significativo" for q, p in zip(l.q, l.p_sim)]})


# =============================================================================
# MÓDULO 2 · VISUALIZACIÓN AGNÓSTICA (Plotly → go.Figure, sirve en gr.Plot y st.plotly_chart)
# =============================================================================
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONDO = "rgba(0,0,0,0)"
ESTILO_MAPA = os.environ.get("MAP_STYLE", "carto-darkmatter")      # mapa base oscuro (sin token)


def _px_map(df, **kw):
    """Mapa coroplético sobre mapa base: usa MapLibre si la versión de Plotly lo trae, si no Mapbox (sin token)."""
    if hasattr(px, "choropleth_map"):
        return px.choropleth_map(df, **kw)
    kw["mapbox_style"] = kw.pop("map_style")
    return px.choropleth_mapbox(df, **kw)


COL_LISA = {"Alto-Alto": "#ef4444", "Bajo-Bajo": "#3b82f6", "Alto-Bajo": "#f59e0b", "Bajo-Alto": "#93c5fd",
            "No significativo": "#334155"}


def _estilo(fig, alto=520):
    fig.update_layout(template="plotly_dark", paper_bgcolor=FONDO, plot_bgcolor=FONDO, height=alto,
                      font=dict(family="Inter, sans-serif", color="#e2e8f0"), margin=dict(l=10, r=10, t=50, b=10),
                      hoverlabel=dict(bgcolor="#0f172a", font_size=13))
    return fig


def _mapa(df, col, titulo, escala, rango=None, centro=None, animar=None, etiqueta=None, hover=None):
    kw = dict(geojson=GJ, locations="cod", featureidkey="properties.cod", color=col, hover_name="departamento",
              hover_data=hover,
              color_continuous_scale=escala, map_style=ESTILO_MAPA, center={"lat": 4.6, "lon": -73.6},
              zoom=4.15, opacity=0.82, labels={col: etiqueta or col})
    if rango is not None: kw["range_color"] = rango
    if centro is not None: kw["color_continuous_midpoint"] = centro
    if animar: kw["animation_frame"] = animar
    fig = _px_map(df, **kw)
    fig.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.6)
    fig.update_layout(title=dict(text=titulo, x=0.01, font=dict(size=16)),
                      coloraxis_colorbar=dict(thickness=12, len=0.6, title=""))
    return _estilo(fig, 600)


def fig_explorar(var, anio):
    d = PANEL[PANEL["anio"] == anio]
    lim = np.abs(PANEL[var]).quantile(0.95)
    if var == "clima":
        return _mapa(d, var, f"{ETQ[var]} · {anio}", "BrBG", (-lim, lim), etiqueta="%")
    return _mapa(d, var, f"{ETQ[var]} · {anio}", ESCALA[var], (PANEL[var].quantile(0.02), PANEL[var].quantile(0.98)), etiqueta="%")


def fig_serie(var, cod):
    fig = go.Figure()
    if cod == "todo":
        base = LLUVIA if var == "clima" else PANEL
        g = base.groupby("anio")[var]
        lo, hi, me = g.quantile(0.1), g.quantile(0.9), g.mean()
        fig.add_scatter(x=hi.index, y=hi.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=lo.index, y=lo.values, mode="lines", fill="tonexty", fillcolor="rgba(148,163,184,0.18)", line=dict(width=0),
                        name="Rango p10–p90 (33 dptos)")
        fig.add_scatter(x=me.index, y=me.values, name="Promedio nacional", mode="lines+markers",
                        line=dict(color=COLOR[var], width=3))
        fig.update_layout(title="Colombia: promedio y dispersión entre departamentos", yaxis_title=ETQ[var],
                          legend=dict(orientation="h", y=-0.2))
        return _estilo(fig, 380)
    if var == "clima":
        d = LLUVIA[LLUVIA["cod"] == cod]; nac = LLUVIA.groupby("anio")["clima"].mean()
        fig.add_bar(x=d["anio"], y=d["clima"], name=NOMBRE[cod],
                    marker_color=np.where(d["clima"] >= 0, "#38bdf8", "#f59e0b"))
        fig.add_scatter(x=nac.index, y=nac.values, name="Colombia", line=dict(color="white", dash="dot"))
    else:
        d = PANEL[PANEL["cod"] == cod]; nac = PANEL.groupby("anio")[var].mean()
        fig.add_scatter(x=d["anio"], y=d[var], name=NOMBRE[cod], mode="lines+markers",
                        line=dict(color=COLOR[var], width=3), fill="tozeroy", fillcolor="rgba(255,255,255,0.04)")
        fig.add_scatter(x=nac.index, y=nac.values, name="Promedio 33 dptos", line=dict(color="white", dash="dot"))
    fig.update_layout(title=f"{NOMBRE[cod]} frente al país", yaxis_title=ETQ[var], legend=dict(orientation="h", y=-0.2))
    return _estilo(fig, 380)


def ficha(var, cod, anio):
    d = PANEL[PANEL["anio"] == anio].set_index("cod")
    if cod == "todo":
        filas = "".join(f"| {ETQ[v]} | **{d[v].mean():.1f}** | {NOMBRE[d[v].idxmin()]} ({d[v].min():.1f}) | "
                        f"{NOMBRE[d[v].idxmax()]} ({d[v].max():.1f}) |\n" for v in VARS)
        l = lisa(mejor_W(), var)
        aa = ", ".join(l.loc[l["cluster"] == "Alto-Alto", "departamento"]) or "ninguno"
        return (f"### 🇨🇴 Colombia · {anio}\n| Indicador | Promedio | Mínimo | Máximo |\n|---|---|---|---|\n{filas}"
                f"\n**Clúster LISA alto-alto ({CORTO[var].lower()}):** {aa}")
    rank = int(d[var].rank(ascending=False)[cod])
    l = lisa(mejor_W(), var).set_index("cod").loc[cod, "cluster"]
    return (f"### 📍 {NOMBRE[cod]} · {anio}\n"
            f"| Indicador | Valor |\n|---|---|\n"
            + "".join(f"| {ETQ[v]} | **{d.loc[cod, v]:.1f}** |\n" for v in VARS)
            + f"| Puesto en {CORTO[var].lower()} (de 33) | **{rank}** |\n| Clúster LISA ({CORTO[var].lower()}) | **{l}** |")


def fig_moran(W_sel):
    t = tabla_moran()
    fig = px.bar(t, x="Matriz", y="I de Moran", color="Variable", barmode="group",
                 color_discrete_map={CORTO[v]: COLOR[v] for v in VARS}, hover_data=["% años significativos"])
    fig.add_vrect(x0=MATRICES.index(W_sel) - 0.5, x1=MATRICES.index(W_sel) + 0.5, fillcolor="white", opacity=0.08,
                  line_width=0, annotation_text="seleccionada", annotation_position="top")
    fig.update_layout(title="Autocorrelación espacial media 2018–2025 por matriz W")
    return _estilo(fig, 420)


def fig_lisa(W_sel, var):
    d = lisa(W_sel, var)
    fig = _px_map(d, geojson=GJ, locations="cod", featureidkey="properties.cod", color="cluster",
                  hover_name="departamento", hover_data={"valor": ":.1f", "cod": False},
                  color_discrete_map=COL_LISA, map_style=ESTILO_MAPA,
                  center={"lat": 4.6, "lon": -73.6}, zoom=4.15, opacity=0.85,
                  category_orders={"cluster": list(COL_LISA)})
    fig.update_layout(title=f"LISA · {CORTO[var]} (media 2018–2025 · W {W_sel})", legend_title="")
    return _estilo(fig, 560)


def fig_mecanismos(M):
    d = M["MEC"]
    colores = {"Autodinámica": "#a78bfa", "Cruce local": "#38bdf8", "Espacial cruzada": "#4ade80"}
    eqs = [CORTO[v] for v in VARS]
    sel = M["SEL"].set_index("Ecuación")["Elegido"]
    fig = make_subplots(1, 3, subplot_titles=[f"Ecuación de {e} · {sel[e]}" for e in eqs], horizontal_spacing=0.13)
    vistos = set()
    for c_, e in enumerate(eqs, 1):
        de = d[d["Ecuación"] == e].sort_values("Mecanismo", ascending=False)
        for mec, g in de.groupby("Mecanismo", sort=False):
            sig = g["p"] < ALPHA
            fig.add_scatter(x=g["coef"], y=g["Término"], mode="markers", row=1, col=c_, name=mec,
                            legendgroup=mec, showlegend=mec not in vistos,
                            error_x=dict(type="data", array=1.96 * g["ee"].fillna(0), color=colores[mec], thickness=2),
                            marker=dict(size=13, color=np.where(sig, colores[mec], "#0f172a"),
                                        line=dict(color=colores[mec], width=2)),
                            customdata=np.c_[g["p"].fillna(np.nan)],
                            hovertemplate="%{y}: %{x:.3f}<br>p = %{customdata[0]:.3f}<extra>" + mec + "</extra>")
            vistos.add(mec)
        fig.add_vline(x=0, line_color="rgba(255,255,255,.6)", line_width=1, row=1, col=c_)
    fig.update_xaxes(title="coeficiente (IC 95 %)")
    fig.update_layout(title="Mecanismos de transmisión · relleno = p < 0.05", legend=dict(orientation="h", y=-0.22))
    return _estilo(fig, 470)


def fig_impactos(M):
    d = M["IMP"][M["IMP"]["Efecto"] == "Total"]
    orden = [CORTO[v] for v in VARS]
    Z = d.pivot(index="Respuesta", columns="Choque", values="Estimado").loc[orden, orden]
    Sg = d.pivot(index="Respuesta", columns="Choque", values="Sig").loc[orden, orden]
    txt = np.where(np.abs(Z.values) < 1e-12, "—", np.char.add(np.round(Z.values, 3).astype(str), np.where(Sg.values, " ★", "")))
    lim = np.abs(Z.values).max()
    fig = go.Figure(go.Heatmap(z=Z.values, x=orden, y=orden, text=txt, texttemplate="%{text}", colorscale="RdBu_r",
                               zmin=-lim, zmax=lim, showscale=False, textfont=dict(size=16)))
    fig.update_layout(title="Efecto total contemporáneo S = (I − A0)⁻¹ · ★ = IC 95 % excluye 0",
                      xaxis_title="Choque en", yaxis_title="Respuesta de", yaxis_autorange="reversed")
    return _estilo(fig, 420)


def _bloque(M, var_choque, var_resp, magnitud):
    """R_h[respuesta, origen] (N × N) para el par de variables elegido, ya escalado por la magnitud del choque."""
    k0, k1 = VARS.index(var_choque), VARS.index(var_resp)
    return M["R"][:, k1 * N:(k1 + 1) * N, k0 * N:(k0 + 1) * N] * magnitud * M["sd"][var_choque]   # [h, i, j]


def fig_simulacion(M, var_choque, origen, magnitud, var_resp):
    Rb = _bloque(M, var_choque, var_resp, magnitud)
    div = "RdBu_r" if var_resp == "ipm" else ("BrBG" if var_resp == "clima" else "PRGn")
    x_ = list(range(H + 1))
    linea = go.Figure()
    if origen == "todo":
        # Vista nacional: para cada departamento de origen j, efecto total en Colombia (suma de los 33)
        propio = np.einsum("hjj->hj", Rb)
        total = Rb.sum(axis=1)
        otros = total - propio
        uniforme = Rb.sum(axis=2).mean(1)            # los 33 a la vez: con W estandarizada todos responden igual
        df = pd.DataFrame([{"cod": c, "departamento": NOMBRE[c], "h": f"Año {h}", "total": total[h, i],
                            "en el propio (pp)": round(float(propio[h, i]), 4), "en los otros 32 (pp)": round(float(otros[h, i]), 4)}
                           for h in x_ for i, c in enumerate(COD)])
        lim = max(np.abs(total).max(), 1e-9)
        fig = _mapa(df, "total", f"Efecto en Colombia según dónde cae el choque · {CORTO[var_choque]} → {CORTO[var_resp]}", div, (-lim, lim),
                    animar="h", etiqueta="pp", hover={"en el propio (pp)": True, "en los otros 32 (pp)": True, "cod": False})
        linea.add_scatter(x=x_, y=propio.mean(1), name="En el departamento de origen", mode="lines+markers",
                          line=dict(color="white", width=3))
        linea.add_scatter(x=x_, y=otros.mean(1), name="En los otros 32 (suma)", mode="lines+markers",
                          line=dict(color=COLOR[var_resp], width=3, dash="dash"))
        linea.add_scatter(x=x_, y=uniforme, name="Choque en los 33 a la vez", line=dict(color="#fbbf24", dash="dot"))
        hpk = int(np.argmax(np.abs(total.mean(1))))
        top = np.argsort(-np.abs(total[hpk]))[:3]
        txt = (f"**Lectura nacional:** un choque de **{magnitud:+.1f} d.e.** ({magnitud * M['sd'][var_choque]:+.1f} pp) en "
               f"{CORTO[var_choque].lower()} alcanza su mayor efecto en el **año {hpk}**. Si golpea a los 33 departamentos a la "
               f"vez, la {CORTO[var_resp].lower()} cambia **{uniforme[hpk]:+.3f} pp en todos** (con W estandarizada la respuesta "
               f"es uniforme). Si golpea a uno solo, los orígenes que más mueven al país son **{', '.join(NOMBRE[COD[i]] for i in top)}**: "
               f"en promedio, el {100 * np.abs(otros[hpk]).mean() / max(np.abs(total[hpk]).mean(), 1e-12):.0f}% del efecto cae fuera del origen.")
    else:
        i0 = COD.index(origen)
        resp = Rb[:, :, i0]                                           # [h, i]
        df = pd.DataFrame([{"cod": c, "departamento": NOMBRE[c], "h": f"Año {h}", "respuesta": resp[h, i]}
                           for h in x_ for i, c in enumerate(COD)])
        lim = max(np.abs(resp).max(), 1e-9)
        fig = _mapa(df, "respuesta", f"Propagación: choque de {CORTO[var_choque].lower()} en {NOMBRE[origen]} → "
                    f"{ETQ[var_resp].lower()}", div, (-lim, lim), animar="h", etiqueta="pp")
        _, W = matriz(M["W_nombre"]); vec = W[i0] > 0
        linea.add_scatter(x=x_, y=resp[:, i0], name=f"{NOMBRE[origen]} (origen)", mode="lines+markers",
                          line=dict(color="white", width=3))
        linea.add_scatter(x=x_, y=resp[:, vec].mean(1), name="Vecinos", mode="lines+markers",
                          line=dict(color=COLOR[var_resp], width=3, dash="dash"))
        linea.add_scatter(x=x_, y=resp[:, ~vec & (np.arange(N) != i0)].mean(1), name="Resto del país",
                          line=dict(color="#94a3b8", dash="dot"))
        hpk = int(np.argmax(np.abs(resp[:, i0])))
        fuera = resp[hpk].sum() - resp[hpk, i0]
        txt = (f"**Lectura:** un choque de **{magnitud:+.1f} d.e.** en {CORTO[var_choque].lower()} de **{NOMBRE[origen]}** cambia "
               f"su {CORTO[var_resp].lower()} **{resp[hpk, i0]:+.3f} pp** en el año {hpk}; la suma en el resto del país es "
               f"**{fuera:+.3f} pp**, concentrada en sus vecinos.")
    with contextlib.suppress(Exception):
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400
    linea.add_hline(y=0, line_color="gray")
    linea.update_layout(title="Trayectoria de la respuesta", xaxis_title="Años después del choque", yaxis_title="pp",
                        legend=dict(x=0.99, xanchor="right", y=0.99, bgcolor="rgba(15,23,42,0.7)", font=dict(size=11)))
    return fig, _estilo(linea, 380), txt


# -----------------------------------------------------------------------------
# NARRATIVA (KPIs, estabilidad y conclusiones automáticas)
# -----------------------------------------------------------------------------
def kpis(M):
    b = M["PAR"]["ipm"][0]
    dep = f"ρ = {b.get('rho', 0):.2f}" if abs(b.get("rho", 0)) > 0 else f"λ = {b.get('lambda', 0):.2f}"
    ipm_ult = PANEL[PANEL["anio"] == ANIOS[-1]]["ipm"].mean()
    ipm_ini = PANEL[PANEL["anio"] == ANIOS[0]]["ipm"].mean()
    tarjetas = [("🗺️", "33", "departamentos conectados en un solo sistema"),
                ("📉", f"{ipm_ult:.1f}%", f"pobreza multidimensional promedio {ANIOS[-1]} ({ipm_ult - ipm_ini:+.1f} pp desde {ANIOS[0]})"),
                ("🔗", f"{100 * M['spill']['ipm']:.0f}%", f"del efecto de un choque de pobreza cae en otros departamentos ({dep})"),
                ("🛰️", f"{100 * M['spill']['clima']:.0f}%", "del choque de lluvia se comparte con los vecinos (CHIRPS)")]
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='kpi-ico'>{i}</div><div class='kpi-val'>{v}</div><div class='kpi-lab'>{l}</div></div>"
        for i, v, l in tarjetas) + "</div>"


def texto_estabilidad(M):
    S = M["SIS"]
    ok = lambda x: "✅" if x < 1 else "⚠️"
    err_n, m_n = S["neumann"]
    C = M["CORR"]
    return (f"### 🔒 Estabilidad\n- radio espectral de **A0** = {S['rA0']:.3f} < 1 → S invertible {ok(S['rA0'])}\n"
            f"- serie de Neumann Σ A0^m → S: error máx. {err_n:.1e} con {m_n} términos\n"
            f"- radio espectral de **Φ** = {S['rPhi']:.3f} < 1 → los choques se disipan {ok(S['rPhi'])}\n"
            f"- simulaciones Monte Carlo estables: {M['n_draws']}/{N_MC} (covarianza conjunta entre ecuaciones)\n"
            f"- correlación de innovaciones: Clima–Agro {C[0, 1]:+.2f} · Clima–Pobreza {C[0, 2]:+.2f} · "
            f"Agro–Pobreza {C[1, 2]:+.2f} (ortogonalizadas por Cholesky clima → agro → pobreza)\n"
            f"- muestra: {M['n']} obs. = 33 × {M['n'] // 33} (desviaciones ortogonales de Lee-Yu)")


def conclusiones(M):
    mec, imp = M["MEC"], M["IMP"]
    pv = lambda eq, t: mec.query("Ecuación==@eq and Término==@t")["p"].min()
    c = lambda eq, t: mec.query("Ecuación==@eq and Término==@t")["coef"].sum()
    s = M["SEL"].set_index("Ecuación")["Elegido"]
    ag = imp.query("Choque=='Agro' and Respuesta=='Pobreza' and Efecto=='Indirecto'")["Estimado"].iloc[0]
    f1 = M["DIAG"]["ipm"]["f1"]
    lluvia_agro = pv("Agro", "Clima(t)") < ALPHA
    if f1 >= 10:
        lectura = (f"Estimado por VI (F de 1ª etapa = {f1:.1f}): {'es un efecto identificado.' if pv('Pobreza', 'Agro(t)') < ALPHA else 'no es significativo.'}")
    else:
        lectura = (f"Es una correlación, no un efecto causal: los instrumentos son débiles (F = {f1:.1f} < 10)"
                   + ("" if lluvia_agro else " y la lluvia no predice el agro") + ".")
    return f"""
## 🧭 Lo que el modelo le dice a Colombia
**Especificación:** W = **{M['W_nombre']}** · efectos fijos **{M['efectos'].lower()}** · T = {M['T']} años · estructuras: Clima **{s['Clima']}**, Agro **{s['Agro']}**, Pobreza **{s['Pobreza']}** · estabilidad: radio(Φ) = **{M['SIS']['rPhi']:.2f}** {'< 1 ✅' if M['SIS']['rPhi'] < 1 else '≥ 1 ⚠️'}

1. **La lluvia es regional:** el {100 * M['spill']['clima']:.0f}% de un choque climático se reparte entre vecinos; los sistemas de lluvia no respetan fronteras departamentales.
2. **Lluvia → agro:** {c('Agro', 'Clima(t)') * 10:+.2f} pp de crecimiento por +10 pp de anomalía (p = {pv('Agro', 'Clima(t)'):.2f}). {'Significativo.' if lluvia_agro else 'Sin efecto: el valor agregado anual está amortiguado por cultivos permanentes, ganadería y riego.'}
3. **Agro → pobreza:** {c('Pobreza', 'Agro(t)'):+.3f} pp de IPM por pp de crecimiento (p = {pv('Pobreza', 'Agro(t)'):.3f}); su derrame a vecinos es {ag:+.3f}. {lectura}
4. **La pobreza se contagia:** el {100 * M['spill']['ipm']:.0f}% del efecto de un choque de IPM cae fuera del departamento y regresa +{100 * M['feed']['ipm']:.1f}% al origen.
5. **Implicación de política:** los programas contra la pobreza rural deben diseñarse por **clúster regional**, no departamento por departamento.

<small>Fuentes: CHIRPS v3.0 (UCSB-CHC), DANE (PIB departamental base 2015 e IPM departamental). Estimación: desviaciones ortogonales de Lee-Yu; VI espacial (S2SLS / GMM de Kelejian-Prucha, instrumentos W·X y W²·X) o, si los instrumentos son débiles, QML de Lee-Yu con función de control. Limitación: T corto → sesgo de Nickell O(1/T) en las persistencias e instrumentos potencialmente débiles (ver F de 1ª etapa).</small>
"""


def _tablas_descarga(M):
    return [("panel_spvar.csv", PANEL), ("mecanismos.csv", M["MEC"]), ("impactos.csv", M["IMP"]),
            ("seleccion_modelos.csv", M["SEL"])]


def descargas(M):
    carpeta = tempfile.mkdtemp()
    rutas = []
    for nombre, df in _tablas_descarga(M):
        r = os.path.join(carpeta, nombre); df.to_csv(r, index=False, encoding="utf-8-sig"); rutas.append(r)
    return rutas


def descargas_zip(M):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, df in _tablas_descarga(M):
            z.writestr(nombre, df.to_csv(index=False).encode("utf-8-sig"))
    return buf.getvalue()


# =============================================================================
# MÓDULO 3 · CONTROLADOR DE EJECUCIÓN DUAL (Gradio en Colab / Streamlit en la nube)
# =============================================================================
CSS_COMUN = """
.hero {padding: 42px 36px 30px; border-radius: 24px; border: 1px solid rgba(148,163,184,.15);
  background: linear-gradient(135deg, rgba(8,47,73,.85), rgba(15,23,42,.9) 55%, rgba(69,26,3,.55));
  box-shadow: 0 20px 60px rgba(0,0,0,.45); position: relative; overflow: hidden;}
.hero:after {content:""; position:absolute; inset:0; background-image: radial-gradient(rgba(255,255,255,.18) 1px, transparent 1px);
  background-size: 22px 22px; opacity:.18; pointer-events:none;}
.hero .eyebrow {letter-spacing:.18em; text-transform:uppercase; font-size:.78rem; color:#67e8f9; font-weight:700;}
.hero h1 {font-size: 2.6rem; line-height:1.1; margin:.35em 0 .3em; color:#f8fafc; font-weight:800;}
.hero h1 span {background: linear-gradient(90deg,#22d3ee,#fbbf24); -webkit-background-clip:text; color:transparent;}
.hero p {font-size:1.08rem; color:#cbd5e1; max-width: 760px;}
.badges {margin-top:14px;} .badge {display:inline-block; margin:4px 6px 0 0; padding:5px 12px; border-radius:999px;
  font-size:.78rem; color:#e2e8f0; background:rgba(148,163,184,.14); border:1px solid rgba(148,163,184,.2);}
.kpis {display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin: 18px 0 6px;}
.kpi {padding:18px; border-radius:18px; background:rgba(15,23,42,.75); border:1px solid rgba(148,163,184,.15);
  transition: transform .2s, border-color .2s;}
.kpi:hover {transform: translateY(-3px); border-color: rgba(34,211,238,.6);}
.kpi-ico {font-size:1.4rem;} .kpi-val {font-size:2rem; font-weight:800; color:#f8fafc;} .kpi-lab {font-size:.85rem; color:#94a3b8;}
.guia {padding: 12px 16px; border-left: 3px solid #22d3ee; background: rgba(34,211,238,.06); border-radius: 10px; color:#cbd5e1;}
.pie {text-align:center; color:#64748b; font-size:.82rem; padding: 18px 0 8px;}
@media (max-width: 900px) {.kpis {grid-template-columns: repeat(2,1fr);} .hero h1 {font-size:1.9rem;}}
"""
CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;
  background: radial-gradient(1200px 600px at 10% -10%, rgba(6,182,212,.18), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(245,158,11,.12), transparent 60%), #060a14;}
""" + CSS_COMUN
CSS_ST = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.stApp {background: radial-gradient(1200px 600px at 10% -10%, rgba(6,182,212,.18), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(245,158,11,.12), transparent 60%), #060a14; color:#e2e8f0;
  font-family: Inter, system-ui, sans-serif;}
.block-container {max-width: 1320px; padding-top: 1.5rem;}
[data-testid="stHeader"] {background: transparent;}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label, .stApp td, .stApp th {color:#e2e8f0;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
.stButton button[kind="primary"] {background: linear-gradient(90deg,#06b6d4,#3b82f6); border:0; color:white;}
.stButton button[kind="secondary"] {background: rgba(148,163,184,.12); color:#e2e8f0; border:1px solid rgba(148,163,184,.25);}
""" + CSS_COMUN
HEAD = """
<meta name="description" content="Observatorio interactivo: cómo la lluvia satelital, la producción agropecuaria y la pobreza se transmiten entre los 33 departamentos de Colombia (SpVAR).">
<meta property="og:title" content="Observatorio SpVAR Colombia · Clima, agro y pobreza">
<meta property="og:description" content="Explora mapas, clústeres y simula choques climáticos en un modelo espacial de 33 departamentos.">
"""
HERO = """
    <div class='hero'>
      <div class='eyebrow'>Observatorio de econometría espacial · Colombia</div>
      <h1>¿Qué le hace la <span>lluvia</span> a la pobreza de Colombia?</h1>
      <p>Un modelo SpVAR conecta los 33 departamentos: mira cómo se mueven el clima satelital, la producción agropecuaria
      y la pobreza, y simula qué pasa cuando un choque golpea a un departamento… o a todo el país.</p>
      <div class='badges'><span class='badge'>🛰️ CHIRPS v3.0 · satélite</span><span class='badge'>🏛️ DANE · PIB e IPM</span>
      <span class='badge'>📐 PySAL · spreg</span><span class='badge'>🗺️ 33 departamentos · 2018–2025</span></div>
    </div>"""
GUIA = {
    "explorar": "<div class='guia'>Pregunta guía: <b>¿dónde llueve más de lo normal, dónde crece el agro y dónde se concentra la pobreza?</b> Mueve el año y pasa el cursor por el mapa.</div>",
    "espacio": "<div class='guia'>Pregunta guía: <b>¿los vecinos se parecen?</b> La matriz W define quién es vecino; el I de Moran mide cuánto se parecen y LISA dice dónde.</div>",
    "modelo": "<div class='guia'>Cada ecuación elige su estructura (MCO, SAR, SEM, SAC, SDM, SDEM) con multiplicadores de Lagrange, AIC y BIC. Cambia W o los efectos fijos y vuelve a estimar.</div>",
    "simulador": "<div class='guia'>Pregunta guía: <b>¿y si mañana llueve mucho más de lo normal?</b> Elige dónde ocurre el choque y mira cómo viaja por el territorio año a año (botón ▶ del mapa).</div>"}
PIE = ("<div class='pie'>Observatorio SpVAR Colombia · Econometría espacial · Datos: CHIRPS v3.0 (UCSB-CHC) y DANE · "
       "Modelo: SpVAR trivariado con selección LM/AIC/BIC y multiplicador S = (I − A0)⁻¹</div>")
TITULO = "Observatorio SpVAR Colombia · Clima, agro y pobreza"
PESTANAS = ["🛰️ Explorar", "🧭 Vecindad y clústeres", "⚙️ Modelo SpVAR", "🌧️ Simulador de choques", "📄 Reporte"]
EFECTOS = ["Departamento + año", "Solo departamento"]
OPC_DPTO = [("🇨🇴 Todo el país (comparar los 33 orígenes)", "todo")] + [(NOMBRE[c], c) for c in sorted(COD, key=lambda c: NOMBRE[c])]
OPC_DPTO_E = [("🇨🇴 Colombia (33 departamentos)", "todo")] + [(NOMBRE[c], c) for c in sorted(COD, key=lambda c: NOMBRE[c])]
OPC_VAR = [(ETQ[v], v) for v in VARS]


# -----------------------------------------------------------------------------
# 3a. INTERFAZ GRADIO (Colab / Jupyter / python app.py) · idéntica a la original
# -----------------------------------------------------------------------------
def lanzar_gradio():
    import gradio as gr
    TEMA = gr.themes.Base(primary_hue="cyan", secondary_hue="amber", neutral_hue="slate",
                          font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                          radius_size=gr.themes.sizes.radius_lg).set(
        body_background_fill="#060a14", body_background_fill_dark="#060a14",
        body_text_color="#e2e8f0", body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#94a3b8", body_text_color_subdued_dark="#94a3b8",
        block_background_fill="rgba(15,23,42,0.72)", block_background_fill_dark="rgba(15,23,42,0.72)",
        block_border_color="rgba(148,163,184,0.15)", block_border_color_dark="rgba(148,163,184,0.15)",
        block_label_text_color="#94a3b8", block_label_text_color_dark="#94a3b8",
        block_title_text_color="#e2e8f0", block_title_text_color_dark="#e2e8f0",
        input_background_fill="#0f172a", input_background_fill_dark="#0f172a",
        border_color_primary="rgba(148,163,184,0.25)", border_color_primary_dark="rgba(148,163,184,0.25)",
        button_primary_background_fill="linear-gradient(90deg,#06b6d4,#3b82f6)",
        button_primary_background_fill_dark="linear-gradient(90deg,#06b6d4,#3b82f6)",
        button_primary_text_color="white", button_primary_text_color_dark="white",
        button_secondary_background_fill="rgba(148,163,184,0.12)", button_secondary_background_fill_dark="rgba(148,163,184,0.12)",
        button_secondary_text_color="#e2e8f0", button_secondary_text_color_dark="#e2e8f0",
        table_even_background_fill="#0f172a", table_even_background_fill_dark="#0f172a",
        table_odd_background_fill="#111a2e", table_odd_background_fill_dark="#111a2e",
        color_accent_soft="rgba(6,182,212,0.15)", color_accent_soft_dark="rgba(6,182,212,0.15)")

    print("Calculando la matriz W óptima y estimando el SpVAR inicial (≈ 30 s la primera vez)…")
    W0 = mejor_W()
    M0 = modelo(W0, "Departamento + año")
    print(f"✓ Modelo listo · W = {W0} · geometría: {GEO_FUENTE}")

    GR6 = int(gr.__version__.split(".")[0]) >= 6
    ESTILO = dict(theme=TEMA, css=CSS, head=HEAD)
    with gr.Blocks(title=TITULO, **({} if GR6 else ESTILO)) as demo:
        estado = gr.State((W0, "Departamento + año"))
        gr.HTML(HERO)
        kpi_html = gr.HTML(kpis(M0))
        with gr.Row():
            cta1 = gr.Button("🛰️ Explorar el territorio", variant="primary", size="lg")
            cta2 = gr.Button("🌧️ Simular un choque", variant="secondary", size="lg")

        with gr.Tabs() as tabs:
            # --- 1. EXPLORAR ---
            with gr.Tab(PESTANAS[0], id="explorar"):
                gr.HTML(GUIA["explorar"])
                with gr.Row():
                    var_e = gr.Dropdown(OPC_VAR, value="ipm", label="Variable")
                    anio_e = gr.Slider(ANIOS[0], ANIOS[-1], value=ANIOS[-1], step=1, label="Año")
                    dpto_e = gr.Dropdown(OPC_DPTO_E, value="todo", label="Zoom a un departamento")
                with gr.Row():
                    with gr.Column(scale=3):
                        mapa_e = gr.Plot(fig_explorar("ipm", ANIOS[-1]), label="Mapa")
                    with gr.Column(scale=2):
                        ficha_e = gr.Markdown(ficha("ipm", "todo", ANIOS[-1]))
                        serie_e = gr.Plot(fig_serie("ipm", "todo"), label="Serie")
            # --- 2. DEPENDENCIA ESPACIAL ---
            with gr.Tab(PESTANAS[1], id="espacio"):
                gr.HTML(GUIA["espacio"])
                with gr.Row():
                    w_d = gr.Dropdown(MATRICES, value=W0, label="Matriz de pesos W")
                    v_d = gr.Radio(OPC_VAR, value="ipm", label="Variable para LISA")
                with gr.Row():
                    moran_d = gr.Plot(fig_moran(W0))
                    lisa_d = gr.Plot(fig_lisa(W0, "ipm"))
            # --- 3. MODELO ---
            with gr.Tab(PESTANAS[2], id="modelo"):
                gr.HTML(GUIA["modelo"])
                with gr.Row():
                    w_m = gr.Dropdown(MATRICES, value=W0, label="Matriz W")
                    ef_m = gr.Radio(EFECTOS, value="Departamento + año", label="Efectos fijos")
                    btn_m = gr.Button("⚙️ Estimar el sistema", variant="primary")
                sel_m = gr.Dataframe(M0["SEL"], label="Estructura elegida por ecuación", interactive=False)
                with gr.Row():
                    mec_m = gr.Plot(fig_mecanismos(M0))
                with gr.Row():
                    imp_m = gr.Plot(fig_impactos(M0))
                    est_m = gr.Markdown(texto_estabilidad(M0))
            # --- 4. SIMULADOR ---
            with gr.Tab(PESTANAS[3], id="simulador"):
                gr.HTML(GUIA["simulador"])
                with gr.Row():
                    vc_s = gr.Dropdown(OPC_VAR, value="clima", label="El choque ocurre en…")
                    or_s = gr.Dropdown(OPC_DPTO, value="todo", label="¿Dónde?")
                    mg_s = gr.Slider(-3, 3, value=1, step=0.5, label="Magnitud (desviaciones estándar)")
                    vr_s = gr.Dropdown(OPC_VAR, value="ipm", label="Ver la respuesta de…")
                btn_s = gr.Button("▶ Simular", variant="primary")
                f0, l0, t0 = fig_simulacion(M0, "clima", "todo", 1, "ipm")
                txt_s = gr.Markdown(t0)
                with gr.Row():
                    with gr.Column(scale=3):
                        mapa_s = gr.Plot(f0)
                    with gr.Column(scale=2):
                        linea_s = gr.Plot(l0)
            # --- 5. REPORTE ---
            with gr.Tab(PESTANAS[4], id="reporte"):
                concl = gr.Markdown(conclusiones(M0))
                btn_r = gr.Button("⬇️ Generar archivos del análisis", variant="secondary")
                files_r = gr.File(label="Descargas (CSV)", file_count="multiple")

        gr.HTML(PIE)

        # --- Eventos ---
        cta1.click(lambda: gr.Tabs(selected="explorar"), None, tabs)
        cta2.click(lambda: gr.Tabs(selected="simulador"), None, tabs)
        for comp in (var_e, anio_e, dpto_e):
            comp.change(lambda v, a, d: (fig_explorar(v, int(a)), ficha(v, d, int(a)), fig_serie(v, d)),
                        [var_e, anio_e, dpto_e], [mapa_e, ficha_e, serie_e])
        for comp in (w_d, v_d):
            comp.change(lambda w, v: (fig_moran(w), fig_lisa(w, v)), [w_d, v_d], [moran_d, lisa_d])

        def estimar(w, ef):
            M = modelo(w, ef)
            f, l, t = fig_simulacion(M, "clima", "todo", 1, "ipm")
            return ((w, ef), M["SEL"], fig_mecanismos(M), fig_impactos(M), texto_estabilidad(M), kpis(M), conclusiones(M), f, l, t)
        btn_m.click(estimar, [w_m, ef_m], [estado, sel_m, mec_m, imp_m, est_m, kpi_html, concl, mapa_s, linea_s, txt_s])
        btn_s.click(lambda st_, vc, o, m, vr: fig_simulacion(modelo(*st_), vc, o, float(m), vr),
                    [estado, vc_s, or_s, mg_s, vr_s], [mapa_s, linea_s, txt_s])
        btn_r.click(lambda st_: descargas(modelo(*st_)), estado, files_r)

    # --- Lanzamiento: dentro de Colab (iframe) + enlace público; se puede re-ejecutar la celda sin reiniciar ---
    en_colab = ENTORNO == "colab"
    gr.close_all()                                   # cierra una interfaz anterior si la celda se ejecuta otra vez
    opciones = dict(share=en_colab, debug=False, show_error=True, inline=ENTORNO in ("colab", "jupyter"), height=1100)
    if GR6:
        opciones.update(ESTILO)
    try:
        demo.queue().launch(**opciones)
    except Exception as e:                            # p. ej., red que bloquea el túnel público
        print(f"⚠ No se pudo crear el enlace público ({e}). La interfaz queda disponible dentro de Colab.")
        opciones["share"] = False
        demo.queue().launch(**opciones)
    print("✓ Interfaz lista. Si no ves el panel arriba, abre el enlace público (*.gradio.live) en otra pestaña.")
    return demo


# -----------------------------------------------------------------------------
# 3b. INTERFAZ STREAMLIT (streamlit run app.py · Streamlit Cloud) · misma jerarquía de pestañas y controles
# -----------------------------------------------------------------------------
def lanzar_streamlit():
    st.set_page_config(page_title=TITULO, page_icon="🛰️", layout="wide")
    st.markdown(f"<style>{CSS_ST}</style>", unsafe_allow_html=True)
    ss = st.session_state

    def plot(fig, key):
        try:
            st.plotly_chart(fig, width="stretch", theme=None, key=key)
        except TypeError:                                      # Streamlit < 1.45
            st.plotly_chart(fig, use_container_width=True, theme=None, key=key)

    def elegir(label, opciones, valor, key):
        valores = [v for _, v in opciones]; etiquetas = dict((v, l) for l, v in opciones)
        return st.selectbox(label, valores, index=valores.index(valor), format_func=etiquetas.get, key=key)

    with st.spinner("Calculando la matriz W óptima y estimando el SpVAR inicial (≈ 30 s la primera vez)…"):
        W0 = mejor_W()
        if "estado" not in ss:
            ss.estado = (W0, "Departamento + año")
        M = modelo(*ss.estado)

    st.markdown(HERO, unsafe_allow_html=True)
    st.markdown(kpis(M), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🛰️ Explorar el territorio", type="primary", width="stretch"):
        ss.pestana = PESTANAS[0]
    if c2.button("🌧️ Simular un choque", type="secondary", width="stretch"):
        ss.pestana = PESTANAS[3]
    try:
        tabs = st.tabs(PESTANAS, key="pestana", on_change="rerun")
    except TypeError:                                          # versiones sin selección programática
        tabs = st.tabs(PESTANAS)

    with tabs[0]:                                              # --- 1. EXPLORAR ---
        st.markdown(GUIA["explorar"], unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a: var_e = elegir("Variable", OPC_VAR, "ipm", "var_e")
        with b: anio_e = st.slider("Año", ANIOS[0], ANIOS[-1], ANIOS[-1], step=1, key="anio_e")
        with c: dpto_e = elegir("Zoom a un departamento", OPC_DPTO_E, "todo", "dpto_e")
        izq, der = st.columns([3, 2])
        with izq: plot(fig_explorar(var_e, int(anio_e)), "mapa_e")
        with der:
            st.markdown(ficha(var_e, dpto_e, int(anio_e)))
            plot(fig_serie(var_e, dpto_e), "serie_e")

    with tabs[1]:                                              # --- 2. DEPENDENCIA ESPACIAL ---
        st.markdown(GUIA["espacio"], unsafe_allow_html=True)
        a, b = st.columns(2)
        with a: w_d = st.selectbox("Matriz de pesos W", MATRICES, index=MATRICES.index(W0), key="w_d")
        with b:
            v_d = st.radio("Variable para LISA", [v for _, v in OPC_VAR], index=2, horizontal=True,
                           format_func=dict((v, l) for l, v in OPC_VAR).get, key="v_d")
        a, b = st.columns(2)
        with a: plot(fig_moran(w_d), "moran_d")
        with b: plot(fig_lisa(w_d, v_d), "lisa_d")

    with tabs[2]:                                              # --- 3. MODELO ---
        st.markdown(GUIA["modelo"], unsafe_allow_html=True)
        a, b, c = st.columns([2, 2, 1])
        with a: w_m = st.selectbox("Matriz W", MATRICES, index=MATRICES.index(ss.estado[0]), key="w_m")
        with b: ef_m = st.radio("Efectos fijos", EFECTOS, index=EFECTOS.index(ss.estado[1]), horizontal=True, key="ef_m")
        with c:
            st.write("")
            if st.button("⚙️ Estimar el sistema", type="primary", width="stretch"):
                with st.spinner("Estimando el sistema…"):
                    modelo(w_m, ef_m)
                ss.estado = (w_m, ef_m)
                ss.pop("sim", None)
                st.rerun()
        st.caption("Estructura elegida por ecuación")
        st.dataframe(M["SEL"], hide_index=True, width="stretch")
        plot(fig_mecanismos(M), "mec_m")
        a, b = st.columns([3, 2])
        with a: plot(fig_impactos(M), "imp_m")
        with b: st.markdown(texto_estabilidad(M))

    with tabs[3]:                                              # --- 4. SIMULADOR ---
        st.markdown(GUIA["simulador"], unsafe_allow_html=True)
        a, b, c, d = st.columns(4)
        with a: vc_s = elegir("El choque ocurre en…", OPC_VAR, "clima", "vc_s")
        with b: or_s = elegir("¿Dónde?", OPC_DPTO, "todo", "or_s")
        with c: mg_s = st.slider("Magnitud (desviaciones estándar)", -3.0, 3.0, 1.0, step=0.5, key="mg_s")
        with d: vr_s = elegir("Ver la respuesta de…", OPC_VAR, "ipm", "vr_s")
        if "sim" not in ss:
            ss.sim = ("clima", "todo", 1.0, "ipm")
        if st.button("▶ Simular", type="primary"):
            ss.sim = (vc_s, or_s, float(mg_s), vr_s)
        f, l, t = fig_simulacion(M, *ss.sim)
        st.markdown(t)
        izq, der = st.columns([3, 2])
        with izq: plot(f, "mapa_s")
        with der: plot(l, "linea_s")

    with tabs[4]:                                              # --- 5. REPORTE ---
        st.markdown(conclusiones(M), unsafe_allow_html=True)
        if st.button("⬇️ Generar archivos del análisis", type="secondary"):
            ss.zip_listo = True
        if ss.get("zip_listo"):
            st.download_button("Descargas (CSV) · analisis_spvar.zip", descargas_zip(M), file_name="analisis_spvar.zip",
                               mime="application/zip")

    st.markdown(PIE, unsafe_allow_html=True)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    if ENTORNO == "streamlit":
        lanzar_streamlit()
    else:
        demo = lanzar_gradio()
