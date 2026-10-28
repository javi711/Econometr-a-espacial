# 🛰️ Observatorio SpVAR Colombia · Clima, agro y pobreza

**¿Qué le hace la lluvia a la pobreza de Colombia?** Aplicación interactiva de econometría espacial que conecta a los
**33 departamentos** en un modelo SpVAR (VAR espacial estructural) de tres variables:

| Variable | Fuente | Transformación |
|---|---|---|
| Clima | CHIRPS v3.0 (UCSB-CHC), precipitación satelital 1991–2025 | Anomalía (%) frente a la normal 1991–2020 |
| Agro | DANE, valor agregado agropecuario 2005–2025 (base 2015) | Crecimiento logarítmico (%) |
| Pobreza | DANE, Índice de Pobreza Multidimensional 2018–2025 | % de personas |

Los datos reales vienen embebidos en el código, así que la app arranca sin descargar nada salvo el mapa.

---

## 📂 Contenido del repositorio

```
├── app.py             # Toda la aplicación: datos, motor econométrico, gráficos e interfaz
├── requirements.txt   # Dependencias con versiones fijas para Streamlit Cloud
└── README.md          # Este archivo
```

`app.py` detecta solo dónde se ejecuta:

- con `streamlit run app.py` → interfaz **Streamlit**;
- en **Google Colab** o Jupyter → interfaz **Gradio** con enlace público;
- con `python app.py` → interfaz Gradio local.

---

## 🚀 Despliegue en Streamlit Cloud

1. Sube `app.py`, `requirements.txt` y `README.md` a la raíz de un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub y pulsa **Create app**.
3. Elige el repositorio, la rama `main` y en **Main file path** escribe `app.py`.
4. En **Advanced settings**, selecciona **Python 3.12 o 3.13**.
5. Pulsa **Deploy**. La primera vez tarda unos minutos instalando librerías; después la app carga en segundos.

> Si el servidor no logra descargar el mapa de departamentos, la app usa un respaldo embebido con la misma
> topología, así que los resultados no cambian.

## 💻 Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Google Colab

Copia todo el contenido de `app.py` en una sola celda y ejecútala (**Entorno de ejecución → Ejecutar todas**).
Se instala lo que falte, la interfaz aparece dentro de Colab y se imprime un enlace `https://….gradio.live`
válido por 72 horas.

---

## 🗂️ Pestañas de la interfaz

| Pestaña | Qué muestra |
|---|---|
| 🛰️ Explorar | Mapa por año y variable, ficha del departamento y serie frente al país |
| 🧭 Vecindad y clústeres | I de Moran para 6 matrices W (Queen, Rook, KNN3–5, distancia inversa) y mapa LISA |
| ⚙️ Modelo SpVAR | Estructura elegida por ecuación, mecanismos de transmisión, impactos y estabilidad |
| 🌧️ Simulador de choques | Impulso-respuesta espacial animado año a año, desde un departamento o todo el país |
| 📄 Reporte | Conclusiones automáticas y descarga de resultados en CSV |

---

## 📐 Metodología

**Sistema estructural recursivo** (clima → agro → pobreza), para cada ecuación *k*:

$$y_{k,t} = \rho_k W y_{k,t} + \sum_{j<k} \left(\beta_{kj}\, y_{j,t} + \theta_{kj}\, W y_{j,t}\right) + \sum_j \left(\gamma_{kj}\, y_{j,t-1} + \delta_{kj}\, W y_{j,t-1}\right) + \mu_{k,i} + \tau_{k,t} + v_{k,t}$$

1. **Panel:** transformación ortonormal de Lee y Yu (2010) (desviaciones ortogonales hacia adelante). Elimina los
   efectos fijos departamentales sin correlacionar los errores; los efectos de año entran como dummies.
2. **Selección de la estructura espacial** por ecuación (MCO, SLX, SAR, SEM, SDM, SDEM, SAC): árbol de
   multiplicadores de Lagrange robustos, prueba LR del bloque SLX, AIC/BIC y prueba de Anselin-Kelejian.
3. **Estimación:** W·y(t) y Agro(t) en la ecuación de Pobreza se tratan como endógenas.
   - Variables instrumentales espaciales (S2SLS / GMM de Kelejian-Prucha) con instrumentos W·X y W²·X.
   - Si los instrumentos son débiles (F de primera etapa < 10), QML de Lee-Yu con función de control para Agro(t).
4. **Multiplicador:** S = (I − A0)⁻¹ con verificación de radio espectral < 1 y convergencia de la serie de Neumann.
5. **Impactos de LeSage y Pace (2009)** por bloque: directo = tr(S)/N, total = 1′S1/N, indirecto = total − directo,
   con intervalos de confianza Monte Carlo que incorporan la correlación entre ecuaciones.
6. **Impulso-respuesta espacial:** Ψₕ = Φʰ S B (L̃ ⊗ I), con Φ = S A1 estable y choques ortogonalizados por Cholesky.

**Limitaciones:** el panel es corto (T = 7 años con rezago), lo que deja un sesgo de Nickell pequeño en las
persistencias y hace que los instrumentos suelan ser débiles. La app reporta el F de primera etapa, Sargan,
Anselin-Kelejian y Hausman para que cada resultado se lea con esa cautela. La relación agro → pobreza debe
interpretarse como asociación salvo que los instrumentos sean fuertes.

---

## ⚙️ Opciones avanzadas (variables de entorno)

| Variable | Efecto |
|---|---|
| `GEO_PATH` | Ruta o URL alternativa del GeoJSON de departamentos |
| `MAP_STYLE` | Estilo del mapa base de Plotly (por defecto `carto-darkmatter`) |
| `SPVAR_NICKELL_IV=1` | Instrumenta también los rezagos con niveles t−1 y t−2 (análisis de sensibilidad) |

---

## 📚 Referencias

- Anselin, L., Bera, A., Florax, R. y Yoon, M. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics*.
- Arellano, M. y Bover, O. (1995). Another look at the instrumental variable estimation of error-components models. *Journal of Econometrics*.
- Kelejian, H. y Prucha, I. (1998). A generalized spatial two-stage least squares procedure. *Journal of Real Estate Finance and Economics*.
- Lee, L.-F. y Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*.
- LeSage, J. y Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Rey, S. y Anselin, L. (2007). PySAL: A Python library of spatial analytical methods. *Review of Regional Studies*.

**Fuentes de datos:** CHIRPS v3.0 (Climate Hazards Center, UC Santa Barbara) · DANE (PIB departamental e IPM) ·
Marco Geoestadístico Nacional (MGN) del DANE para los límites departamentales.
