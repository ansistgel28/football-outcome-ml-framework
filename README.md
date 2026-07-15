# ⚽ MODELO MUNDIAL 2026 - COMPETENCIA DE MODELOS Y CONTEXTO COMPETITIVO

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-ScikitLearn-F7931E?style=flat)
![Project](https://img.shields.io/badge/Status-Active-success?style=flat)
![Domain](https://img.shields.io/badge/Domain-Sports%20Analytics-blue?style=flat)
![Type](https://img.shields.io/badge/System-Predictive%20Model-purple?style=flat)

> Sistema de predicción de resultados del Mundial 2026 basado en modelos de regresión y contexto competitivo

# 👥 INTEGRANTES DEL PROYECTO

Este proyecto fue desarrollado por el siguiente equipo:

- 👤 **Integrante 1:** BUSTAMANTE VELA JEAN FRANK
- 👤 **Integrante 2:** MARREROS CORTEGANA MIGUEL ANGEL 
- 👤 **Integrante 3:** FERNANDEZ CAMPOS VALENTIN 
- 👤 **Integrante 4:** SALON TRIGOSO FRANK

```
```
📌 **Curso:** MODELOS DE REGRESIÓN Y APRENDIZAJE SUPERVISADO / Ciencia de Datos / Machine Learning  
📌 **Proyecto:** Modelo Mundial 2026 - Predicción de resultados  
📌 **Institución:** Universidad Nacional Toribio Rodríguez de Mendoza de Amazonas

# 📄 ABSTRACT

Este proyecto construye un sistema de predicción de resultados para el Mundial 2026 basado en múltiples modelos de regresión.  
El sistema estima goles esperados por partido, evalúa distintos modelos predictivos y posteriormente ajusta las predicciones mediante un componente de contexto competitivo asociado a la fase de grupos.

El objetivo es generar:

- probabilidades de resultado (victoria local, empate, victoria visitante)  
- estimación de marcadores más probables  
- simulación de clasificación a la siguiente fase del torneo  

El modelo integra información histórica, ranking FIFA, Elo y dinámica competitiva.

---

## 🏛️ Arquitectura del Sistema

Este sistema sigue una arquitectura modular de machine learning con múltiples capas de procesamiento.

```mermaid
graph TB

    %% DATA SOURCES
    A[📊 Data Sources] --> A1[results.csv]
    A --> A2[fifa_ranking.csv]
    A --> A3[elo_ranking.csv]
    A --> A4[fixtures_2026.csv]

    %% DATA PROCESSING
    A --> B[⚙️ Data Processing Layer]
    B --> B1[01_construir_data_modelo.py]

    %% CONTEXT ENGINEERING
    B --> C[🎯 Context Engineering Layer]
    C --> C1[05_crear_contexto_grupos.py]

    %% MODEL TRAINING
    C --> D[🧠 Model Training Layer]
    D --> D1[02_entrenar_predecir_modelos.py]

    %% EVALUATION
    D --> E[📈 Evaluation Layer]
    E --> E1[competencia_modelos.csv]
    E --> E2[mejor_modelo.txt]

    %% PREDICTION ENGINE
    E --> F[⚽ Prediction Engine]
    F --> F1[Poisson + Dixon-Coles]

    %% VISUALIZATION
    F --> G[📊 Visualization Layer]
    G --> G1[03_visualizar_partido_modelos.py]

    %% CLASSIFICATION SIMULATION
    G --> H[🏆 Classification Engine]
    H --> H1[06_predecir_clasificados.py]
    H --> H2[tabla_grupos_predicha.csv]
    H --> H3[clasificados_16avos.csv]

    %% OUTPUTS
    H --> I[📦 Outputs]
    I --> I1[top10_marcadores.csv]
    I --> I2[predicciones_fixture.csv]

```
---

# 🏗️ METHODOLOGY (PIPELINE GENERAL)
```
DATA HISTÓRICA
↓
FEATURE ENGINEERING
↓
ENTRENAMIENTO DE MODELOS
↓
PREDICCIÓN DE GOLES (λ_home, λ_away)
↓
EVALUACIÓN (MAE, RMSE, R2)
↓
SELECCIÓN DEL MEJOR MODELO
↓
AJUSTE POR CONTEXTO COMPETITIVO
↓
POISSON + DIXON-COLES → PROBABILIDADES FINALES
↓
SIMULACIÓN DE CLASIFICACIÓN (FASE DE GRUPOS)
↓
DETERMINACIÓN DE CLASIFICADOS A SIGUIENTE FASE
```
---

## 📁 **Estructura del Proyecto**

```
modelo regresion/
│
├── data/
│   ├── results.csv
│   ├── fifa_ranking.csv
│   ├── elo_ranking.csv
│   ├── fixtures_2026.csv
│   ├── fixtures_2026_group_stage.csv
│   ├── data_modelo.csv
│   ├── fixtures_modelo.csv
│   └── fixtures_contexto_grupos.csv
│
├── outputs/
│   ├── competencia_modelos.csv
│   ├── mejor_modelo.txt
│   ├── modelo_home.pkl
│   ├── modelo_away.pkl
│   ├── predicciones_fixture.csv
│   ├── predicciones_todos_modelos.csv
│   ├── top10_marcadores.csv
│   ├── top10_todos_modelos.csv
│   ├── tabla_grupos_predicha.csv
│   ├── clasificados_grupos.csv
│   ├── mejores_terceros.csv
│   ├── clasificados_16avos.csv
│   ├── cruces_16avos_predichos.csv
│   └── panel_tablas_grupos.png
│
├── 00_actualizar_results_mundial.py
├── 01_construir_data_modelo.py
├── 02_entrenar_predecir_modelos.py
├── 03_visualizar_partido_modelos.py
├── 04_mostrar_modelo_final.py
├── 05_crear_contexto_grupos.py
├── 06_predecir_clasificados.py
├── requirements.txt
└── README.txt
```

---
# ⚙️ REQUIREMENTS
```
pip install -r requirements.txt
numpy
pandas
scipy
scikit-learn
matplotlib
joblib
```

---
# 🚀 EXECUTION PIPELINE
```
1) cd "/home/jeanki/Escritorio/modelo regresion"

2) python 00_actualizar_results_mundial.py

3) python 01_construir_data_modelo.py

4) python 05_crear_contexto_grupos.py

5) python 02_entrenar_predecir_modelos.py

6) python 03_visualizar_partido_modelos.py (MATCH_ID = 60)

7) python 04_mostrar_modelo_final.py

8) python 06_predecir_clasificados.py
```
---

# 🤖 MODELOS
```
- Linear Regression
- Ridge
- Random Forest
- Gradient Boosting
- Poisson Regressor
```
---

# 📊 FEATURES
```
- home_gf12
- home_ga12
- home_pts12
- ome_prev_matches
- way_gf12
- away_ga12
- away_pts12
- away_prev_matches
- diff_fifa
- diff_elo
- h2h
- neutral
- home_advantage
```
---

# 🎯 CONTEXT
```
lambda_home_final = lambda_home_base * factor_contexto_home  
lambda_away_final = lambda_away_base * factor_contexto_away  

- 1.15 = must win  
- 0.90 = rotation  
- 1.00 = neutral   
```
---

# 📦 OUTPUTS
```
-competencia_modelos.csv
-mejor_modelo.txt
-predicciones_fixture.csv
-top10_marcadores.csv
-tabla_grupos_predicha.csv
-clasificados_16avos.csv
-cruces_16avos_predichos.csv
```
---

# ⚠️ NOTE
```
El modelo no garantiza el resultado real de los partidos.
Su objetivo es estimar probabilidades usando información histórica, forma reciente, 
ranking, Elo, historial y contexto competitivo del grupo.
```
# ⚽ Clasificados a dieciseisavos de final — Mundial 2026

Estas son las **32 selecciones** que, según la predicción del modelo, avanzaron a los dieciseisavos de final.

| N.º | Selección | Grupo | Clasificación |
|:---:|---|:---:|---|
| 1 | México | A | Primero |
| 2 | Sudáfrica | A | Segundo |
| 3 | Corea del Sur | A | Mejor tercero |
| 4 | Suiza | B | Primero |
| 5 | Canadá | B | Segundo |
| 6 | Bosnia y Herzegovina | B | Mejor tercero |
| 7 | Brasil | C | Primero |
| 8 | Marruecos | C | Segundo |
| 9 | Estados Unidos | D | Primero |
| 10 | Australia | D | Segundo |
| 11 | Turquía | D | Mejor tercero |
| 12 | Alemania | E | Primero |
| 13 | Costa de Marfil | E | Segundo |
| 14 | Ecuador | E | Mejor tercero |
| 15 | Países Bajos | F | Primero |
| 16 | Japón | F | Segundo |
| 17 | Suecia | F | Mejor tercero |
| 18 | Egipto | G | Primero |
| 19 | Bélgica | G | Segundo |
| 20 | Irán | G | Mejor tercero |
| 21 | España | H | Primero |
| 22 | Cabo Verde | H | Segundo |
| 23 | Francia | I | Primero |
| 24 | Noruega | I | Segundo |
| 25 | Senegal | I | Mejor tercero |
| 26 | Argentina | J | Primero |
| 27 | Austria | J | Segundo |
| 28 | Colombia | K | Primero |
| 29 | Portugal | K | Segundo |
| 30 | Inglaterra | L | Primero |
| 31 | Croacia | L | Segundo |
| 32 | Ghana | L | Mejor tercero |

> Fuente: `outputs/clasificados_16avos.csv`.
