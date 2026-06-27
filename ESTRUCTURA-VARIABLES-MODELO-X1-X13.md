# 📊 MODELO MATEMÁTICO DEL PROYECTO DE PREDICCIÓN

En este proyecto se trabajan dos variables objetivo numéricas independientes para modelar el resultado de un encuentro deportivo:

- \( y_{\text{home}} \): Goles anotados por el equipo local  
- \( y_{\text{away}} \): Goles anotados por el equipo visitante  

Debido a esta naturaleza bidimensional, cada algoritmo se entrena de forma independiente en dos versiones:

1. **modelo_home** → Predice \( y_{\text{home}} \)  
2. **modelo_away** → Predice \( y_{\text{away}} \)

---

## 📋 VARIABLES PREDICTORAS

El vector de entrada está compuesto por 13 características socio-deportivas:

\[
X = (X_1, X_2, X_3, \dots, X_{13})
\]

| Variable matemática | Variable en el código | Descripción |
| :---: | :--- | :--- |
| \(X_1\) | `home_gf12` | Goles a favor recientes del equipo local |
| \(X_2\) | `home_ga12` | Goles en contra recientes del equipo local |
| \(X_3\) | `home_pts12` | Puntos recientes del equipo local |
| \(X_4\) | `home_prev_matches` | Partidos recientes del equipo local |
| \(X_5\) | `away_gf12` | Goles a favor del visitante |
| \(X_6\) | `away_ga12` | Goles en contra del visitante |
| \(X_7\) | `away_pts12` | Puntos recientes del visitante |
| \(X_8\) | `away_prev_matches` | Partidos recientes del visitante |
| \(X_9\) | `diff_fifa` | Diferencia en ranking FIFA |
| \(X_{10}\) | `diff_elo` | Diferencia en rating Elo |
| \(X_{11}\) | `h2h` | Historial de enfrentamientos directos |
| \(X_{12}\) | `neutral` | Indicador de campo neutral (binario) |
| \(X_{13}\) | `home_advantage` | Factor de ventaja de localía |

---

## 🧠 ALGORITMOS IMPLEMENTADOS

### 1. Regresión Lineal Múltiple

\[
\hat{y} = \beta_0 + \sum_{i=1}^{13} \beta_i X_i
\]

Aplicado a cada equipo:

\[
\hat{y}_{\text{home}} = \beta_{0,h} + \sum_{i=1}^{13} \beta_{i,h} X_i
\]
\[
\hat{y}_{\text{away}} = \beta_{0,a} + \sum_{i=1}^{13} \beta_{i,a} X_i
\]

**Interpretación:**
- \( \beta > 0 \): aumenta la predicción de goles  
- \( \beta < 0 \): reduce la predicción de goles  
- \( \beta \approx 0 \): baja influencia  

---

### 2. Regresión Ridge (L2)

\[
\min_{\beta} \sum (y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2
\]

- Reduce sobreajuste  
- Estabiliza coeficientes  
- Maneja multicolinealidad  

---

### 3. Random Forest

\[
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)
\]

- Ensamble de árboles  
- Captura relaciones no lineales  
- Robusto a ruido  

---

### 4. Gradient Boosting

\[
F_m(X) = F_{m-1}(X) + \nu h_m(X)
\]

Residuos:

\[
r_{im} = y_i - F_{m-1}(X_i)
\]

- Corrección secuencial de errores  
- Alta precisión predictiva  

---

### 5. Poisson Regressor

\[
Y \sim \text{Poisson}(\lambda)
\]

\[
\log(\lambda) = \beta_0 + \sum_{i=1}^{13} \beta_i X_i
\]

\[
\lambda = e^{(\beta_0 + \sum \beta_i X_i)}
\]

Definición:

\[
\lambda_{\text{home}}, \lambda_{\text{away}}
\]

---

## 🎲 MATRIZ DE PROBABILIDADES

\[
P(Y=a) = \frac{e^{-\lambda} \lambda^a}{a!}
\]

\[
P(a,b) = P(\text{home}=a)\cdot P(\text{away}=b)
\]

Resultados:

- Victoria local → \(a > b\)  
- Empate → \(a = b\)  
- Victoria visitante → \(a < b\)

---

## 🔧 AJUSTES AVANZADOS

### 📌 Dixon-Coles

\[
P'(a,b) = P(a,b)\cdot \tau(a,b,\rho)
\]

---

### 📌 Contexto competitivo

\[
\lambda_{home} = \lambda_{home\_base} \cdot factor
\]
\[
\lambda_{away} = \lambda_{away\_base} \cdot factor
\]

- 1.10 → ofensivo  
- 1.00 → neutral  
- 0.90 → defensivo  

---

## 🔄 FLUJO DEL MODELO

DATA HISTÓRICA  
↓  
FEATURE ENGINEERING  
↓  
ENTRENAMIENTO DE MODELOS  
↓  
PREDICCIÓN DE GOLES  
↓  
EVALUACIÓN (MAE, RMSE, R²)  
↓  
SELECCIÓN DEL MEJOR MODELO  
↓  
AJUSTE POR CONTEXTO  
↓  
POISSON + DIXON-COLES  
↓  
PROBABILIDADES FINALES  

---

## ⚽ SALIDAS DEL SISTEMA

- Marcadores más probables  
- Probabilidades 1X2  
- Clasificación de grupos  
- Simulación de fases finales  

---

## ⚠️ NOTA FINAL

El modelo no garantiza resultados reales.

Se basa en datos históricos, forma reciente, rankings (FIFA/Elo), enfrentamientos directos y contexto competitivo.
