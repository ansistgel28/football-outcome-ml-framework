# ============================================================
# 02_entrenar_y_predecir.py
# Competencia de modelos + predicciones Mundial 2026
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.stats import poisson

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)
import joblib


# 1. CONFIGURACIÓN GENERAL

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_DATA_MODELO = DATA_DIR / "data_modelo.csv"
ARCHIVO_FIXTURES_MODELO = DATA_DIR / "fixtures_modelo.csv"
ARCHIVO_CONTEXTO_GRUPOS = DATA_DIR / "fixtures_contexto_grupos.csv"

OUT_DIR.mkdir(exist_ok=True)

VARIABLES = [
    "home_gf12",
    "home_ga12",
    "home_pts12",
    "home_prev_matches",
    "away_gf12",
    "away_ga12",
    "away_pts12",
    "away_prev_matches",
    "diff_fifa",
    "diff_elo",
    "h2h",
    "neutral",
    "home_advantage",
]

MAX_GOLES = 8
RHO_DIXON_COLES = -0.08


# 2. MODELOS CANDIDATOS

def crear_modelos_candidatos():
    modelos = {
        "Regresion Lineal": Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", LinearRegression())
        ]),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", Ridge(alpha=1.0))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "Poisson": Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", PoissonRegressor(alpha=0.01, max_iter=3000))
        ])
    }

    return modelos

# 3. FUNCIONES DE PROBABILIDAD

def ajuste_dixon_coles(goles_home, goles_away, lambda_home, lambda_away, rho=RHO_DIXON_COLES):
    if goles_home == 0 and goles_away == 0:
        return 1 - (lambda_home * lambda_away * rho)
    elif goles_home == 0 and goles_away == 1:
        return 1 + (lambda_home * rho)
    elif goles_home == 1 and goles_away == 0:
        return 1 + (lambda_away * rho)
    elif goles_home == 1 and goles_away == 1:
        return 1 - rho
    else:
        return 1


def crear_matriz_probabilidades(lambda_home, lambda_away, max_goles=MAX_GOLES):
    lambda_home = max(float(lambda_home), 0.05)
    lambda_away = max(float(lambda_away), 0.05)

    matriz = np.zeros((max_goles + 1, max_goles + 1))

    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            prob_home = poisson.pmf(i, lambda_home)
            prob_away = poisson.pmf(j, lambda_away)
            ajuste = ajuste_dixon_coles(i, j, lambda_home, lambda_away)

            matriz[i, j] = prob_home * prob_away * ajuste

    matriz = matriz / matriz.sum()

    return matriz


def probabilidades_resultado(matriz):
    prob_home = np.tril(matriz, -1).sum()
    prob_draw = np.trace(matriz)
    prob_away = np.triu(matriz, 1).sum()

    return prob_home, prob_draw, prob_away


def clase_por_probabilidades(prob_home, prob_draw, prob_away):
    probabilidades = {
        "H": prob_home,
        "D": prob_draw,
        "A": prob_away
    }

    return max(probabilidades, key=lambda clase: probabilidades[clase])


def resultado_real(home_score, away_score):
    if home_score > away_score:
        return "H"
    elif home_score == away_score:
        return "D"
    else:
        return "A"


def obtener_top10(matriz, home_team, away_team, match_id):
    filas = []

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):

            if i > j:
                resultado = f"Victoria {home_team}"
            elif i == j:
                resultado = "Empate"
            else:
                resultado = f"Victoria {away_team}"

            filas.append({
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "score": f"{i}-{j}",
                "home_goals": i,
                "away_goals": j,
                "probability": matriz[i, j],
                "probability_percent": matriz[i, j] * 100,
                "result": resultado
            })

    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values("probability", ascending=False).head(10).reset_index(drop=True)
    tabla["rank"] = tabla.index + 1

    return tabla[[
        "match_id",
        "rank",
        "home_team",
        "away_team",
        "score",
        "probability_percent",
        "result"
    ]]

# 4. CARGA DE DATA

def cargar_data():
    if not ARCHIVO_DATA_MODELO.exists():
        raise FileNotFoundError("Primero ejecuta: python 01_construir_data_modelo.py")

    data = pd.read_csv(ARCHIVO_DATA_MODELO)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    data = data.dropna(subset=[
        "date",
        "home_score",
        "away_score"
    ])

    for col in VARIABLES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["home_score"] = pd.to_numeric(data["home_score"], errors="coerce")
    data["away_score"] = pd.to_numeric(data["away_score"], errors="coerce")

    data = data.dropna(subset=VARIABLES + ["home_score", "away_score"])

    data["home_score"] = data["home_score"].astype(float)
    data["away_score"] = data["away_score"].astype(float)

    data = data.sort_values("date").reset_index(drop=True)

    return data


def calcular_pesos_temporales(train):
    fecha_max = train["date"].max()
    dias_antiguedad = (fecha_max - train["date"]).dt.days

    pesos = np.exp(-0.001 * dias_antiguedad)

    return pesos

# 5. ENTRENAMIENTO CON PESOS

def entrenar_con_pesos(modelo, X, y, pesos):
    try:
        if isinstance(modelo, Pipeline):
            ultimo_paso = list(modelo.named_steps.keys())[-1]
            modelo.fit(X, y, **{ultimo_paso + "__sample_weight": pesos})
        else:
            modelo.fit(X, y, sample_weight=pesos)
    except TypeError:
        modelo.fit(X, y)

    return modelo

# 6. COMPETENCIA DE MODELOS

def competencia_modelos(data):
    corte = int(len(data) * 0.80)

    train = data.iloc[:corte].copy()
    test = data.iloc[corte:].copy()

    X_train = train[VARIABLES]
    X_test = test[VARIABLES]

    y_home_train = train["home_score"]
    y_away_train = train["away_score"]

    y_home_test = test["home_score"]
    y_away_test = test["away_score"]

    pesos = calcular_pesos_temporales(train)

    modelos_candidatos = crear_modelos_candidatos()

    resultados = []

    print()
    print("========================================")
    print("INICIANDO COMPETENCIA DE MODELOS")
    print("========================================")

    for nombre in modelos_candidatos.keys():

        print("Entrenando:", nombre)

        modelo_home = crear_modelos_candidatos()[nombre]
        modelo_away = crear_modelos_candidatos()[nombre]

        modelo_home = entrenar_con_pesos(modelo_home, X_train, y_home_train, pesos)
        modelo_away = entrenar_con_pesos(modelo_away, X_train, y_away_train, pesos)

        pred_home = modelo_home.predict(X_test)
        pred_away = modelo_away.predict(X_test)

        pred_home = np.maximum(pred_home, 0.05)
        pred_away = np.maximum(pred_away, 0.05)

        mae_home = mean_absolute_error(y_home_test, pred_home)
        mae_away = mean_absolute_error(y_away_test, pred_away)

        rmse_home = np.sqrt(mean_squared_error(y_home_test, pred_home))
        rmse_away = np.sqrt(mean_squared_error(y_away_test, pred_away))

        r2_home = r2_score(y_home_test, pred_home)
        r2_away = r2_score(y_away_test, pred_away)

        mae_promedio = (mae_home + mae_away) / 2
        rmse_promedio = (rmse_home + rmse_away) / 2
        r2_promedio = (r2_home + r2_away) / 2

        reales = []
        predichos = []

        for idx, fila in test.iterrows():
            X_fila = fila[VARIABLES].to_frame().T

            lambda_home = modelo_home.predict(X_fila)[0]
            lambda_away = modelo_away.predict(X_fila)[0]

            lambda_home = max(float(lambda_home), 0.05)
            lambda_away = max(float(lambda_away), 0.05)

            matriz = crear_matriz_probabilidades(lambda_home, lambda_away)

            prob_home, prob_draw, prob_away = probabilidades_resultado(matriz)

            clase_predicha = clase_por_probabilidades(
                prob_home,
                prob_draw,
                prob_away
            )

            clase_real = resultado_real(
                fila["home_score"],
                fila["away_score"]
            )

            reales.append(clase_real)
            predichos.append(clase_predicha)

        accuracy = accuracy_score(reales, predichos)

        matriz_conf = confusion_matrix(
            reales,
            predichos,
            labels=["H", "D", "A"]
        )

        resultados.append({
            "modelo": nombre,
            "mae_home": mae_home,
            "mae_away": mae_away,
            "mae_promedio": mae_promedio,
            "rmse_home": rmse_home,
            "rmse_away": rmse_away,
            "rmse_promedio": rmse_promedio,
            "r2_home": r2_home,
            "r2_away": r2_away,
            "r2_promedio": r2_promedio,
            "accuracy_resultado": accuracy,
            "confusion_HH": matriz_conf[0, 0],
            "confusion_HD": matriz_conf[0, 1],
            "confusion_HA": matriz_conf[0, 2],
            "confusion_DH": matriz_conf[1, 0],
            "confusion_DD": matriz_conf[1, 1],
            "confusion_DA": matriz_conf[1, 2],
            "confusion_AH": matriz_conf[2, 0],
            "confusion_AD": matriz_conf[2, 1],
            "confusion_AA": matriz_conf[2, 2],
        })

    tabla_resultados = pd.DataFrame(resultados)

    tabla_resultados = tabla_resultados.sort_values(
        by=["mae_promedio", "rmse_promedio"],
        ascending=[True, True]
    ).reset_index(drop=True)

    tabla_resultados.to_csv(
        OUT_DIR / "competencia_modelos.csv",
        index=False,
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("RESULTADOS DE LA COMPETENCIA")
    print("========================================")
    print(tabla_resultados[[
        "modelo",
        "mae_home",
        "mae_away",
        "mae_promedio",
        "rmse_promedio",
        "r2_promedio",
        "accuracy_resultado"
    ]].to_string(index=False))
    print("========================================")

    mejor_modelo = tabla_resultados.iloc[0]["modelo"]

    print()
    print("MEJOR MODELO SEGUN MAE PROMEDIO:", mejor_modelo)
    print()

    with open(OUT_DIR / "mejor_modelo.txt", "w", encoding="utf-8") as f:
        f.write(str(mejor_modelo))

    return mejor_modelo, tabla_resultados


# 7. ENTRENAMIENTO FINAL DEL MODELO GANADOR

def entrenar_modelo_final(data, mejor_modelo):
    X = data[VARIABLES]
    y_home = data["home_score"]
    y_away = data["away_score"]

    pesos = calcular_pesos_temporales(data)

    modelo_home = crear_modelos_candidatos()[mejor_modelo]
    modelo_away = crear_modelos_candidatos()[mejor_modelo]

    modelo_home = entrenar_con_pesos(modelo_home, X, y_home, pesos)
    modelo_away = entrenar_con_pesos(modelo_away, X, y_away, pesos)

    joblib.dump(modelo_home, OUT_DIR / "modelo_home.pkl")
    joblib.dump(modelo_away, OUT_DIR / "modelo_away.pkl")

    print("Modelos finales guardados:")
    print(OUT_DIR / "modelo_home.pkl")
    print(OUT_DIR / "modelo_away.pkl")

    return modelo_home, modelo_away

# 8. PREDICCIÓN DEL FIXTURE

def predecir_fixture(modelo_home, modelo_away):
    if not ARCHIVO_FIXTURES_MODELO.exists():
        raise FileNotFoundError("Primero ejecuta: python 01_construir_data_modelo.py")

    fixtures = pd.read_csv(ARCHIVO_FIXTURES_MODELO)

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")

    for col in VARIABLES:
        fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce")

    fixtures = fixtures.dropna(subset=VARIABLES).reset_index(drop=True)

    predicciones = []
    top10_total = []

    for match_id_fallback, (_, fila) in enumerate(fixtures.iterrows(), start=1):

        X_nuevo = fila[VARIABLES].to_frame().T

        lambda_home = modelo_home.predict(X_nuevo)[0]
        lambda_away = modelo_away.predict(X_nuevo)[0]

        lambda_home = max(float(lambda_home), 0.05)
        lambda_away = max(float(lambda_away), 0.05)

        matriz = crear_matriz_probabilidades(lambda_home, lambda_away)

        prob_home, prob_draw, prob_away = probabilidades_resultado(matriz)

        clase = clase_por_probabilidades(
            prob_home,
            prob_draw,
            prob_away
        )

        if "match_no" in fixtures.columns and not pd.isna(fila["match_no"]):
            match_id = int(fila["match_no"])
        else:
            match_id = match_id_fallback

        predicciones.append({
            "match_id": match_id,
            "date": fila["date"].date(),
            "stage": fila["stage"] if "stage" in fixtures.columns else "Group stage",
            "group": fila["group"] if "group" in fixtures.columns else "",
            "home_team": fila["home_team"],
            "away_team": fila["away_team"],
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "prob_home_win": prob_home,
            "prob_draw": prob_draw,
            "prob_away_win": prob_away,
            "prob_home_win_percent": prob_home * 100,
            "prob_draw_percent": prob_draw * 100,
            "prob_away_win_percent": prob_away * 100,
            "predicted_class": clase
        })

        top10 = obtener_top10(
            matriz=matriz,
            home_team=fila["home_team"],
            away_team=fila["away_team"],
            match_id=match_id
        )

        top10_total.append(top10)

    predicciones = pd.DataFrame(predicciones)
    top10_total = pd.concat(top10_total, ignore_index=True)

    predicciones.to_csv(
        OUT_DIR / "predicciones_fixture.csv",
        index=False,
        encoding="utf-8"
    )

    top10_total.to_csv(
        OUT_DIR / "top10_marcadores.csv",
        index=False,
        encoding="utf-8"
    )

    print()
    print("Predicciones guardadas:")
    print(OUT_DIR / "predicciones_fixture.csv")
    print(OUT_DIR / "top10_marcadores.csv")

    print()
    print("Primeras predicciones:")
    print(predicciones.head(10).to_string(index=False))

    return predicciones, top10_total

# ENTRENAR TODOS LOS MODELOS FINALES

def entrenar_todos_los_modelos_finales(data, tabla_resultados):
    X = data[VARIABLES]
    y_home = data["home_score"]
    y_away = data["away_score"]

    pesos = calcular_pesos_temporales(data)

    modelos_home = {}
    modelos_away = {}

    print()
    print("========================================")
    print("ENTRENAMIENTO FINAL DE TODOS LOS MODELOS")
    print("========================================")

    for nombre_modelo in tabla_resultados["modelo"]:

        print("Entrenando modelo final:", nombre_modelo)

        modelo_home = crear_modelos_candidatos()[nombre_modelo]
        modelo_away = crear_modelos_candidatos()[nombre_modelo]

        modelo_home = entrenar_con_pesos(modelo_home, X, y_home, pesos)
        modelo_away = entrenar_con_pesos(modelo_away, X, y_away, pesos)

        modelos_home[nombre_modelo] = modelo_home
        modelos_away[nombre_modelo] = modelo_away

        nombre_archivo = nombre_modelo.lower().replace(" ", "_")

        joblib.dump(
            modelo_home,
            OUT_DIR / f"modelo_home_{nombre_archivo}.pkl"
        )

        joblib.dump(
            modelo_away,
            OUT_DIR / f"modelo_away_{nombre_archivo}.pkl"
        )

    print("Todos los modelos fueron entrenados y guardados.")

    return modelos_home, modelos_away

# PREDICCION DEL FIXTURE CON TODOS LOS MODELOS
def predecir_fixture_todos_modelos(modelos_home, modelos_away, mejor_modelo):
    if not ARCHIVO_FIXTURES_MODELO.exists():
        raise FileNotFoundError("Primero ejecuta: python 01_construir_data_modelo.py")

    fixtures = pd.read_csv(ARCHIVO_FIXTURES_MODELO)
    contexto_grupos = cargar_contexto_grupos()

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")

    for col in VARIABLES:
        fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce")

    fixtures = fixtures.dropna(subset=VARIABLES).reset_index(drop=True)

    predicciones_total = []
    top10_total = []

    for nombre_modelo in modelos_home.keys():

        print("Generando predicciones con:", nombre_modelo)

        modelo_home = modelos_home[nombre_modelo]
        modelo_away = modelos_away[nombre_modelo]

        for match_id_fallback, (_, fila) in enumerate(fixtures.iterrows(), start=1):

            X_nuevo = fila[VARIABLES].to_frame().T
            lambda_home = modelo_home.predict(X_nuevo)[0]
            lambda_away = modelo_away.predict(X_nuevo)[0]

            lambda_home = max(float(lambda_home), 0.05)
            lambda_away = max(float(lambda_away), 0.05)

            factor_contexto_home = 1.00
            factor_contexto_away = 1.00

            fila_contexto = obtener_contexto_partido(
                contexto_grupos=contexto_grupos,
                fila=fila
            )

            if fila_contexto is not None:
                lambda_home, lambda_away, factor_contexto_home, factor_contexto_away = ajustar_lambdas_por_contexto(
                    lambda_home=lambda_home,
                    lambda_away=lambda_away,
                    fila_contexto=fila_contexto
                )

            matriz = crear_matriz_probabilidades(lambda_home, lambda_away)
            
            prob_home, prob_draw, prob_away = probabilidades_resultado(matriz)

            clase = clase_por_probabilidades(
                prob_home,
                prob_draw,
                prob_away
            )

            if "match_no" in fixtures.columns and not pd.isna(fila["match_no"]):
                match_id = int(fila["match_no"])
            else:
                match_id = match_id_fallback

            predicciones_total.append({
                "match_id": match_id,
                "modelo": nombre_modelo,
                "modelo_ganador_global": nombre_modelo == mejor_modelo,
                "date": fila["date"].date(),
                "stage": fila["stage"] if "stage" in fixtures.columns else "Group stage",
                "group": fila["group"] if "group" in fixtures.columns else "",
                "home_team": fila["home_team"],
                "away_team": fila["away_team"],
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "factor_contexto_home": factor_contexto_home,
                "factor_contexto_away": factor_contexto_away,
                "prob_home_win": prob_home,
                "prob_draw": prob_draw,
                "prob_away_win": prob_away,
                "prob_home_win_percent": prob_home * 100,
                "prob_draw_percent": prob_draw * 100,
                "prob_away_win_percent": prob_away * 100,
                "predicted_class": clase
            })

            top10 = obtener_top10(
                matriz=matriz,
                home_team=fila["home_team"],
                away_team=fila["away_team"],
                match_id=match_id
            )

            top10["modelo"] = nombre_modelo
            top10["modelo_ganador_global"] = nombre_modelo == mejor_modelo

            top10_total.append(top10)

    predicciones_total = pd.DataFrame(predicciones_total)
    top10_total = pd.concat(top10_total, ignore_index=True)

    predicciones_total.to_csv(
        OUT_DIR / "predicciones_todos_modelos.csv",
        index=False,
        encoding="utf-8"
    )

    top10_total.to_csv(
        OUT_DIR / "top10_todos_modelos.csv",
        index=False,
        encoding="utf-8"
    )

    # También se guarda la predicción solo del mejor modelo
    predicciones_mejor = predicciones_total[
        predicciones_total["modelo"] == mejor_modelo
    ].copy()

    top10_mejor = top10_total[
        top10_total["modelo"] == mejor_modelo
    ].copy()

    predicciones_mejor.to_csv(
        OUT_DIR / "predicciones_fixture.csv",
        index=False,
        encoding="utf-8"
    )

    top10_mejor.to_csv(
        OUT_DIR / "top10_marcadores.csv",
        index=False,
        encoding="utf-8"
    )

    print()
    print("Predicciones generadas:")
    print(OUT_DIR / "predicciones_todos_modelos.csv")
    print(OUT_DIR / "top10_todos_modelos.csv")
    print(OUT_DIR / "predicciones_fixture.csv")
    print(OUT_DIR / "top10_marcadores.csv")

    print()
    print("Primeras predicciones de todos los modelos:")
    print(predicciones_total.head(15).to_string(index=False))

    return predicciones_total, top10_total

# cargando contexto de fase de grupos etapas finales
def cargar_contexto_grupos():
    if not ARCHIVO_CONTEXTO_GRUPOS.exists():
        print("No existe fixtures_contexto_grupos.csv. Se usara prediccion sin ajuste contextual.")
        return None

    contexto = pd.read_csv(ARCHIVO_CONTEXTO_GRUPOS)

    contexto["date"] = pd.to_datetime(contexto["date"]).dt.strftime("%Y-%m-%d")

    columnas_necesarias = [
        "date",
        "home_team",
        "away_team",
        "home_must_win",
        "away_must_win",
        "home_need_goal_difference",
        "away_need_goal_difference",
        "home_rotation_risk",
        "away_rotation_risk",
        "home_already_qualified",
        "away_already_qualified",
    ]

    columnas_faltantes = [
        col for col in columnas_necesarias if col not in contexto.columns
    ]

    if columnas_faltantes:
        print("Faltan columnas de contexto:")
        print(columnas_faltantes)
        return None

    return contexto

def obtener_flag_contexto(fila_contexto, columna):
    if columna not in fila_contexto.index:
        return 0

    valor = fila_contexto[columna]

    if pd.isna(valor):
        return 0

    return int(valor)


def ajustar_lambdas_por_contexto(lambda_home, lambda_away, fila_contexto):
    factor_home = 1.00
    factor_away = 1.00

    # ========================================================
    # LOCAL
    # ========================================================

    home_must_win = obtener_flag_contexto(fila_contexto, "home_must_win")
    home_needs_result = obtener_flag_contexto(fila_contexto, "home_needs_result")
    home_playing_for_first = obtener_flag_contexto(fila_contexto, "home_playing_for_first")
    home_risk_drop_to_third = obtener_flag_contexto(fila_contexto, "home_risk_drop_to_third")
    home_need_goal_difference = obtener_flag_contexto(fila_contexto, "home_need_goal_difference")
    home_rotation_risk = obtener_flag_contexto(fila_contexto, "home_rotation_risk")
    home_already_qualified = obtener_flag_contexto(fila_contexto, "home_already_qualified")

    if home_must_win == 1:
        factor_home *= 1.10
    elif home_needs_result == 1:
        factor_home *= 1.05

    if home_playing_for_first == 1:
        factor_home *= 1.03

    if home_risk_drop_to_third == 1:
        factor_home *= 1.05

    if home_need_goal_difference == 1:
        factor_home *= 1.05

    if home_rotation_risk == 1:
        factor_home *= 0.90
    elif home_already_qualified == 1 and home_playing_for_first == 0 and home_risk_drop_to_third == 0:
        factor_home *= 0.97

    # ========================================================
    # VISITANTE
    # ========================================================

    away_must_win = obtener_flag_contexto(fila_contexto, "away_must_win")
    away_needs_result = obtener_flag_contexto(fila_contexto, "away_needs_result")
    away_playing_for_first = obtener_flag_contexto(fila_contexto, "away_playing_for_first")
    away_risk_drop_to_third = obtener_flag_contexto(fila_contexto, "away_risk_drop_to_third")
    away_need_goal_difference = obtener_flag_contexto(fila_contexto, "away_need_goal_difference")
    away_rotation_risk = obtener_flag_contexto(fila_contexto, "away_rotation_risk")
    away_already_qualified = obtener_flag_contexto(fila_contexto, "away_already_qualified")

    if away_must_win == 1:
        factor_away *= 1.10
    elif away_needs_result == 1:
        factor_away *= 1.05

    if away_playing_for_first == 1:
        factor_away *= 1.03

    if away_risk_drop_to_third == 1:
        factor_away *= 1.05

    if away_need_goal_difference == 1:
        factor_away *= 1.05

    if away_rotation_risk == 1:
        factor_away *= 0.90
    elif away_already_qualified == 1 and away_playing_for_first == 0 and away_risk_drop_to_third == 0:
        factor_away *= 0.97

    lambda_home_ajustada = lambda_home * factor_home
    lambda_away_ajustada = lambda_away * factor_away

    lambda_home_ajustada = max(lambda_home_ajustada, 0.05)
    lambda_away_ajustada = max(lambda_away_ajustada, 0.05)

    return lambda_home_ajustada, lambda_away_ajustada, factor_home, factor_away

def obtener_contexto_partido(contexto_grupos, fila):
    if contexto_grupos is None:
        return None

    fecha_partido = pd.to_datetime(fila["date"]).strftime("%Y-%m-%d")

    condicion = (
        (contexto_grupos["date"] == fecha_partido) &
        (contexto_grupos["home_team"] == fila["home_team"]) &
        (contexto_grupos["away_team"] == fila["away_team"])
    )

    if condicion.any():
        return contexto_grupos[condicion].iloc[0]

    return None

# 9. EJECUCIÓN PRINCIPAL

def main():
    data = cargar_data()

    print()
    print("========================================")
    print("DATA CARGADA")
    print("========================================")
    print("Filas:", data.shape[0])
    print("Columnas:", data.shape[1])
    print("Fecha mínima:", data["date"].min())
    print("Fecha máxima:", data["date"].max())
    print("========================================")

    mejor_modelo, tabla_resultados = competencia_modelos(data)

    modelos_home, modelos_away = entrenar_todos_los_modelos_finales(data, tabla_resultados)

    predecir_fixture_todos_modelos(modelos_home, modelos_away, mejor_modelo)

    print()
    print("========================================")
    print("PROCESO COMPLETADO")
    print("========================================")
    print("Archivos generados en outputs/:")
    print("1. competencia_modelos.csv")
    print("2. mejor_modelo.txt")
    print("3. modelo_home.pkl")
    print("4. modelo_away.pkl")
    print("5. predicciones_fixture.csv")
    print("6. top10_marcadores.csv")
    print("========================================")


if __name__ == "__main__":
    main()
