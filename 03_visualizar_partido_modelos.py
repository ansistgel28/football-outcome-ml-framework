# ============================================================
# 03_visualizar_partido.py
# Visualizar predicciones de todos los modelos por match_id
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_PREDICCIONES = OUT_DIR / "predicciones_todos_modelos.csv"
ARCHIVO_TOP10 = OUT_DIR / "top10_todos_modelos.csv"
ARCHIVO_COMPETENCIA = OUT_DIR / "competencia_modelos.csv"
ARCHIVO_MEJOR_MODELO = OUT_DIR / "mejor_modelo.txt"

# Cambia este número según el partido que quieras visualizar
#partido= n-1 segun el fixtures_modelo.csv
MATCH_ID = 65


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

def cargar_archivos():
    if not ARCHIVO_PREDICCIONES.exists():
        raise FileNotFoundError("No existe outputs/predicciones_todos_modelos.csv")

    if not ARCHIVO_TOP10.exists():
        raise FileNotFoundError("No existe outputs/top10_todos_modelos.csv")

    if not ARCHIVO_COMPETENCIA.exists():
        raise FileNotFoundError("No existe outputs/competencia_modelos.csv")

    pred = pd.read_csv(ARCHIVO_PREDICCIONES)
    top10 = pd.read_csv(ARCHIVO_TOP10)
    competencia = pd.read_csv(ARCHIVO_COMPETENCIA)

    if ARCHIVO_MEJOR_MODELO.exists():
        mejor_modelo = ARCHIVO_MEJOR_MODELO.read_text(encoding="utf-8").strip()
    else:
        mejor_modelo = competencia.sort_values("mae_promedio").iloc[0]["modelo"]

    return pred, top10, competencia, mejor_modelo

# ============================================================
# MOSTRAR CONTEXTO COMPETITIVO DEL PARTIDO EN TABLA
# ============================================================

def interpretar_factor_contexto(factor):
    if factor > 1:
        return "Aumenta intensidad ofensiva"
    elif factor < 1:
        return "Reduce intensidad ofensiva"
    else:
        return "Sin ajuste contextual"


def mostrar_contexto_partido(pred_partido):
    columnas_contexto = [
        "factor_contexto_home",
        "factor_contexto_away"
    ]

    for col in columnas_contexto:
        if col not in pred_partido.columns:
            print()
            print("No se encontraron columnas de contexto en predicciones_todos_modelos.csv")
            return

    fila = pred_partido.iloc[0]

    equipo_local = fila["home_team"]
    equipo_visitante = fila["away_team"]

    factor_local = float(fila["factor_contexto_home"])
    factor_visitante = float(fila["factor_contexto_away"])

    tabla_contexto = pd.DataFrame([
        {
            "rol": "LOCAL",
            "columna_modelo": "home_team",
            "equipo": equipo_local,
            "factor_contexto": round(factor_local, 3),
            "interpretacion": interpretar_factor_contexto(factor_local)
        },
        {
            "rol": "VISITANTE",
            "columna_modelo": "away_team",
            "equipo": equipo_visitante,
            "factor_contexto": round(factor_visitante, 3),
            "interpretacion": interpretar_factor_contexto(factor_visitante)
        }
    ])

    print()
    print("========================================")
    print("CONTEXTO COMPETITIVO DEL PARTIDO")
    print("========================================")
    print(tabla_contexto.to_string(index=False))
    print("========================================")

# ============================================================
# GRAFICAR CONTEXTO COMPETITIVO DEL PARTIDO
# ============================================================

def graficar_contexto_partido(pred_partido):
    columnas_necesarias = [
        "factor_contexto_home",
        "factor_contexto_away"
    ]

    for col in columnas_necesarias:
        if col not in pred_partido.columns:
            print("No se encontraron columnas de contexto para graficar.")
            return

    fila = pred_partido.iloc[0]

    equipo_local = fila["home_team"]
    equipo_visitante = fila["away_team"]

    factor_local = float(fila["factor_contexto_home"])
    factor_visitante = float(fila["factor_contexto_away"])

    def interpretar_factor(factor):
        if factor > 1:
            return "Aumenta intensidad ofensiva"
        elif factor < 1:
            return "Reduce intensidad ofensiva"
        else:
            return "Sin ajuste contextual"

    tabla_contexto = pd.DataFrame([
        {
            "Rol": "LOCAL",
            "Equipo": equipo_local,
            "Columna": "home_team",
            "Factor": round(factor_local, 3),
            "Interpretacion": interpretar_factor(factor_local)
        },
        {
            "Rol": "VISITANTE",
            "Equipo": equipo_visitante,
            "Columna": "away_team",
            "Factor": round(factor_visitante, 3),
            "Interpretacion": interpretar_factor(factor_visitante)
        }
    ])

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    tabla_plot = ax.table(
        cellText=tabla_contexto.astype(str).values.tolist(),
        colLabels=tabla_contexto.columns.tolist(),
        cellLoc="center",
        loc="center"
    )

    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(10)
    tabla_plot.scale(1, 1.6)

    plt.title(
        "Contexto competitivo del partido\n"
        + "MATCH_ID "
        + str(MATCH_ID)
        + ": "
        + equipo_local
        + " (LOCAL) vs "
        + equipo_visitante
        + " (VISITANTE)",
        fontsize=13
    )

    plt.tight_layout()

    ruta_salida = OUT_DIR / f"contexto_match_{MATCH_ID}.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")

    print("Tabla de contexto competitivo guardada:", ruta_salida)

    plt.show()

# ============================================================
# TABLA COMPARATIVA DE TODOS LOS MODELOS
# ============================================================

def mostrar_comparacion_modelos(pred_partido, competencia, mejor_modelo):
    tabla = pred_partido.copy()

    tabla = tabla.merge(
        competencia[[
            "modelo",
            "mae_promedio",
            "rmse_promedio",
            "r2_promedio",
            "accuracy_resultado"
        ]],
        on="modelo",
        how="left"
    )

    tabla["modelo_ganador"] = tabla["modelo"].apply(
        lambda x: "SI" if x == mejor_modelo else "NO"
    )

    columnas = [
        "modelo",
        "modelo_ganador",
        "home_team",
        "away_team",
        "lambda_home",
        "lambda_away",
        "factor_contexto_home",
        "factor_contexto_away",
        "prob_home_win_percent",
        "prob_draw_percent",
        "prob_away_win_percent",
        "predicted_class",
        "mae_promedio",
        "rmse_promedio",
        "r2_promedio",
        "accuracy_resultado"    
    ]

    tabla = tabla[columnas].copy()

    columnas_redondear = [
        "lambda_home",
        "lambda_away",
        "factor_contexto_home",
        "factor_contexto_away",
        "prob_home_win_percent",
        "prob_draw_percent",
        "prob_away_win_percent",
        "mae_promedio",
        "rmse_promedio",
        "r2_promedio",
        "accuracy_resultado"
    ]

    for col in columnas_redondear:
        tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(3)

    print()
    print("========================================")
    print("COMPARACION DE PREDICCIONES POR MODELO")
    print("========================================")
    print(tabla.to_string(index=False))
    print("========================================")

    return tabla


# ============================================================
# GRAFICO COMPARATIVO DE PROBABILIDADES
# ============================================================

def graficar_comparacion(tabla, home_team, away_team):
    modelos = tabla["modelo"].tolist()
    x = list(range(len(modelos)))

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - 0.25 for i in x],
        tabla["prob_home_win_percent"],
        width=0.25,
        label="Victoria " + home_team
    )

    plt.bar(
        x,
        tabla["prob_draw_percent"],
        width=0.25,
        label="Empate"
    )

    plt.bar(
        [i + 0.25 for i in x],
        tabla["prob_away_win_percent"],
        width=0.25,
        label="Victoria " + away_team
    )

    plt.xticks(x, modelos, rotation=30, ha="right")
    plt.ylabel("Probabilidad (%)")
    plt.title("Comparación de probabilidades por modelo")
    plt.legend()
    plt.tight_layout()

    ruta_salida = OUT_DIR / f"comparacion_modelos_match_{MATCH_ID}.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")

    print("Gráfico comparativo guardado:", ruta_salida)

    plt.show()


# ============================================================
# MOSTRAR TOP 10 EN CONSOLA
# ============================================================

def mostrar_top10_por_modelo(top10_partido, mejor_modelo):
    modelos = top10_partido["modelo"].unique()

    print()
    print("========================================")
    print("TOP 10 MARCADORES POR MODELO")
    print("========================================")

    for modelo in modelos:
        tabla_modelo = top10_partido[top10_partido["modelo"] == modelo].copy()

        etiqueta = "  <-- MODELO GANADOR" if modelo == mejor_modelo else ""

        print()
        print("Modelo:", modelo + etiqueta)

        tabla_mostrar = tabla_modelo[[
            "rank",
            "score",
            "probability_percent",
            "result"
        ]].copy()

        tabla_mostrar["probability_percent"] = tabla_mostrar["probability_percent"].round(3)

        print(tabla_mostrar.to_string(index=False))


# ============================================================
# GRAFICAR TABLA TOP 10 DE UN MODELO
# ============================================================

def graficar_top10_modelo(tabla_top10, home_team, away_team, modelo, mejor_modelo):
    if tabla_top10.empty:
        print("No hay Top 10 para el modelo:", modelo)
        return

    tabla_visual = tabla_top10.copy()

    tabla_visual = tabla_visual[[
        "rank",
        "score",
        "probability_percent",
        "result"
    ]].copy()

    tabla_visual["probability_percent"] = (
        tabla_visual["probability_percent"].round(2).astype(str) + "%"
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")

    tabla_plot = ax.table(
        cellText=tabla_visual.astype(str).values.tolist(),
        colLabels=tabla_visual.columns.tolist(),
        cellLoc="center",
        loc="center"
    )

    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(10)
    tabla_plot.scale(1, 1.5)

    etiqueta_ganador = " - MODELO GANADOR" if modelo == mejor_modelo else ""

    plt.title(
        "Top 10 marcadores: "
        + home_team
        + " vs "
        + away_team
        + "\nModelo: "
        + modelo
        + etiqueta_ganador,
        fontsize=13
    )

    plt.tight_layout()

    nombre_archivo = (
        "top10_match_"
        + str(MATCH_ID)
        + "_"
        + modelo.lower().replace(" ", "_")
        + ".png"
    )

    ruta_salida = OUT_DIR / nombre_archivo
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")

    print("Tabla Top 10 guardada:", ruta_salida)

    plt.show()


# ============================================================
# GRAFICAR TABLAS TOP 10 DE TODOS LOS MODELOS
# ============================================================

def graficar_top10_todos_los_modelos(top10_partido, home_team, away_team, mejor_modelo):
    modelos = top10_partido["modelo"].unique()

    for modelo in modelos:
        tabla_modelo = top10_partido[top10_partido["modelo"] == modelo].copy()

        graficar_top10_modelo(
            tabla_top10=tabla_modelo,
            home_team=home_team,
            away_team=away_team,
            modelo=modelo,
            mejor_modelo=mejor_modelo
        )


# ============================================================
# GRAFICO PASTEL DEL MODELO GANADOR
# ============================================================

def graficar_pastel_modelo_ganador(pred_partido, mejor_modelo):
    fila = pred_partido[pred_partido["modelo"] == mejor_modelo]

    if fila.empty:
        print("No se encontró el modelo ganador en este partido.")
        return

    fila = fila.iloc[0]

    etiquetas = [
        "Victoria " + fila["home_team"],
        "Empate",
        "Victoria " + fila["away_team"]
    ]

    valores = [
        fila["prob_home_win_percent"],
        fila["prob_draw_percent"],
        fila["prob_away_win_percent"]
    ]

    plt.figure(figsize=(7, 7))
    plt.pie(valores, labels=etiquetas, autopct="%1.2f%%", startangle=90)
    plt.title("Modelo ganador: " + mejor_modelo)
    plt.tight_layout()

    ruta_salida = OUT_DIR / f"pastel_match_{MATCH_ID}_{mejor_modelo.lower().replace(' ', '_')}.png"
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")

    print("Gráfico pastel guardado:", ruta_salida)

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    pred, top10, competencia, mejor_modelo = cargar_archivos()

    pred_partido = pred[pred["match_id"] == MATCH_ID].copy()
    top10_partido = top10[top10["match_id"] == MATCH_ID].copy()

    if pred_partido.empty:
        print("No existe ese MATCH_ID:", MATCH_ID)
        return

    if top10_partido.empty:
        print("No existe Top 10 para ese MATCH_ID:", MATCH_ID)
        return

    home_team = pred_partido.iloc[0]["home_team"]
    away_team = pred_partido.iloc[0]["away_team"]

    print()
    print("========================================")
    print("PARTIDO SELECCIONADO")
    print("========================================")
    print("MATCH_ID:", MATCH_ID)
    print("Equipo LOCAL:", home_team)
    print("Equipo VISITANTE:", away_team)
    print("Modelo ganador global:", mejor_modelo)
    print("========================================")

    mostrar_contexto_partido(pred_partido)
    graficar_contexto_partido(pred_partido)

    tabla_comparacion = mostrar_comparacion_modelos(
        pred_partido=pred_partido,
        competencia=competencia,
        mejor_modelo=mejor_modelo
    )

    graficar_comparacion(
        tabla=tabla_comparacion,
        home_team=home_team,
        away_team=away_team
    )

    mostrar_top10_por_modelo(
        top10_partido=top10_partido,
        mejor_modelo=mejor_modelo
    )

    graficar_top10_todos_los_modelos(
        top10_partido=top10_partido,
        home_team=home_team,
        away_team=away_team,
        mejor_modelo=mejor_modelo
    )

    graficar_pastel_modelo_ganador(
        pred_partido=pred_partido,
        mejor_modelo=mejor_modelo
    )


if __name__ == "__main__":
    main()
