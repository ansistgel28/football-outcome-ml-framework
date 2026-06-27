import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ARCHIVO_RESULTS = DATA_DIR / "results.csv"
ARCHIVO_BACKUP = DATA_DIR / "results_backup_antes_mundial_2026.csv"

TORNEO = "FIFA World Cup"
CIUDAD = "World Cup 2026"
PAIS = "Canada/Mexico/United States"
NEUTRAL = True


# ============================================================
# PARTIDOS YA JUGADOS DEL MUNDIAL 2026
# ============================================================

partidos_mundial_2026 = [
    ("2026-06-11", "Mexico", "South Africa", 2, 0),
    ("2026-06-11", "South Korea", "Czech Republic", 2, 1),

    ("2026-06-12", "Canada", "Bosnia and Herzegovina", 1, 1),
    ("2026-06-12", "United States", "Paraguay", 4, 1),

    ("2026-06-13", "Qatar", "Switzerland", 1, 1),
    ("2026-06-13", "Brazil", "Morocco", 1, 1),
    ("2026-06-13", "Haiti", "Scotland", 0, 1),

    ("2026-06-14", "Australia", "Turkey", 2, 0),
    ("2026-06-14", "Germany", "Curacao", 7, 1),
    ("2026-06-14", "Netherlands", "Japan", 2, 2),
    ("2026-06-14", "Ivory Coast", "Ecuador", 1, 0),
    ("2026-06-14", "Sweden", "Tunisia", 5, 1),

    ("2026-06-15", "Spain", "Cape Verde", 0, 0),
    ("2026-06-15", "Belgium", "Egypt", 1, 1),
    ("2026-06-15", "Saudi Arabia", "Uruguay", 1, 1),
    ("2026-06-15", "Iran", "New Zealand", 2, 2),

    ("2026-06-16", "France", "Senegal", 3, 1),
    ("2026-06-16", "Iraq", "Norway", 1, 4),
    ("2026-06-16", "Argentina", "Algeria", 3, 0),

    ("2026-06-17", "Austria", "Jordan", 3, 1),
    ("2026-06-17", "Portugal", "DR Congo", 1, 1),
    ("2026-06-17", "England", "Croatia", 4, 2),
    ("2026-06-17", "Ghana", "Panama", 1, 0),
    ("2026-06-17", "Uzbekistan", "Colombia", 1, 3),

    ("2026-06-18", "Czech Republic", "South Africa", 1, 1),
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", 4, 1),
    ("2026-06-18", "Canada", "Qatar", 6, 0),
    ("2026-06-18", "Mexico", "South Korea", 1, 0),

    ("2026-06-19", "United States", "Australia", 2, 0),
    ("2026-06-19", "Scotland", "Morocco", 0, 1),
    ("2026-06-19", "Brazil", "Haiti", 3, 0),
    ("2026-06-19", "Turkey", "Paraguay", 0, 1),

    ("2026-06-20", "Netherlands", "Sweden", 5, 1),
    ("2026-06-20", "Germany", "Ivory Coast", 2, 1),
    ("2026-06-20", "Ecuador", "Curacao", 0, 0),
    ("2026-06-20", "Tunisia", "Japan", 0, 3),

    ("2026-06-21", "Spain", "Saudi Arabia", 4, 0),
    ("2026-06-21", "Belgium", "Iran", 0, 0),
    ("2026-06-21", "Uruguay", "Cape Verde", 2, 2),
    ("2026-06-21", "New Zealand", "Egypt", 1, 3),
    ("2026-06-22", "Argentina", "Austria", 2, 0),
    ("2026-06-22", "France", "Iraq", 3, 0),
    ("2026-06-22", "Senegal", "Norway", 2, 3),
    ("2026-06-22", "Algeria", "Jordan", 2, 1),
    ("2026-06-21", "Spain", "Saudi Arabia", 4, 0),
    ("2026-06-21", "Belgium", "Iran", 0, 0),
    ("2026-06-21", "Uruguay", "Cape Verde", 2, 2),
    ("2026-06-21", "New Zealand", "Egypt", 1, 3),

    ("2026-06-22", "Argentina", "Austria", 2, 0),
    ("2026-06-22", "France", "Iraq", 3, 0),
    ("2026-06-22", "Senegal", "Norway", 2, 3),
    ("2026-06-22", "Algeria", "Jordan", 2, 1),

    ("2026-06-23", "Portugal", "Uzbekistan", 5, 0),
    ("2026-06-23", "England", "Ghana", 0, 0),
    ("2026-06-23", "Panama", "Croatia", 0, 1),
    ("2026-06-23", "Colombia", "DR Congo", 1, 0),

    ("2026-06-24", "Switzerland", "Canada", 2, 1),
    ("2026-06-24", "Bosnia and Herzegovina", "Qatar", 3, 1),
    ("2026-06-24", "Scotland", "Brazil", 0, 3),
    ("2026-06-24", "Morocco", "Haiti", 4, 2),
    ("2026-06-24", "Czech Republic", "Mexico", 0, 3),
    ("2026-06-24", "South Africa", "South Korea", 1, 0),

    ("2026-06-25", "Ecuador", "Germany", 2, 1),
    ("2026-06-25", "Curacao", "Ivory Coast", 0, 2),
    ("2026-06-25", "Tunisia", "Netherlands", 1, 3),
    ("2026-06-25", "Japan", "Sweden", 1, 1),
    ("2026-06-25", "Turkey", "United States", 3, 2),
    ("2026-06-25", "Paraguay", "Australia", 0, 0),

    ("2026-06-26", "Norway", "France", 1, 4),
    ("2026-06-26", "Senegal", "Iraq", 5, 0),
    ("2026-06-26", "Uruguay", "Spain", 0, 1),
    ("2026-06-26", "Cape Verde", "Saudi Arabia", 0, 0),
    ("2026-06-26", "New Zealand", "Belgium", 1, 5),
    ("2026-06-26", "Egypt", "Iran", 1, 1), 
]


# FUNCIONES AUXILIARES

def normalizar_booleano(valor):
    if isinstance(valor, bool):
        return valor

    texto = str(valor).strip().lower()

    if texto in ["true", "1", "yes", "si", "sí"]:
        return True

    return False


def score_vacio(valor):
    if pd.isna(valor):
        return True

    texto = str(valor).strip()

    if texto == "":
        return True

    if texto.lower() in ["nan", "none", "null"]:
        return True

    return False


# ACTUALIZAR RESULTS.CSV

def main():
    if not ARCHIVO_RESULTS.exists():
        raise FileNotFoundError(f"No existe el archivo: {ARCHIVO_RESULTS}")

    results = pd.read_csv(ARCHIVO_RESULTS)

    if not ARCHIVO_BACKUP.exists():
        results.to_csv(ARCHIVO_BACKUP, index=False, encoding="utf-8")
        print("Backup creado:", ARCHIVO_BACKUP)
    else:
        print("Backup ya existía:", ARCHIVO_BACKUP)

    columnas_base = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral"
    ]

    for col in columnas_base:
        if col not in results.columns:
            results[col] = None

    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["home_team"] = results["home_team"].astype(str).str.strip()
    results["away_team"] = results["away_team"].astype(str).str.strip()
    results["neutral"] = results["neutral"].apply(normalizar_booleano)

    fecha_maxima_antes = results["date"].max()

    filas_nuevas = []
    partidos_actualizados = 0
    partidos_existentes = 0

    for fecha, local, visitante, goles_local, goles_visitante in partidos_mundial_2026:

        existe = results[
            (results["date"] == fecha) &
            (results["home_team"] == local) &
            (results["away_team"] == visitante)
        ]

        if not existe.empty:
            indices = existe.index

            for indice in indices:
                score_local_vacio = score_vacio(results.loc[indice, "home_score"])
                score_visitante_vacio = score_vacio(results.loc[indice, "away_score"])

                if score_local_vacio or score_visitante_vacio:
                    results.loc[indice, "home_score"] = goles_local
                    results.loc[indice, "away_score"] = goles_visitante
                    results.loc[indice, "tournament"] = TORNEO
                    results.loc[indice, "city"] = CIUDAD
                    results.loc[indice, "country"] = PAIS
                    results.loc[indice, "neutral"] = NEUTRAL

                    partidos_actualizados += 1
                    print("Marcador actualizado:", fecha, local, "vs", visitante)
                else:
                    partidos_existentes += 1
                    print("Ya existe con marcador:", fecha, local, "vs", visitante)

        else:
            filas_nuevas.append({
                "date": fecha,
                "home_team": local,
                "away_team": visitante,
                "home_score": goles_local,
                "away_score": goles_visitante,
                "tournament": TORNEO,
                "city": CIUDAD,
                "country": PAIS,
                "neutral": NEUTRAL
            })

    if len(filas_nuevas) > 0:
        nuevos = pd.DataFrame(filas_nuevas)
        results_actualizado = pd.concat([results, nuevos], ignore_index=True)
    else:
        results_actualizado = results.copy()

    results_actualizado["date"] = pd.to_datetime(results_actualizado["date"], errors="coerce")
    results_actualizado = results_actualizado.sort_values("date").reset_index(drop=True)
    results_actualizado["date"] = results_actualizado["date"].dt.strftime("%Y-%m-%d")

    results_actualizado.to_csv(ARCHIVO_RESULTS, index=False, encoding="utf-8")

    print()
    print("========================================")
    print("RESULTS.CSV ACTUALIZADO")
    print("========================================")
    print("Partidos nuevos agregados:", len(filas_nuevas))
    print("Partidos existentes actualizados con marcador:", partidos_actualizados)
    print("Partidos que ya tenían marcador:", partidos_existentes)
    print("Fecha máxima antes:", fecha_maxima_antes)
    print("Nueva fecha máxima:", results_actualizado["date"].max())
    print("Archivo actualizado:", ARCHIVO_RESULTS)
    print("========================================")


if __name__ == "__main__":
    main()
