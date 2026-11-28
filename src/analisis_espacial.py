"""
SIDECAN - Módulo de Análisis Geoespacial y Modelado

Funciones principales:
1. Limpieza y filtrado geográfico de clínicas físicas y rescates en Long Beach.
2. Cálculo vectorizado de distancias geodésicas (Haversine en NumPy).
3. Normalización Min-Max y cálculo del Índice de Prioridad Zonal (0-100).
4. Exportación del ranking general de cuadrantes (~1.1 km x 1.1 km) y códigos postales (ZIP).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# Pesos calibrados para priorizar déficit de esterilización (w2) y distancia a clínicas (w4)
PESOS_POR_DEFECTO = {
    "densidad": 0.20,          # w1: volumen total de casos en el sector
    "no_esterilizados": 0.30,  # w2: déficit reproductivo (% sin esterilizar) [PRIORIZADO]
    "vulnerables": 0.20,       # w3: gravedad clínica (% cachorros, enfermos o heridos)
    "distancia": 0.30,         # w4: aislamiento geográfico (km a clínica más cercana) [PRIORIZADO]
}


def limpiar_coordenadas(
    df: pd.DataFrame,
    lat_col: str = "latitud",
    lon_col: str = "longitud",
    lat_bounds: tuple = (33.70, 33.90),
    lon_bounds: tuple = (-118.25, -118.00),
) -> tuple:
    """Separa filas con coordenadas válidas dentro del área de estudio de las inválidas."""
    df_copy = df.copy()
    df_copy[lat_col] = pd.to_numeric(df_copy[lat_col], errors="coerce")
    df_copy[lon_col] = pd.to_numeric(df_copy[lon_col], errors="coerce")

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    es_valida = (
        df_copy[lat_col].notna()
        & df_copy[lon_col].notna()
        & np.isfinite(df_copy[lat_col])
        & np.isfinite(df_copy[lon_col])
        & (df_copy[lat_col] != 0.0)
        & (df_copy[lon_col] != 0.0)
        & df_copy[lat_col].between(lat_min, lat_max)
        & df_copy[lon_col].between(lon_min, lon_max)
    )
    return df_copy.loc[es_valida].copy(), df_copy.loc[~es_valida].copy()


def cargar_y_preparar_datos(
    ruta_animales: str = "shelter_clean_perros_gatos_espanol.csv",
    ruta_clinicas: str = "veterinarias_long_beach_espanol.csv",
    solo_long_beach: bool = True,
) -> tuple:
    """Carga, audita y filtra los datasets de rescates y clínicas veterinarias físicas."""
    df_animales_raw = pd.read_csv(ruta_animales, low_memory=False)
    df_clinicas_raw = pd.read_csv(ruta_clinicas, low_memory=False)

    # 1. Clínicas físicas activas (excluye servicios móviles y negocios desde casa)
    clinicas_ok, _ = limpiar_coordenadas(df_clinicas_raw)
    filtro_fisicas = (
        clinicas_ok["negocio_desde_casa"].astype(str).str.strip().str.lower() != "yes"
    ) & (
        clinicas_ok["descripcion_licencia"].astype(str).str.strip().str.lower()
        != "mobile services - misc."
    )
    clinicas_limpias = clinicas_ok.loc[filtro_fisicas].copy()

    # Completar nombres comerciales ausentes con nombre_propietario (evita NaN)
    clinicas_limpias["nombre_clinica"] = (
        clinicas_limpias["nombre_comercial"]
        .fillna(clinicas_limpias["nombre_propietario"])
        .astype(str)
        .str.strip()
    )
    clinicas_limpias = clinicas_limpias.drop_duplicates(
        subset=["latitud", "longitud"], keep="first"
    ).reset_index(drop=True)

    # 2. Animales con coordenada válida en vía pública
    animales_ok, _ = limpiar_coordenadas(df_animales_raw)
    if "es_coord_valida_mapa" in animales_ok.columns:
        flag_mapa = animales_ok["es_coord_valida_mapa"].astype(str).str.lower() == "true"
        animales_ok = animales_ok.loc[flag_mapa].copy()

    # 3. Filtrar estrictamente a la jurisdicción de Long Beach para evitar sesgo de frontera
    if solo_long_beach and "jurisdiccion" in animales_ok.columns:
        animales_ok = animales_ok[
            animales_ok["jurisdiccion"].astype(str).str.strip().str.upper() == "LONG BEACH"
        ].copy()

        # Excluir registros mal etiquetados con dirección explícita de la ciudad de Cerritos
        # (conservando las calles llamadas "Cerritos Ave" dentro de Long Beach)
        mask_cerritos_ciudad = (
            animales_ok["cruce_calle"]
            .astype(str)
            .str.contains(r"(?:,\s*CERRITOS|\bCERRITOS\s+CA|\bCERR\s+CA)", case=False, regex=True, na=False)
            & ~animales_ok["cruce_calle"]
            .astype(str)
            .str.contains(r"LONG\s+BEACH", case=False, regex=True, na=False)
        )
        animales_ok = animales_ok.loc[~mask_cerritos_ciudad].copy()

    animales_ok["codigo_postal_extraido"] = (
        animales_ok["cruce_calle"].astype(str).str.extract(r"\b(908\d{2})\b")
    )
    return animales_ok.reset_index(drop=True), clinicas_limpias


def calcular_distancia_minima_haversine(
    lat_animales: np.ndarray,
    lon_animales: np.ndarray,
    lat_clinicas: np.ndarray,
    lon_clinicas: np.ndarray,
    radio_tierra_km: float = 6371.0,
) -> tuple:
    """
    Calcula de forma vectorizada con NumPy la distancia geodésica mínima (km)
    desde cada rescate hasta la clínica veterinaria física más próxima.
    """
    lat1 = np.radians(np.asarray(lat_animales, dtype=np.float64))[:, np.newaxis]
    lon1 = np.radians(np.asarray(lon_animales, dtype=np.float64))[:, np.newaxis]
    lat2 = np.radians(np.asarray(lat_clinicas, dtype=np.float64))[np.newaxis, :]
    lon2 = np.radians(np.asarray(lon_clinicas, dtype=np.float64))[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    a_clip = np.clip(a, 0.0, 1.0)
    distancias_km = 2.0 * radio_tierra_km * np.arcsin(np.sqrt(a_clip))

    idx_min = np.argmin(distancias_km, axis=1)
    dist_min = distancias_km[np.arange(len(lat1)), idx_min]
    return dist_min, idx_min


def min_max_norm(serie: pd.Series) -> pd.Series:
    """Normaliza una serie numérica al rango estándar [0, 1] mediante escalado Min-Max."""
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(0.0, index=serie.index)
    return (serie - serie.min()) / rango


def validar_pesos(pesos: dict) -> tuple:
    """Verifica que los 4 pesos existan y sumen exactamente 1.0."""
    w1 = float(pesos["densidad"])
    w2 = float(pesos["no_esterilizados"])
    w3 = float(pesos["vulnerables"])
    w4 = float(pesos["distancia"])
    suma = w1 + w2 + w3 + w4
    if not np.isclose(suma, 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1.0, pero suman {suma:.4f}")
    return w1, w2, w3, w4


def calcular_ranking_cuadrantes(
    df_animales: pd.DataFrame,
    pesos: dict = None,
    step_lat: float = 0.010,
    step_lon: float = 0.012,
    lat_origen: float = 33.70,
    lon_origen: float = -118.25,
    min_casos: int = 30,
) -> pd.DataFrame:
    """
    Agrupa los rescates en cuadrantes regulares (~1.1 km x 1.1 km) y calcula
    el Índice de Prioridad Zonal en escala de 0 a 100.
    """
    pesos = pesos or PESOS_POR_DEFECTO
    w1, w2, w3, w4 = validar_pesos(pesos)

    df = df_animales.copy()
    df["grid_lat"] = np.floor((df["latitud"] - lat_origen) / step_lat).astype(int)
    df["grid_lon"] = np.floor((df["longitud"] - lon_origen) / step_lon).astype(int)
    df["zona_id"] = (
        "Z_"
        + df["grid_lat"].astype(str).str.zfill(2)
        + "_"
        + df["grid_lon"].astype(str).str.zfill(2)
    )

    zonas = df.groupby(["zona_id", "grid_lat", "grid_lon"], as_index=False).agg(
        total_casos=("id_canil", "count"),
        casos_perros=("tipo_animal", lambda s: int((s == "DOG").sum())),
        casos_gatos=("tipo_animal", lambda s: int((s == "CAT").sum())),
        tasa_esterilizados=("esta_esterilizado", "mean"),
        pct_vulnerables=("es_vulnerable", lambda s: float(s.mean() * 100.0)),
        distancia_media_km=("distancia_clinica_km", "mean"),
        distancia_mediana_km=("distancia_clinica_km", "median"),
        lat_centro=("latitud", "mean"),
        lon_centro=("longitud", "mean"),
        cruce_referencia=("cruce_calle", lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "N/A"),
        zip_principal=(
            "codigo_postal_extraido",
            lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) > 0 else "Sin ZIP",
        ),
        clinica_mas_cercana_frecuente=(
            "clinica_mas_cercana",
            lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "N/A",
        ),
    )

    zonas["pct_no_esterilizados"] = (1.0 - zonas["tasa_esterilizados"]) * 100.0

    # Límites exactos de cada celda (útiles para dibujar cuadrantes en mapas interactivos)
    zonas["lat_min_celda"] = lat_origen + zonas["grid_lat"] * step_lat
    zonas["lat_max_celda"] = zonas["lat_min_celda"] + step_lat
    zonas["lon_min_celda"] = lon_origen + zonas["grid_lon"] * step_lon
    zonas["lon_max_celda"] = zonas["lon_min_celda"] + step_lon

    # Filtrar cuadrantes con volumen mínimo para estabilidad estadística de porcentajes
    zonas = zonas[zonas["total_casos"] >= min_casos].copy()

    # Normalización Min-Max [0, 1]
    zonas["norm_densidad"] = min_max_norm(zonas["total_casos"])
    zonas["norm_no_esterilizados"] = min_max_norm(zonas["pct_no_esterilizados"])
    zonas["norm_vulnerables"] = min_max_norm(zonas["pct_vulnerables"])
    zonas["norm_distancia"] = min_max_norm(zonas["distancia_media_km"])

    # Índice de Prioridad Zonal (0 a 100)
    zonas["indice_prioridad"] = (
        w1 * zonas["norm_densidad"]
        + w2 * zonas["norm_no_esterilizados"]
        + w3 * zonas["norm_vulnerables"]
        + w4 * zonas["norm_distancia"]
    ) * 100.0

    zonas["nivel_prioridad"] = pd.cut(
        zonas["indice_prioridad"],
        bins=[-0.1, 35.0, 50.0, 100.1],
        labels=["Baja (Verde)", "Media (Amarilla)", "Alta (Roja)"],
    )

    zonas = zonas.sort_values("indice_prioridad", ascending=False).reset_index(drop=True)
    zonas.insert(0, "ranking", np.arange(1, len(zonas) + 1))

    cols_orden = [
        "ranking", "zona_id", "zip_principal", "nivel_prioridad", "indice_prioridad",
        "total_casos", "casos_perros", "casos_gatos",
        "pct_no_esterilizados", "pct_vulnerables",
        "distancia_media_km", "distancia_mediana_km",
        "norm_densidad", "norm_no_esterilizados", "norm_vulnerables", "norm_distancia",
        "cruce_referencia", "clinica_mas_cercana_frecuente",
        "lat_centro", "lon_centro",
        "lat_min_celda", "lat_max_celda", "lon_min_celda", "lon_max_celda",
    ]
    return zonas[cols_orden].round(4)


def calcular_ranking_zip(
    df_animales: pd.DataFrame,
    df_clinicas: pd.DataFrame,
    pesos: dict = None,
    min_casos: int = 100,
) -> pd.DataFrame:
    """Genera un ranking macro por Código Postal (ZIP) de Long Beach con conteo de clínicas."""
    pesos = pesos or PESOS_POR_DEFECTO
    w1, w2, w3, w4 = validar_pesos(pesos)

    clin_por_zip = df_clinicas["codigo_postal"].astype(str).value_counts().to_dict()

    zips = (
        df_animales.dropna(subset=["codigo_postal_extraido"])
        .groupby("codigo_postal_extraido", as_index=False)
        .agg(
            total_casos=("id_canil", "count"),
            casos_perros=("tipo_animal", lambda s: int((s == "DOG").sum())),
            casos_gatos=("tipo_animal", lambda s: int((s == "CAT").sum())),
            pct_no_esterilizados=("esta_esterilizado", lambda s: float((1.0 - s.mean()) * 100.0)),
            pct_vulnerables=("es_vulnerable", lambda s: float(s.mean() * 100.0)),
            distancia_media_km=("distancia_clinica_km", "mean"),
            distancia_mediana_km=("distancia_clinica_km", "median"),
            lat_centro=("latitud", "mean"),
            lon_centro=("longitud", "mean"),
        )
    )
    zips = zips.rename(columns={"codigo_postal_extraido": "codigo_postal"})
    zips = zips[zips["total_casos"] >= min_casos].copy()
    zips["clinicas_fisicas"] = zips["codigo_postal"].map(clin_por_zip).fillna(0).astype(int)

    zips["indice_prioridad"] = (
        w1 * min_max_norm(zips["total_casos"])
        + w2 * min_max_norm(zips["pct_no_esterilizados"])
        + w3 * min_max_norm(zips["pct_vulnerables"])
        + w4 * min_max_norm(zips["distancia_media_km"])
    ) * 100.0

    zips["nivel_prioridad"] = pd.cut(
        zips["indice_prioridad"],
        bins=[-0.1, 35.0, 55.0, 100.1],
        labels=["Baja (Verde)", "Media (Amarilla)", "Alta (Roja)"],
    )

    zips = zips.sort_values("indice_prioridad", ascending=False).reset_index(drop=True)
    zips.insert(0, "ranking", np.arange(1, len(zips) + 1))

    cols = [
        "ranking", "codigo_postal", "nivel_prioridad", "indice_prioridad",
        "clinicas_fisicas", "total_casos", "casos_perros", "casos_gatos",
        "pct_no_esterilizados", "pct_vulnerables",
        "distancia_media_km", "distancia_mediana_km",
        "lat_centro", "lon_centro",
    ]
    return zips[cols].round(4)


def ejecutar_pipeline_espacial(
    ruta_animales: str = "shelter_clean_perros_gatos_espanol.csv",
    ruta_clinicas: str = "veterinarias_long_beach_espanol.csv",
    carpeta_salida: str = ".",
    pesos: dict = None,
) -> dict:
    """Ejecuta el flujo completo de análisis geoespacial y guarda los archivos CSV resultantes."""
    pesos = pesos or PESOS_POR_DEFECTO
    animales_lb, clinicas_lb = cargar_y_preparar_datos(
        ruta_animales=ruta_animales,
        ruta_clinicas=ruta_clinicas,
        solo_long_beach=True,
    )

    dist_min, idx_min = calcular_distancia_minima_haversine(
        animales_lb["latitud"].to_numpy(),
        animales_lb["longitud"].to_numpy(),
        clinicas_lb["latitud"].to_numpy(),
        clinicas_lb["longitud"].to_numpy(),
    )
    animales_lb["distancia_clinica_km"] = dist_min
    animales_lb["clinica_mas_cercana"] = clinicas_lb["nombre_clinica"].iloc[idx_min].to_numpy()

    ranking_cuadrantes = calcular_ranking_cuadrantes(animales_lb, pesos=pesos, min_casos=30)
    ranking_zip = calcular_ranking_zip(animales_lb, clinicas_lb, pesos=pesos, min_casos=100)

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_cuadrantes = os.path.join(carpeta_salida, "ranking_zonas_prioridad_long_beach.csv")
    ruta_zip = os.path.join(carpeta_salida, "ranking_zip_long_beach.csv")
    ruta_animales_dist = os.path.join(carpeta_salida, "animales_long_beach_con_distancia.csv")

    ranking_cuadrantes.to_csv(ruta_cuadrantes, index=False)
    ranking_zip.to_csv(ruta_zip, index=False)
    animales_lb.to_csv(ruta_animales_dist, index=False)

    return {
        "animales": animales_lb,
        "clinicas": clinicas_lb,
        "ranking_cuadrantes": ranking_cuadrantes,
        "ranking_zip": ranking_zip,
    }


if __name__ == "__main__":
    resultados = ejecutar_pipeline_espacial()
    animales = resultados["animales"]
    clinicas = resultados["clinicas"]
    rc = resultados["ranking_cuadrantes"]
    rz = resultados["ranking_zip"]

    print("=== RESUMEN PIPELINE GEOESPACIAL SIDECAN (LONG BEACH) ===")
    print(f"Clínicas físicas activas en Long Beach: {len(clinicas)}")
    print(f"Rescates analizados en Long Beach: {len(animales):,}")
    print(
        f"Distancia a clínica más cercana -> Media: {animales['distancia_clinica_km'].mean():.2f} km | "
        f"Mediana: {animales['distancia_clinica_km'].median():.2f} km | "
        f"Máx: {animales['distancia_clinica_km'].max():.2f} km"
    )
    print(f"Cuadrantes evaluados (>=30 casos): {len(rc)} (cubren {rc['total_casos'].sum():,} casos)")
    print("\n--- TOP 10 CUADRANTES DE MÁXIMA PRIORIDAD ---")
    cols_print = [
        "ranking", "zona_id", "zip_principal", "total_casos",
        "pct_no_esterilizados", "pct_vulnerables", "distancia_media_km", "indice_prioridad"
    ]
    print(rc[cols_print].head(10).round(2).to_string(index=False))

    print("\n--- RANKING GENERAL POR CÓDIGO POSTAL (ZIP) ---")
    print(
        rz[[
            "ranking", "codigo_postal", "clinicas_fisicas", "total_casos",
            "pct_no_esterilizados", "distancia_media_km", "indice_prioridad", "nivel_prioridad"
        ]].round(2).to_string(index=False)
    )