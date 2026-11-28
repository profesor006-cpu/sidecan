"""
SIDECAN - Módulo de Análisis Exploratorio y Trazabilidad Sanitaria

Funciones principales:
1. Demografía de perros y gatos en situación de calle (edad, grupos etarios, esterilización y reincidencia).
2. Estacionalidad mensual y trimestral de ingresos y camadas vulnerables (años completos 2017-2025).
3. Trazabilidad sanitaria: condición clínica al ingreso frente a desenlaces, estadía y supervivencia.
4. Comparativa operativa entre rescates en campo (FIELD) y entregas en mostrador (OTC).
5. Cruce entre distancia a la clínica veterinaria más cercana, gravedad clínica y supervivencia.
"""

from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd


MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}


def chi2_2x2(tabla: pd.DataFrame) -> tuple:
    """
    Calcula el estadístico Chi-cuadrado de Pearson (con corrección de Yates)
    y el p-valor exacto para una tabla de contingencia 2x2 usando solo NumPy y math.
    """
    obs = tabla.to_numpy(dtype=np.float64)
    if obs.shape != (2, 2):
        raise ValueError("Se requiere una tabla de contingencia 2x2.")
    total = obs.sum()
    if total == 0:
        return 0.0, 1.0
    filas = obs.sum(axis=1, keepdims=True)
    cols = obs.sum(axis=0, keepdims=True)
    esperado = (filas @ cols) / total
    diff = np.maximum(np.abs(obs - esperado) - 0.5, 0.0)
    chi2 = float(np.sum((diff ** 2) / np.maximum(esperado, 1e-12)))
    p_valor = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p_valor


def agrupar_condicion_clinica(condicion: str) -> str:
    """Clasifica las categorías detalladas de ingreso en grupos clínicos interpretables."""
    c = str(condicion).strip().upper()
    if c == "NORMAL":
        return "Sano / Normal"
    if c == "UNDER AGE/WEIGHT":
        return "Cría / Bajo peso"
    if c in ("ILL MILD", "ILL MODERATE"):
        return "Enfermo (Leve/Mod.)"
    if c == "ILL SEVERE":
        return "Enfermo (Grave)"
    if c in ("INJURED MILD", "INJURED MODERATE"):
        return "Herido (Leve/Mod.)"
    if c == "INJURED SEVERE":
        return "Herido (Grave)"
    if c in ("FERAL", "FRACTIOUS", "BEHAVIOR MILD", "BEHAVIOR MODERATE", "BEHAVIOR SEVERE"):
        return "Feral / Conducta"
    if c == "AGED":
        return "Edad avanzada"
    return "Otra / Reporte I/I"


def cargar_base_demografica(
    ruta_animales: str = "shelter_clean_perros_gatos_espanol.csv",
    solo_long_beach: bool = True,
) -> pd.DataFrame:
    """
    Carga el historial limpio de perros y gatos en situación de calle y prepara
    las variables analíticas requeridas para el análisis estadístico y sanitario.
    """
    df = pd.read_csv(ruta_animales, low_memory=False)

    if solo_long_beach and "jurisdiccion" in df.columns:
        df = df[df["jurisdiccion"].astype(str).str.strip().str.upper() == "LONG BEACH"].copy()
        mask_cerritos = (
            df["cruce_calle"]
            .astype(str)
            .str.contains(r"(?:,\s*CERRITOS|\bCERRITOS\s+CA|\bCERR\s+CA)", case=False, regex=True, na=False)
            & ~df["cruce_calle"]
            .astype(str)
            .str.contains(r"LONG\s+BEACH", case=False, regex=True, na=False)
        )
        df = df.loc[~mask_cerritos].copy()

    for col in ("fecha_nacimiento", "fecha_ingreso", "fecha_salida"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Recalcular edad en meses y estado reproductivo binario si faltasen
    if "edad_meses" not in df.columns:
        df["edad_meses"] = ((df["fecha_ingreso"] - df["fecha_nacimiento"]).dt.days / 30.4375).round(1)

    if "esta_esterilizado" not in df.columns and "sexo" in df.columns:
        df["esta_esterilizado"] = df["sexo"].map({
            "Spayed": 1.0, "Neutered": 1.0, "Female": 0.0, "Male": 0.0, "Unknown": np.nan
        })

    df["anio_ingreso"] = df["fecha_ingreso"].dt.year
    df["mes_ingreso"] = df["fecha_ingreso"].dt.month
    df["trimestre"] = "T" + df["fecha_ingreso"].dt.quarter.astype(str)

    df["grupo_etario"] = pd.cut(
        df["edad_meses"],
        bins=[-0.1, 6.0, 12.0, 84.0, 400.0],
        labels=["Cachorro/Cría (<6m)", "Joven (6-12m)", "Adulto (1-7a)", "Senior (>7a)"],
    )
    df["grupo_condicion"] = df["condicion_ingreso"].apply(agrupar_condicion_clinica)
    df["es_grave"] = df["condicion_ingreso"].isin(["ILL SEVERE", "INJURED SEVERE"]).astype(int)
    df["es_herido_o_enfermo"] = df["condicion_ingreso"].astype(str).str.startswith(("ILL", "INJURED")).astype(int)
    df["es_under_age"] = (df["condicion_ingreso"] == "UNDER AGE/WEIGHT").astype(int)

    return df.reset_index(drop=True)


def resumen_demografico(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores demográficos, reproductivos, de reincidencia y supervivencia por especie."""
    filas = []
    for etiqueta, sub in [("TOTAL", df), ("DOG", df[df["tipo_animal"] == "DOG"]), ("CAT", df[df["tipo_animal"] == "CAT"])]:
        conteo_animal = sub.groupby("id_animal").size()
        n_unicos = int(len(conteo_animal))
        n_reinc = int((conteo_animal >= 2).sum())

        filas.append({
            "especie": etiqueta,
            "total_ingresos": int(len(sub)),
            "animales_unicos": n_unicos,
            "reincidentes": n_reinc,
            "pct_reincidentes": round((n_reinc / max(n_unicos, 1)) * 100.0, 2),
            "edad_mediana_meses": round(float(sub["edad_meses"].median()), 2),
            "edad_media_meses": round(float(sub["edad_meses"].mean()), 2),
            "pct_menores_6m": round(float((sub["edad_meses"].dropna() < 6.0).mean() * 100.0), 2),
            "pct_sexo_desconocido": round(float(sub["esta_esterilizado"].isna().mean() * 100.0), 2),
            "pct_esterilizados": round(float(sub["esta_esterilizado"].mean() * 100.0), 2),
            "pct_no_esterilizados": round(float((1.0 - sub["esta_esterilizado"].mean()) * 100.0), 2),
            "pct_vulnerables": round(float(sub["es_vulnerable"].mean() * 100.0), 2),
            "pct_under_age": round(float(sub["es_under_age"].mean() * 100.0), 2),
            "pct_graves": round(float(sub["es_grave"].mean() * 100.0), 2),
            "estancia_mediana_dias": round(float(sub["duracion_estancia"].median()), 2),
            "estancia_media_dias": round(float(sub["duracion_estancia"].mean()), 2),
            "tasa_supervivencia_pct": round(float(sub["sobrevivio"].mean() * 100.0), 2),
        })
    return pd.DataFrame(filas)


def analizar_estacionalidad(df: pd.DataFrame, anio_inicio: int = 2017, anio_fin: int = 2025) -> tuple:
    """
    Analiza la estacionalidad mensual y trimestral sobre años calendario completos
    para evitar distorsiones por meses incompletos del último año registrado.
    """
    sub = df[df["anio_ingreso"].between(anio_inicio, anio_fin)].copy()
    n_anios = float(sub["anio_ingreso"].nunique())

    mensual = sub.groupby("mes_ingreso", as_index=False).agg(
        total_acumulado=("id_canil", "count"),
        perros_acumulado=("tipo_animal", lambda s: int((s == "DOG").sum())),
        gatos_acumulado=("tipo_animal", lambda s: int((s == "CAT").sum())),
        under_age_acumulado=("es_under_age", "sum"),
        tasa_supervivencia_pct=("sobrevivio", lambda s: float(s.mean() * 100.0)),
    )
    mensual["mes_nombre"] = mensual["mes_ingreso"].map(MESES_ES)
    mensual["promedio_anual_total"] = (mensual["total_acumulado"] / n_anios).round(1)
    mensual["promedio_anual_perros"] = (mensual["perros_acumulado"] / n_anios).round(1)
    mensual["promedio_anual_gatos"] = (mensual["gatos_acumulado"] / n_anios).round(1)
    mensual["promedio_anual_under_age"] = (mensual["under_age_acumulado"] / n_anios).round(1)
    mensual["pct_under_age"] = ((mensual["under_age_acumulado"] / mensual["total_acumulado"]) * 100.0).round(2)
    mensual["tasa_supervivencia_pct"] = mensual["tasa_supervivencia_pct"].round(2)

    trimestral = sub.groupby("trimestre", as_index=False).agg(
        total_casos=("id_canil", "count"),
        casos_perros=("tipo_animal", lambda s: int((s == "DOG").sum())),
        casos_gatos=("tipo_animal", lambda s: int((s == "CAT").sum())),
        casos_under_age=("es_under_age", "sum"),
        pct_under_age=("es_under_age", lambda s: float(s.mean() * 100.0)),
        tasa_supervivencia_pct=("sobrevivio", lambda s: float(s.mean() * 100.0)),
    )
    trimestral["pct_volumen_anual"] = ((trimestral["total_casos"] / trimestral["total_casos"].sum()) * 100.0).round(2)
    return mensual, trimestral.round(2)


def analizar_trazabilidad_clinica(df: pd.DataFrame) -> pd.DataFrame:
    """Cruza la condición clínica al ingreso con estancia, desenlaces y tasa de supervivencia."""
    tabla = df.groupby("grupo_condicion", as_index=False).agg(
        total_casos=("id_canil", "count"),
        pct_gatos=("tipo_animal", lambda s: float((s == "CAT").mean() * 100.0)),
        pct_no_esterilizados=("esta_esterilizado", lambda s: float((1.0 - s.mean()) * 100.0)),
        estancia_mediana_dias=("duracion_estancia", "median"),
        estancia_media_dias=("duracion_estancia", "mean"),
        tasa_supervivencia_pct=("sobrevivio", lambda s: float(s.mean() * 100.0)),
        pct_adopcion=("tipo_salida", lambda s: float((s == "ADOPTION").mean() * 100.0)),
        pct_devuelto_dueno=("tipo_salida", lambda s: float((s == "RETURN TO OWNER").mean() * 100.0)),
        pct_rescate_ong=("tipo_salida", lambda s: float((s == "RESCUE").mean() * 100.0)),
        pct_eutanasia=("tipo_salida", lambda s: float((s == "EUTHANASIA").mean() * 100.0)),
        pct_fallecido=("tipo_salida", lambda s: float((s == "DIED").mean() * 100.0)),
    )
    tabla["pct_del_total"] = (tabla["total_casos"] / len(df)) * 100.0
    tabla = tabla.sort_values("total_casos", ascending=False).reset_index(drop=True)
    return tabla.round(2)


def comparar_field_vs_otc(df: pd.DataFrame) -> pd.DataFrame:
    """Compara estadísticamente los rescates en vía pública (FIELD) frente a entregas en mostrador (OTC)."""
    fo = df[df["subtipo_ingreso"].isin(["FIELD", "OTC"])].copy()
    filas = []

    for especie, sub in [("TOTAL", fo), ("DOG", fo[fo["tipo_animal"] == "DOG"]), ("CAT", fo[fo["tipo_animal"] == "CAT"])]:
        ct_grave = pd.crosstab(sub["subtipo_ingreso"], sub["es_grave"])
        chi2_g, p_g = chi2_2x2(ct_grave)
        sub_s = sub.dropna(subset=["sobrevivio"])
        ct_surv = pd.crosstab(sub_s["subtipo_ingreso"], sub_s["sobrevivio"])
        chi2_s, p_s = chi2_2x2(ct_surv)

        for canal in ("FIELD", "OTC"):
            c = sub[sub["subtipo_ingreso"] == canal]
            filas.append({
                "especie": especie,
                "canal_ingreso": canal,
                "total_casos": int(len(c)),
                "edad_mediana_meses": round(float(c["edad_meses"].median()), 2),
                "pct_no_esterilizados": round(float((1.0 - c["esta_esterilizado"].mean()) * 100.0), 2),
                "pct_vulnerables": round(float(c["es_vulnerable"].mean() * 100.0), 2),
                "pct_herido_o_enfermo": round(float(c["es_herido_o_enfermo"].mean() * 100.0), 2),
                "pct_graves": round(float(c["es_grave"].mean() * 100.0), 2),
                "pct_under_age": round(float(c["es_under_age"].mean() * 100.0), 2),
                "estancia_mediana_dias": round(float(c["duracion_estancia"].median()), 2),
                "tasa_supervivencia_pct": round(float(c["sobrevivio"].mean() * 100.0), 2),
                "pct_devuelto_dueno": round(float((c["tipo_salida"] == "RETURN TO OWNER").mean() * 100.0), 2),
                "chi2_gravedad_field_vs_otc": round(chi2_g, 2),
                "p_valor_gravedad": p_g,
                "chi2_supervivencia_field_vs_otc": round(chi2_s, 2),
                "p_valor_supervivencia": p_s,
            })

    return pd.DataFrame(filas)


def analizar_distancia_vs_sanidad(
    ruta_animales_distancia: str = "animales_long_beach_con_distancia.csv",
) -> tuple:
    """
    Cruza la distancia geodésica a la clínica veterinaria más cercana con
    esterilización, gravedad clínica y supervivencia.
    """
    df = pd.read_csv(ruta_animales_distancia, low_memory=False)
    df["es_grave"] = df["condicion_ingreso"].isin(["ILL SEVERE", "INJURED SEVERE"]).astype(int)
    df["rango_distancia"] = pd.cut(
        df["distancia_clinica_km"],
        bins=[0.0, 0.75, 1.5, 2.5, 15.0],
        labels=["< 0.75 km", "0.75 - 1.5 km", "1.5 - 2.5 km", "> 2.5 km"],
    )

    tabla = df.groupby("rango_distancia", observed=False, as_index=False).agg(
        total_casos=("id_canil", "count"),
        pct_gatos=("tipo_animal", lambda s: float((s == "CAT").mean() * 100.0)),
        pct_no_esterilizados=("esta_esterilizado", lambda s: float((1.0 - s.mean()) * 100.0)),
        pct_vulnerables=("es_vulnerable", lambda s: float(s.mean() * 100.0)),
        pct_graves=("es_grave", lambda s: float(s.mean() * 100.0)),
        estancia_mediana_dias=("duracion_estancia", "median"),
        tasa_supervivencia_pct=("sobrevivio", lambda s: float(s.mean() * 100.0)),
    ).round(2)

    # Pruebas estadísticas comparando zonas cercanas (<0.75 km) frente a zonas alejadas (>2.5 km)
    extremos = df[(df["distancia_clinica_km"] < 0.75) | (df["distancia_clinica_km"] > 2.5)].copy()
    extremos["grupo_dist"] = np.where(extremos["distancia_clinica_km"] > 2.5, "> 2.5 km", "< 0.75 km")

    pruebas = []
    for col, nombre in [
        ("sobrevivio", "Supervivencia"),
        ("es_grave", "Condición grave al ingreso"),
        ("es_vulnerable", "Vulnerabilidad clínica"),
        ("esta_esterilizado", "Tasa de esterilización"),
    ]:
        sub = extremos.dropna(subset=[col])
        chi2_val, p_val = chi2_2x2(pd.crosstab(sub["grupo_dist"], sub[col]))
        tasas = sub.groupby("grupo_dist")[col].mean() * 100.0
        pruebas.append({
            "indicador": nombre,
            "cerca_lt_0_75km_pct": round(float(tasas["< 0.75 km"]), 2),
            "lejos_gt_2_5km_pct": round(float(tasas["> 2.5 km"]), 2),
            "diferencia_pp": round(float(tasas["> 2.5 km"] - tasas["< 0.75 km"]), 2),
            "chi2": round(chi2_val, 2),
            "p_valor": p_val,
        })

    return tabla, pd.DataFrame(pruebas)


def ejecutar_pipeline_estadistico(
    ruta_animales: str = "shelter_clean_perros_gatos_espanol.csv",
    ruta_animales_distancia: str = "animales_long_beach_con_distancia.csv",
    carpeta_salida: str = ".",
) -> dict:
    """Ejecuta el análisis exploratorio completo y exporta todas las tablas de resultados."""
    df_lb = cargar_base_demografica(ruta_animales=ruta_animales, solo_long_beach=True)

    tbl_demo = resumen_demografico(df_lb)
    tbl_mensual, tbl_trimestral = analizar_estacionalidad(df_lb, anio_inicio=2017, anio_fin=2025)
    tbl_clinica = analizar_trazabilidad_clinica(df_lb)
    tbl_field_otc = comparar_field_vs_otc(df_lb)

    os.makedirs(carpeta_salida, exist_ok=True)
    tbl_demo.to_csv(os.path.join(carpeta_salida, "eda_resumen_demografico.csv"), index=False)
    tbl_mensual.to_csv(os.path.join(carpeta_salida, "eda_estacionalidad_mensual.csv"), index=False)
    tbl_clinica.to_csv(os.path.join(carpeta_salida, "eda_trazabilidad_condicion.csv"), index=False)
    tbl_field_otc.to_csv(os.path.join(carpeta_salida, "eda_comparativa_field_otc.csv"), index=False)

    resultados = {
        "datos_lb": df_lb,
        "demografia": tbl_demo,
        "estacionalidad_mensual": tbl_mensual,
        "estacionalidad_trimestral": tbl_trimestral,
        "trazabilidad_clinica": tbl_clinica,
        "field_vs_otc": tbl_field_otc,
    }

    if os.path.exists(ruta_animales_distancia):
        tbl_dist, tbl_pruebas_dist = analizar_distancia_vs_sanidad(ruta_animales_distancia)
        tbl_dist.to_csv(os.path.join(carpeta_salida, "eda_distancia_vs_sanidad.csv"), index=False)
        resultados["distancia_vs_sanidad"] = tbl_dist
        resultados["pruebas_distancia"] = tbl_pruebas_dist

    return resultados


if __name__ == "__main__":
    res = ejecutar_pipeline_estadistico()

    print("=== SIDECAN: RESUMEN ESTADÍSTICO Y TRAZABILIDAD SANITARIA (LONG BEACH) ===")
    print("\n--- 1. DEMOGRAFÍA Y ESTADO REPRODUCTIVO ---")
    print(
        res["demografia"][[
            "especie", "total_ingresos", "edad_mediana_meses",
            "pct_no_esterilizados", "pct_vulnerables", "estancia_mediana_dias", "tasa_supervivencia_pct"
        ]].to_string(index=False)
    )

    print("\n--- 2. ESTACIONALIDAD TRIMESTRAL (2017-2025) ---")
    print(res["estacionalidad_trimestral"].to_string(index=False))

    print("\n--- 3. TRAZABILIDAD SANITARIA POR CONDICIÓN DE INGRESO ---")
    print(
        res["trazabilidad_clinica"][[
            "grupo_condicion", "total_casos", "estancia_mediana_dias",
            "tasa_supervivencia_pct", "pct_adopcion", "pct_eutanasia"
        ]].to_string(index=False)
    )

    print("\n--- 4. COMPARATIVA OPERATIVA: FIELD VS OTC ---")
    print(
        res["field_vs_otc"][[
            "especie", "canal_ingreso", "total_casos",
            "pct_herido_o_enfermo", "pct_graves", "estancia_mediana_dias", "tasa_supervivencia_pct"
        ]].to_string(index=False)
    )

    if "distancia_vs_sanidad" in res:
        print("\n--- 5. COSTO SANITARIO DE LA DISTANCIA A CLÍNICAS VETERINARIAS ---")
        print(res["distancia_vs_sanidad"].to_string(index=False))
        print("\nPruebas Chi-cuadrado (< 0.75 km vs > 2.5 km):")
        print(res["pruebas_distancia"].to_string(index=False))
