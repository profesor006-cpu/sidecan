"""
SIDECAN — Orquestador Principal del Pipeline Analítico

Ejecuta en orden:
1. Limpieza y auditoría inicial desde data/raw hacia data/processed.
2. Pipeline geoespacial (Haversine en NumPy + Índice de Prioridad Zonal 0-100 + Ranking).
3. Pipeline estadístico y trazabilidad sanitaria (Demografía, estacionalidad, FIELD vs OTC y Chi2).
"""

from pathlib import Path

from src.data_processing import ejecutar_pipeline_limpieza
from src.estadisticas import ejecutar_pipeline_estadistico
from src.analisis_espacial import ejecutar_pipeline_espacial

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"


def main() -> None:
    """Ejecuta el flujo integrado de SIDECAN de principio a fin."""
    ruta_animales = str(DATA_PROCESSED / "shelter_clean_perros_gatos_espanol.csv")
    ruta_clinicas = str(DATA_PROCESSED / "veterinarias_long_beach_espanol.csv")
    ruta_animales_dist = str(DATA_PROCESSED / "animales_long_beach_con_distancia.csv")
    carpeta_salida = str(DATA_PROCESSED)

    print("=== SIDECAN: INICIANDO PIPELINE INTEGRADO ===")

    res_limpieza = ejecutar_pipeline_limpieza(str(DATA_RAW), carpeta_salida)
    print(
        f"[1/3] Limpieza OK -> {len(res_limpieza['refugio']):,} rescates limpios | "
        f"{len(res_limpieza['veterinarias_lb']):,} licencias LB | "
        f"{len(res_limpieza['veterinarias_usa']):,} clínicas área metropolitana."
    )

    res_espacial = ejecutar_pipeline_espacial(
        ruta_animales=ruta_animales,
        ruta_clinicas=ruta_clinicas,
        carpeta_salida=carpeta_salida,
    )
    animales = res_espacial["animales"]
    clinicas = res_espacial["clinicas"]
    cuadrantes = res_espacial["ranking_cuadrantes"]
    zips = res_espacial["ranking_zip"]

    print(
        f"[2/3] Espacial OK -> {len(clinicas)} clínicas físicas | "
        f"{len(animales):,} rescates válidos | "
        f"Mediana distancia: {animales['distancia_clinica_km'].median():.2f} km | "
        f"{len(cuadrantes)} cuadrantes y {len(zips)} ZIPs rankeados."
    )

    res_eda = ejecutar_pipeline_estadistico(
        ruta_animales=ruta_animales,
        ruta_animales_distancia=ruta_animales_dist,
        carpeta_salida=carpeta_salida,
    )
    demo = res_eda["demografia"]
    total_ing = int(demo.loc[demo["especie"] == "TOTAL", "total_ingresos"].iloc[0])
    print(
        f"[3/3] Estadística EDA OK -> {total_ing:,} registros demográficos analizados | "
        f"Resultados guardados en {DATA_PROCESSED}."
    )
    print("=== PIPELINE SIDECAN COMPLETADO EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
