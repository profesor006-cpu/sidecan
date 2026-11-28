"""
SIDECAN - Módulo de Limpieza y Preparación de Datos
"""

import os
import numpy as np
import pandas as pd


BBOX_LAT = (33.70, 33.90)
BBOX_LON = (-118.25, -118.00)

# estructura temporal
RENOMBRAR_REFUGIO_ES = {
    "Kennel ID": "id_canil",
    "Animal ID": "id_animal",
    "Animal Type": "tipo_animal",
    "Sex": "sexo",
    "DOB": "fecha_nacimiento",
    "Intake Date": "fecha_ingreso",
    "Outcome Date": "fecha_salida",
    "Intake Condition": "condicion_ingreso",
    "Intake Subtype": "subtipo_ingreso",
    "Outcome Type": "tipo_salida",
    "intake_duration": "duracion_estancia",
    "was_outcome_alive": "salio_con_vida",
    "Jurisdiction": "jurisdiccion",
    "Crossing": "cruce_calle",
    "latitude": "latitud",
    "longitude": "longitud",
}

# Estructura temporal
RENOMBRAR_VET_LB_ES = {
    "objectid": "id_objeto",
    "license #": "numero_licencia",
    "dba": "nombre_comercial",
    "owner name": "nombre_propietario",
    "license description": "descripcion_licencia",
    "site location": "direccion_sitio",
    "zip code": "codigo_postal",
    "latitude": "latitud",
    "longitude": "longitud",
    "x": "coordenada_x",
    "y": "coordenada_y",
    "license status": "estado_licencia",
    "start date": "fecha_inicio",
    "inactive date": "fecha_inactividad",
    "milestone date": "fecha_hito",
    "milestone": "hito",
    "company type": "tipo_empresa",
    "# employees": "num_empleados",
    "# units": "num_unidades",
    "business activity": "actividad_comercial",
    "home-based business": "negocio_desde_casa",
    "independent contractor": "contratista_independiente",
    "class description": "descripcion_clase",
    "exempt (nonprofit)": "exento_sin_fines_lucro",
    "outside city": "fuera_de_ciudad",
    "bid merchant": "distrito_comercial_comerciantes",
    "bid property": "distrito_comercial_propiedad",
    "bid tourism": "distrito_comercial_turismo",
    "council district": "distrito_municipal",
    "census tract": "sector_censal",
    "milestone simple": "hito_resumido",
    "former rda": "antigua_rda",
}


def limpiar_dataset_refugio(ruta_csv: str, en_espanol: bool = True) -> pd.DataFrame:
    """Filtra y limpia el dataset del refugio de Long Beach para DOG y CAT en situación STRAY."""
    df_raw = pd.read_csv(ruta_csv, low_memory=False)

    #1. Filtros base: perros y gatos en situación de calle
    df = df_raw[
        (df_raw["Animal Type"].isin(["DOG", "CAT"])) &
        (df_raw["Intake Type"] == "STRAY")
    ].copy()

    #2. Eliminar duplicados administrativos y de estancia (NO por Animal ID para conservar reingresos)
    df = df[df["Outcome Type"] != "DUPLICATE"].copy()
    df = df.drop_duplicates(subset=["Kennel ID", "Intake Date"]).copy()

    #3. Parseo estricto y consistencia temporal
    for col in ["DOB", "Intake Date", "Outcome Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    #conservar serie histórica sistemática desde 2017 (descarta 7 registros aislados de 2013-2016)
    df = df[df["Intake Date"] >= "2017-01-01"].copy()
    df = df[df["Outcome Date"].isna() | (df["Outcome Date"] >= df["Intake Date"])].copy()

    #anular DOB imposibles (ej. año 2031) sin borrar el registro del rescate
    df.loc[df["DOB"] > df["Intake Date"], "DOB"] = pd.NaT

    #variables temporales útiles para Rol 3
    df["edad_meses"] = ((df["Intake Date"] - df["DOB"]).dt.days / 30.4375).round(1)
    df["anio_ingreso"] = df["Intake Date"].dt.year
    df["mes_ingreso"] = df["Intake Date"].dt.month

    #4. Duración de estadía válida
    df = df[df["intake_duration"].isna() | (df["intake_duration"] >= 0)].copy()

    #5. Separar Sex -> sexo_biologico y esta_esterilizado (NaN para Unknown)
    df["sexo_biologico"] = df["Sex"].map({
        "Spayed": "Female",
        "Neutered": "Male",
        "Female": "Female",
        "Male": "Male",
        "Unknown": "Unknown",
    }).fillna("Unknown")

    df["esta_esterilizado"] = df["Sex"].map({
        "Spayed": 1.0,
        "Neutered": 1.0,
        "Female": 0.0,
        "Male": 0.0,
        "Unknown": np.nan,
    })

    #6. Estandarizar Intake Condition (corrige typos y dobles espacios) + flag de vulnerabilidad
    df["Intake Condition"] = (
        df["Intake Condition"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"ILL MODERATETE": "ILL MODERATE"})
    )

    condiciones_vulnerables = {
        "UNDER AGE/WEIGHT",
        "ILL MILD", "ILL MODERATE", "ILL SEVERE",
        "INJURED MILD", "INJURED MODERATE", "INJURED SEVERE",
        "AGED", "WELFARE SEIZURES",
    }
    df["es_vulnerable"] = df["Intake Condition"].isin(condiciones_vulnerables).astype(int)
    df["Intake Subtype"] = df["Intake Subtype"].fillna("UNKNOWN").astype(str).str.upper().str.strip()

    #7. Indicador limpio de supervivencia (corrige los casos DISPOSAL marcados como vivos en was_outcome_alive)
    df["sobrevivio"] = np.where(
        df["outcome_is_dead"], 0.0,
        np.where(df["outcome_is_alive"] & df["Outcome Type"].notna(), 1.0, np.nan),
    )
    df["was_outcome_alive"] = df["was_outcome_alive"].fillna(0).astype(int)

    #8. Depuración geográfica y Bounding Box de Long Beach y alrededores
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    df = df[
        df["latitude"].between(*BBOX_LAT) &
        df["longitude"].between(*BBOX_LON)
    ].copy()

    #9. Auditoría real de la Trampa del Refugio y coordenadas por defecto
    lat_r = df["latitude"].round(4)
    lon_r = df["longitude"].round(4)

    mask_refugio_real = (
        ((lat_r == 33.8096) & (lon_r == -118.0826)) |
        ((lat_r == 33.8093) & (lon_r == -118.0826)) |
        df["Crossing"].astype(str).str.contains(r"7700.*SPRING", case=False, regex=True, na=False) |
        df["Intake Subtype"].isin(["AT SHELTER", "ABAN SHLTR", "BORN@SHELT"])
    )
    mask_centroide_lb = (lat_r == 33.7701) & (lon_r == -118.1937)
    mask_crossing_nulo = df["Crossing"].isna()

    df["es_coord_valida_mapa"] = ~(mask_refugio_real | mask_centroide_lb | mask_crossing_nulo)

    cols_finales = [
        "Kennel ID", "Animal ID", "Animal Type",
        "Sex", "sexo_biologico", "esta_esterilizado",
        "DOB", "Intake Date", "Outcome Date", "anio_ingreso", "mes_ingreso", "edad_meses",
        "Intake Condition", "es_vulnerable", "Intake Subtype",
        "Outcome Type", "intake_duration", "was_outcome_alive", "sobrevivio",
        "Jurisdiction", "Crossing", "latitude", "longitude", "es_coord_valida_mapa",
    ]
    df_out = df[cols_finales].reset_index(drop=True)
    if en_espanol:
        df_out = df_out.rename(columns=RENOMBRAR_REFUGIO_ES)
    return df_out


def limpiar_dataset_clinicas(ruta_csv: str) -> pd.DataFrame:
    """Filtra y depura el directorio de clínicas veterinarias para el área de Long Beach."""
    df_raw = pd.read_csv(ruta_csv, low_memory=False)

    df = df_raw[
        df_raw["state"].astype(str).str.strip().str.upper().isin(["CA", "CALIFORNIA"])
    ].copy()

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[
        df["latitude"].between(*BBOX_LAT) &
        df["longitude"].between(*BBOX_LON)
    ].copy()

    for col in ["name", "city", "state", "address"]:
        df[col] = df[col].astype(str).str.strip()

    #excluir hospital equino y oficina de control animal
    df = df[~df["name"].str.contains(r"Equine|Animal Care Contrl", case=False, regex=True, na=False)].copy()

    #eliminar duplicados por nombre+dirección, coordenadas exactas y duplicado mal geocodificado (zip 90810)
    df = df.drop_duplicates(subset=["name", "address"], keep="first")
    df = df.drop_duplicates(subset=["latitude", "longitude"], keep="first")
    df = df[~(
        df["address"].str.contains("2251 E Carson St", case=False) &
        (df["postal_code"].astype(str) == "90810")
    )].copy()

    cols = ["name", "city", "state", "address", "postal_code", "latitude", "longitude"]
    return df[cols].reset_index(drop=True)


def preparar_veterinarias_long_beach(ruta_csv: str) -> pd.DataFrame:
    """Estandariza al esquema en español el padrón municipal de veterinarias de Long Beach."""
    df_raw = pd.read_csv(ruta_csv, low_memory=False)
    return df_raw.rename(columns=RENOMBRAR_VET_LB_ES)


def ejecutar_pipeline_limpieza(carpeta_raw: str, carpeta_salida: str) -> dict:
    """Ejecuta la limpieza completa desde data/raw y guarda los CSV procesados."""
    os.makedirs(carpeta_salida, exist_ok=True)

    shelter_clean = limpiar_dataset_refugio(
        os.path.join(carpeta_raw, "animal-shelter-intakes-and-outcomes.csv"),
        en_espanol=True,
    )
    vet_lb_es = preparar_veterinarias_long_beach(
        os.path.join(carpeta_raw, "veterinarias_long_beach_filtradas.csv")
    )
    vet_usa_clean = limpiar_dataset_clinicas(
        os.path.join(carpeta_raw, "animal_hospitals_usa.csv")
    )

    shelter_clean.to_csv(
        os.path.join(carpeta_salida, "shelter_clean_perros_gatos_espanol.csv"), index=False
    )
    vet_lb_es.to_csv(
        os.path.join(carpeta_salida, "veterinarias_long_beach_espanol.csv"), index=False
    )
    vet_usa_clean.to_csv(
        os.path.join(carpeta_salida, "vet_clinics_clean_long_beach.csv"), index=False
    )

    return {
        "refugio": shelter_clean,
        "veterinarias_lb": vet_lb_es,
        "veterinarias_usa": vet_usa_clean,
    }


if __name__ == "__main__":
    res = ejecutar_pipeline_limpieza("data/raw", "data/processed")
    print(f"Refugio limpio guardado: {len(res['refugio']):,} filas x {res['refugio'].shape[1]} columnas")
    print(f"Veterinarias LB guardadas: {len(res['veterinarias_lb']):,} registros")
    print(f"Clínicas USA limpias guardadas: {len(res['veterinarias_usa']):,} registros")
