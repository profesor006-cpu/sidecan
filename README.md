# SIDECAN — Sistema de Integración de Datos, Estadística y Control Animal
**Proyecto Final del Módulo Python — Samsung Innovation Campus (SIC 2025/2026)**  

---
## 1. Estructura del Proyecto

```text
SIDECAN/
├── main.py                              # Orquestador del pipeline completo
├── app.py                               # (Próximamente) Interfaz interactiva en Streamlit
├── requirements.txt                     # Dependencias del proyecto
├── .gitignore
├── data/
│   ├── raw/                             # Bases crudas originales (refugio + clínicas USA/LB)
│   └── processed/                       # Tablas limpias, distancias y rankings calculados
├── src/
│   ├── __init__.py
│   ├── data_processing.py               # Limpieza y auditoría inicial
│   ├── analisis_espacial.py             # Haversine en NumPy + Índice Zonal + Ranking
│   └── estadisticas.py                  # Demografía, estacionalidad, FIELD vs OTC y Chi2
├── reports/                             # (Próximamente) Imgenes generadas
└── entregables_sic/                     # PPT oficial Samsung, Formulario SIC y Video
```

---

## 2. Instalación y Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el pipeline completo (análisis espacial + estadística)
python main.py

```
