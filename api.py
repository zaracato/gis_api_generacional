from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import joblib
from shapely.geometry import Point

app = FastAPI()

# Configurar conexión a PostgreSQL
db_host = 'localhost'
db_port = '5432'
db_name = 'datos_comerciales'
db_user = 'user_datos'
db_password = 'cr0n0smac'
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# Cargar el modelo entrenado y el mapeo
model = joblib.load('modelo_combustible_abarrotes.pkl')
nse_mapping = joblib.load('nse_mapping_abarrotes.pkl')

# Cargar los datos necesarios
with engine.connect() as conn:
    # Cargar comercios
    query_comercios = text("SELECT longitude, latitude, tipo FROM catalogo_comercios WHERE tipo ILIKE :tipo")
    params = {"tipo": "%COMERCIO AL POR MENOR%"}
    comercios = pd.read_sql(query_comercios, conn, params=params)
    comercios_gdf = gpd.GeoDataFrame(
        comercios, geometry=gpd.points_from_xy(comercios['longitude'], comercios['latitude']), crs="EPSG:4326"
    )

    # Cargar manzanas
    query_manzanas = "SELECT longitud, latitud, pob_tot, nse FROM catalogo_manzanas WHERE beb_noalcoh > 0"
    manzanas = pd.read_sql(query_manzanas, conn)
    manzanas_gdf = gpd.GeoDataFrame(
        manzanas, geometry=gpd.points_from_xy(manzanas['longitud'], manzanas['latitud']), crs="EPSG:4326"
    )

# Radio en metros para el cálculo
_distancia = 500


@app.get("/")
def root():
    return {"message": "Bienvenido a la API de predicción de ventas de abarrotes"}


@app.get("/predict")
def predecir_ventas(lon: float, lat: float, pob_tot: int, nse: str):
    """
    Predice ventas potenciales de abarrotes en una manzana específica.

    Parámetros:
        lon (float): Longitud de la manzana.
        lat (float): Latitud de la manzana.
        pob_tot (int): Población total de la manzana.
        nse (str): Nivel socioeconómico.

    Retorno:
        JSON: Predicción del modelo.
    """
    try:
        # Crear un GeoDataFrame para la manzana
        manzana = gpd.GeoDataFrame({
            'longitud': [lon],
            'latitud': [lat],
            'pob_tot': [pob_tot],
            'nse': [nse]
        }, geometry=[Point(lon, lat)], crs="EPSG:4326")

        # Transformar CRS a métrico
        metric_crs = 'EPSG:6372'
        manzana = manzana.to_crs(metric_crs)
        comercios_gdf_metric = comercios_gdf.to_crs(metric_crs)
        manzanas_gdf_metric = manzanas_gdf.to_crs(metric_crs)

        # Calcular abarrotes cercanos
        manzana['abarrotes_cercanas'] = manzana.apply(
            lambda row: (comercios_gdf_metric.geometry.distance(row.geometry) <= _distancia).sum(),
            axis=1
        )

        # Calcular población cercana
        manzana['poblacion_cercana'] = manzana.apply(
            lambda row: manzanas_gdf_metric.loc[
                manzanas_gdf_metric.geometry.distance(row.geometry) <= _distancia, 'pob_tot'
            ].sum(),
            axis=1
        )

        # Codificar NSE
        manzana['NSE_encoded'] = manzana['nse'].map(nse_mapping)

        # Preparar datos para predicción
        features = ['abarrotes_cercanas', 'pob_tot', 'poblacion_cercana', 'NSE_encoded']
        X_pred = manzana[features].fillna(0)

        # Realizar la predicción
        prediccion = model.predict(X_pred)

        return {"prediction": prediccion[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
