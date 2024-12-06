from fastapi import FastAPI, HTTPException, APIRouter
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
model = joblib.load('/var/www/gis_api_generacional/modelo_combustible_abarrotes.pkl')
nse_mapping = joblib.load('/var/www/gis_api_generacional/nse_mapping_abarrotes.pkl')

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
async def predecir_ventas(lon: float, lat: float, distancia: int = _distancia):
    """
    Predice ventas potenciales de abarrotes en una manzana específica basada en la ubicación.

    Parámetros:
        lon (float): Longitud de la ubicación.
        lat (float): Latitud de la ubicación.

    Retorno:
        JSON: Predicción del modelo.
    """
    try:
        # Crear un punto con las coordenadas proporcionadas
        punto = gpd.GeoDataFrame(
            {'geometry': [Point(lon, lat)]}, crs="EPSG:4326"
        )

        # Transformar CRS a métrico para realizar cálculos espaciales
        metric_crs = 'EPSG:6372'
        punto_metric = punto.to_crs(metric_crs)
        manzanas_gdf_metric = manzanas_gdf.to_crs(metric_crs)

        # Buscar la manzana más cercana
        manzanas_gdf_metric['distance'] = manzanas_gdf_metric.geometry.distance(punto_metric.iloc[0].geometry)
        manzana_cercana = manzanas_gdf_metric.loc[manzanas_gdf_metric['distance'].idxmin()]

        # Extraer datos de la manzana encontrada
        pob_tot = int(manzana_cercana['pob_tot'])  # Conversión explícita a int
        nse = manzana_cercana['nse']

        # Calcular abarrotes cercanos
        comercios_gdf_metric = comercios_gdf.to_crs(metric_crs)
        abarrotes_cercanas = int((comercios_gdf_metric.geometry.distance(punto_metric.iloc[0].geometry) <= distancia).sum())

        # Calcular población cercana
        poblacion_cercana = int(manzanas_gdf_metric.loc[
            manzanas_gdf_metric.geometry.distance(punto_metric.iloc[0].geometry) <= distancia, 'pob_tot'
        ].sum())

        # Codificar NSE
        nse_encoded = nse_mapping.get(nse, 0)  # Mapeo con valor por defecto 0 si el NSE no existe

        # Preparar datos para predicción
        X_pred = pd.DataFrame([{
            'abarrotes_cercanas': abarrotes_cercanas,
            'pob_tot': pob_tot,
            'poblacion_cercana': poblacion_cercana,
            'NSE_encoded': nse_encoded
        }]).fillna(0)

        # Realizar la predicción
        prediccion = model.predict(X_pred)
        return {
            "prediction": float(prediccion[0]),  # Asegurando que sea float
            "poblacion": pob_tot,
            "poblacion_cercana": poblacion_cercana,
            "NSE": nse
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    
@app.get("/nse_around")
async def obtener_nse_around(lon: float, lat: float, distancia: int = _distancia):
    """
    Retorna las manzanas alrededor de un punto (lon, lat) dentro del doble de la distancia especificada,
    incluyendo el NSE y la población total por cada manzana.

    Parámetros:
        lon (float): Longitud de la ubicación.
        lat (float): Latitud de la ubicación.
        distancia (int): Distancia base en metros. Se utilizará el doble de este valor.

    Retorno:
        JSON: Lista de manzanas con su NSE y población.
    """
    try:
        # Crear punto base
        punto = gpd.GeoDataFrame({'geometry': [Point(lon, lat)]}, crs="EPSG:4326")

        # Transformar a CRS métrico
        metric_crs = 'EPSG:6372'
        punto_metric = punto.to_crs(metric_crs)
        manzanas_gdf_metric = manzanas_gdf.to_crs(metric_crs)

        # Filtrar manzanas dentro del doble de la distancia indicada
        distancia_doble = 2 * distancia
        manzanas_cercanas = manzanas_gdf_metric[
            manzanas_gdf_metric.geometry.distance(punto_metric.iloc[0].geometry) <= distancia_doble
        ]

        # Agrupar por NSE y sumar poblaciones
        agrupado = manzanas_cercanas.groupby('nse')['pob_tot'].sum().reset_index(name='pob_total_nse')
        poblacion_total = agrupado['pob_total_nse'].sum()


        # Crear diccionario para acceder al porcentaje por NSE
        porcentaje_por_nse = {
            row['nse']: (row['pob_total_nse'] / poblacion_total * 100 if poblacion_total > 0 else 0)
            for _, row in agrupado.iterrows()
        }

        # Construir resultado con la información por manzana
        resultado = []
        for _, row in manzanas_cercanas.iterrows():
            resultado.append({
                "nse": row['nse'],
                "poblacion_total": int(row['pob_tot']),
                "porcentaje": porcentaje_por_nse.get(row['nse'], 0),
                "longitud": float(row['longitud']),
                "latitud": float(row['latitud'])
            })

        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener NSE: {str(e)}")
