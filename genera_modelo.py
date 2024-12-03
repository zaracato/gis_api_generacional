import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sqlalchemy import create_engine
import joblib
from sqlalchemy import text
# Configurar los detalles de conexión a PostgreSQL
db_host = 'localhost'
db_port = '5432'
db_name = 'datos_comerciales'
db_user = 'user_datos'
db_password = 'cr0n0smac'

# Crear la cadena de conexión
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
# Configuración de conexión a PostgreSQL
engine = create_engine(connection_string)

# Carga de datos con especificación de codificación y low_memory=False
#manzanas = pd.read_csv('MANZANAS ZAPOPAN.csv', encoding='latin-1', low_memory=False)
# Cargar datos desde el archivo CSV
#manzanas = pd.read_csv('MANZANAS ZAPOPAN.csv', encoding='latin-1', low_memory=False)
# Agregar la columna 'manzana_id' al DataFrame
#manzanas['manzana_id'] = range(1, len(manzanas) + 1)

# Subir datos a la base de datos PostgreSQL
#manzanas.to_sql('catalogo_manzanas', engine, if_exists='replace', index=False)

#print(manzanas.head())
#print(manzanas.info())
with engine.connect() as conn:
    # Obtener manzanas
    manzanas = pd.read_sql('SELECT longitud, latitud, pob_tot, nse, beb_noalcoh FROM catalogo_manzanas WHERE  beb_noalcoh > 0;', conn)

    # Obtener comercios
    query = text("SELECT longitude, latitude, tipo FROM catalogo_comercios WHERE tipo ILIKE :tipo;")
    comercios = pd.read_sql(query, conn, params={"tipo": "%COMERCIO AL POR MENOR EN TIENDAS DE ABARROTES, ULTRAMARINOS Y MISCELANEAS%"})
# Convertir a GeoDataFrames
manzanas_gdf = gpd.GeoDataFrame(
    manzanas, geometry=gpd.points_from_xy(manzanas['longitud'], manzanas['latitud']), crs="EPSG:4326"
)

comercios_gdf = gpd.GeoDataFrame(
    comercios, geometry=gpd.points_from_xy(comercios['longitude'], comercios['latitude']), crs="EPSG:4326"
)

#print("Datos de 'manzanas' subidos a la base de datos.")
## Leer los datos desde PostgreSQL
#query = """
#SELECT * FROM catalogo_manzanas
#"""
#manzanas_db = pd.read_sql(query, engine)

#comercios = pd.read_csv('NE COMERCIOS ZAPOPAN.csv', encoding='latin-1', low_memory=False)
#personas = pd.read_csv('PERSONAS ZAPOPAN.csv', encoding='latin-1', low_memory=False)
_distancia = 500
# Verificar que las columnas necesarias existen
#BEB_NOALCOH	REF_COLA	REF_SABOR	AGUA_EMB	AGUA_GARRA	JUGOS	CONCENTRADOS	OTRAS_BNA
#for df_name, df, required_columns in [
#    ('manzanas', manzanas, ['LONGITUD', 'LATITUD', 'POB_TOT', 'NSE',  'BEB_NOALCOH','REF_COLA','REF_SABOR',	'AGUA_EMB',	'AGUA_GARRA',	'JUGOS','CONCENTRADOS',	'OTRAS_BNA']),
#    ('comercios', comercios, ['Longitude', 'Latitude', 'TIPO']),
#    ('personas', personas, ['Longitud', 'Latitud'])
#]:
#   missing_cols = [col for col in required_columns if col not in df.columns]
#    if missing_cols:
#        print(f"El DataFrame '{df_name}' no contiene las siguientes columnas necesarias: {missing_cols}")
#        sys.exit()
# Conversión a GeoDataFrames
#manzanas_gdf = gpd.GeoDataFrame(
#    manzanas_db, 
#    geometry=gpd.points_from_xy(manzanas_db['LONGITUD'], manzanas_db['LATITUD'])
#)

#comercios_gdf = gpd.GeoDataFrame(
#    comercios, 
#    geometry=gpd.points_from_xy(comercios['Longitude'], comercios['Latitude'])
#)

#personas_gdf = gpd.GeoDataFrame(
#    personas, 
#    geometry=gpd.points_from_xy(personas['Longitud'], personas['Latitud'])
#)
# Asegurarse de que las geometrías están en CRS WGS84 (EPSG:4326)
manzanas_gdf.set_crs(epsg=4326, inplace=True)
comercios_gdf.set_crs(epsg=4326, inplace=True)
#personas_gdf.set_crs(epsg=4326, inplace=True)

# Convertir a una proyección métrica (por ejemplo, UTM zona 13N para Zapopan)
# Asegúrate de que este EPSG es el correcto para tu área
metric_crs = 'EPSG:6372'  # Puedes cambiar esto a 'EPSG:32613' si corresponde
manzanas_gdf = manzanas_gdf.to_crs(metric_crs)
comercios_gdf = comercios_gdf.to_crs(metric_crs)
#personas_gdf = personas_gdf.to_crs(metric_crs)
# Filtrar abarrotes utilizando la columna 'TIPO'
#abarrotes = comercios_gdf[comercios_gdf['TIPO'].str.contains('COMERCIO AL POR MENOR EN TIENDAS DE ABARROTES, ULTRAMARINOS Y MISCELANEAS', case=False, na=False)]

# Verificar cuántas abarrotes se han identificado
#print(f"Número de tiendas identificadas: {len(abarrotes)}")
# Función para calcular número de abarrotes en un radio de 3 km
def contar_abarrotes(row):
    distancia = comercios_gdf.geometry.distance(row.geometry)
    return (distancia <= _distancia).sum()

# Aplicar la función a cada manzana
print("Calculando el número de abarrotes cercanas para cada manzana...")
manzanas_gdf['abarrotes_cercanas'] = manzanas_gdf.apply(contar_abarrotes, axis=1)
# Función para calcular la población cercana
def calcular_poblacion_cercana(row):
    distancia = manzanas_gdf.geometry.distance(row.geometry)
    poblacion_cercana = manzanas_gdf.loc[distancia <= _distancia, 'pob_tot'].sum()
    return poblacion_cercana

print("Calculando la población cercana para cada manzana...")
manzanas_gdf['poblacion_cercana'] = manzanas_gdf.apply(calcular_poblacion_cercana, axis=1)
# Asegúrate de que el diccionario nse_mapping está definido
nse_mapping = {
    'E': 1,
    'D': 2,
    'D+': 3,
    'C-': 4,
    'C': 5,
    'C+': 6,
    'A/B': 7,
    'DES': 0  # Ajusta según corresponda en tu contexto
}

# Aplicar el mapeo a 'NSE' en manzanas_gdf
manzanas_gdf['NSE_encoded'] = manzanas_gdf['nse'].map(nse_mapping)
# Definir las características para el modelo
features = ['abarrotes_cercanas', 'pob_tot', 'poblacion_cercana', 'NSE_encoded']


# Verificar que las columnas existen en 'manzanas_gdf'
faltantes = set(features) - set(manzanas_gdf.columns)
if faltantes:
    print(f"Las siguientes columnas faltan en 'manzanas_gdf': {faltantes}")
    sys.exit()

# Preparación de datos para el modelo
X = manzanas_gdf[features]
y = manzanas_gdf['beb_noalcoh']  # BEB_NOALCOHAsegúrate de que 'COMBUSTIBLE' es la variable objetivo correcta

# Combinar 'X' e 'y' y eliminar filas con valores faltantes
datos = pd.concat([X, y], axis=1).dropna()
X = datos[features]
y = datos['beb_noalcoh']
# Verificar los tipos de datos antes de la conversión
print("Tipos de datos antes de la conversión:")
print(X.dtypes)

# Verificar los valores únicos en 'NSE'
print("Valores únicos en 'NSE':")
print(X['NSE_encoded'].unique())
print(X['NSE_encoded'].head(20))

# Verificar valores faltantes en 'NSE_encoded'
num_faltantes_nse = X['NSE_encoded'].isnull().sum()
print(f"Número de valores faltantes en 'NSE_encoded' después del mapeo: {num_faltantes_nse}")

# Manejar valores faltantes
# Opción 1: Eliminar filas con 'NSE_encoded' faltante
X = X.dropna(subset=['NSE_encoded'])
y = y.loc[X.index]

# Opción 2: Imputar valores faltantes (si es apropiado)
# X['NSE_encoded'].fillna(X['NSE_encoded'].median(), inplace=True)

# Verificar si quedan columnas de tipo 'object'
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"La columna '{col}' es de tipo 'object'. Intentando convertir a numérico.")
        X[col] = pd.to_numeric(X[col], errors='coerce')
        # Manejar valores faltantes si es necesario
        num_faltantes = X[col].isnull().sum()
        print(f"Número de valores faltantes en '{col}' después de la conversión: {num_faltantes}")
        X = X.dropna(subset=[col])
        y = y.loc[X.index]

# Verificar los tipos de datos después de la conversión
print("Tipos de datos después de la conversión:")
print(X.dtypes)
print(f"Número de filas en X: {X.shape[0]}")
print(f"Número de filas en y: {y.shape[0]}")
# Verificar valores faltantes en otras columnas
print("Valores faltantes por columna:")
print(X[features].isnull().sum())

# Si hay valores faltantes en otras columnas, manejarlos adecuadamente
# Por ejemplo, eliminar filas con valores faltantes
X = X.dropna()
y = y.loc[X.index]

# División de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2)

# Entrenamiento del modelo
from xgboost import XGBRegressor

print("Entrenando el modelo...")
model = XGBRegressor()
model.fit(X_train, y_train)

# Evaluación del modelo
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE del modelo: {rmse}")
print("acuracy del modelo: ",model.score(X_test, y_test))

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_combustible_abarrotes.pkl')
print("Modelo guardado exitosamente.")

# Guardar las transformaciones en un pipeline
joblib.dump(nse_mapping, 'nse_mapping_abarrotes.pkl')
