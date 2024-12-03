import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import pandas as pd


# Configuración de conexión a PostgreSQL
db_host = 'localhost'
db_port = '5432'
db_name = 'datos_comerciales'
db_user = 'user_datos'
db_password = 'cr0n0smac'
#Crea los datos



# Crear la cadena de conexión
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
# Configuración de conexión a PostgreSQL
engine = create_engine(connection_string)

# Carga de datos con especificación de codificación y low_memory=False
manzanas = pd.read_csv('MANZANAS ZAPOPAN.csv', encoding='latin-1', low_memory=False)
# Cargar datos desde el archivo CSV
comercios = pd.read_csv('NE COMERCIOS ZAPOPAN.csv', encoding='latin-1', low_memory=False)
# Agregar la columna 'manzana_id' al DataFrame
manzanas['manzana_id'] = range(1, len(manzanas) + 1)
print("cargo las bases de datos ")
# Subir datos a la base de datos PostgreSQL
# Subir datos a la base de datos PostgreSQL en chunks para evitar consumo excesivo de memoria
chunk_size = 1000  # Puedes ajustar este valor según los recursos del servidor

manzanas.to_sql('catalogo_manzanas', engine, if_exists='replace', index=False, chunksize=chunk_size)
print("guardo manzanas en chunks")

comercios.to_sql('catalogo_comercios', engine, if_exists='replace', index=False, chunksize=chunk_size)
print("guardo comercios en chunks")

#print(manzanas.head())
#print(manzanas.info())
# Nombre de la tabla que deseas modificar
tablas = ['catalogo_manzanas', 'catalogo_comercios']
#hace un ciclo
for table_name in tablas:

    try:
        print("carga la tabla {table_name}")
        # Conectar a la base de datos
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        conn.autocommit = True  # Habilitar autocommit
        cursor = conn.cursor()

        # Obtener los nombres de columnas actuales
        cursor.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;"), [table_name])
        columns = cursor.fetchall()

        for column in columns:
            old_column_name = column[0]  # Nombre actual de la columna
            new_column_name = old_column_name.lower()  # Convertir a minúsculas

            # Si ya está en minúsculas, omitir
            if old_column_name == new_column_name:
                continue

            # Renombrar columna
            query = sql.SQL("ALTER TABLE {table} RENAME COLUMN {old_column} TO {new_column};").format(
                table=sql.Identifier(table_name),
                old_column=sql.Identifier(old_column_name),
                new_column=sql.Identifier(new_column_name)
            )
            cursor.execute(query)
            print(f"Renombrada columna: {old_column_name} -> {new_column_name}")

        print("Todos los nombres de columnas se han transformado a minúsculas.")

    except psycopg2.Error as e:
        print(f"Error al conectar o modificar la tabla: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
