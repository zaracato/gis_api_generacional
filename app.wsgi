import os
import sys

# Ruta del entorno virtual
venv_path = '/var/www/gis_api_generacional/venv'
sys.path.insert(0, os.path.join(venv_path, 'lib', 'python3.12', 'site-packages'))

# Agregar la ruta del proyecto
sys.path.append('/var/www/gis_api_generacional')

# Configurar la aplicación 
from api import app as application
