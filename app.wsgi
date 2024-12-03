import os
import sys

# Añadir el entorno virtual
venv_path = '/var/www/gis_api_generacional/venv'
sys.path.insert(0, os.path.join(venv_path, 'lib', 'python3.12', 'site-packages'))

# Configurar la aplicación
from your_application import app as application