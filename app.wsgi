import sys
import os

sys.path.insert(0, '/var/www/gis_api_generacional')
activate_this = '/var/www/gis_api_generacional/venv/bin/activate_this.py'
exec(open(activate_this).read(), {'__file__': activate_this})

from app import app as application  # Ajusta 'app' al nombre de tu aplicación
app.wsgi