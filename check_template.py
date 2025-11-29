
import os
import django
from django.conf import settings
from django.template import Engine

# Configure Django settings
settings.configure(
    TEMPLATES=[{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.abspath('theme/templates')],
        'APP_DIRS': True,
    }],
    INSTALLED_APPS=['theme', 'django.contrib.staticfiles'],
)
django.setup()

def check_template(template_path):
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        engine = Engine.get_default()
        template = engine.from_string(content)
        print(f"Template {template_path} is valid.")
    except Exception as e:
        print(f"Template error in {template_path}:")
        print(e)

check_template('theme/templates/theme/rag_tools.html')
