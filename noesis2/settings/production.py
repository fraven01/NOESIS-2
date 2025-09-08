from .base import *

# Production overrides
DEBUG = False

# Static files collection target
STATIC_ROOT = BASE_DIR / "staticfiles"

# Use JSON logging in production
LOGGING['root']['handlers'] = ['json_console']
LOGGING['loggers']['django.request'] = {
    'handlers': ['mail_admins', 'json_console'],
    'level': 'ERROR',
    'propagate': False,
}

