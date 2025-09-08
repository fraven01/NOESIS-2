from .base import *

# Production overrides
DEBUG = False

# Use JSON logging in production
LOGGING['root']['handlers'] = ['json_console']

