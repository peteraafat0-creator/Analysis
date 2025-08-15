"""
Configuration settings for the Advanced Data Analysis Dashboard
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Application Configuration
APP_TITLE = os.getenv('APP_TITLE', 'Advanced Data Analysis Dashboard')
APP_DESCRIPTION = os.getenv('APP_DESCRIPTION', 'Comprehensive reporting and analysis platform')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# File Upload Configuration
MAX_FILE_SIZE_MB = 200
ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']

# Data Processing Configuration
DATE_FORMATS = [
    '%Y-%m-%d',
    '%d/%m/%Y',
    '%m/%d/%Y',
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M:%S'
]

# Visualization Configuration
DEFAULT_COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

CHART_HEIGHT = 400
CHART_WIDTH = 800

# Dashboard Configuration
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 1200