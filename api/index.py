import sys
import os

# Ensure the parent directory is in sys.path so app and analyzer modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
