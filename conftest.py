# conftest.py
import sys, os

# Insert the absolute path to ./src at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

