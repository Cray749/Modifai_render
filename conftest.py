# conftest.py — makes pytest discover modifai as a package from d:\Far away\
import sys
import os

# Ensure the project root is on the path so `import modifai` works
sys.path.insert(0, os.path.dirname(__file__))
