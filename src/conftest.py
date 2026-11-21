"""
Asegura que la raíz del repo esté en sys.path para que
`from src.preprocessing.clean import ...` funcione sin importar desde qué
directorio se invoque pytest (el repo no está empaquetado/instalado).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
