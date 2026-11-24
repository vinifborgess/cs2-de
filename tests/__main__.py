"""``python -m tests`` runs every tests/test_*.py module."""
import importlib
import pkgutil
import sys

import tests
from tests._runner import run

code = 0
for mod in sorted(m.name for m in pkgutil.iter_modules(tests.__path__) if m.name.startswith("test_")):
    print(f"== tests.{mod}")
    code |= run(vars(importlib.import_module(f"tests.{mod}")))
sys.exit(code)
