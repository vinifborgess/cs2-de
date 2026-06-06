"""(raw -> bronze -> silver -> gold)."""
from .bronze import parse_demo_to_bronze
from .silver import bronze_to_silver, load_silver_lake
from .gold import run_gold

__all__ = ["parse_demo_to_bronze", "bronze_to_silver", "load_silver_lake", "run_gold"]
__version__ = "0.1.0"
