from pathlib import Path

# Ensure important sub-directories exist when the package is imported
PACKAGE_ROOT = Path(__file__).resolve().parent
(PACKAGE_ROOT / "evaluation").mkdir(exist_ok=True)