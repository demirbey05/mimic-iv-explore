import pandas as pd
from pathlib import Path


def load_mimic_tables(data_path: str = "../data/") -> dict[str, pd.DataFrame]:
    """
    Load all MIMIC-IV tables from hosp and icu folders.

    Args:
        data_path: Path to the data directory containing hosp and icu folders.

    Returns:
        Dictionary with table names as keys and DataFrames as values.
        Keys are prefixed with folder name (e.g., 'hosp.admissions', 'icu.chartevents').
    """
    data_dir = Path(data_path)
    tables = {}

    for folder in ["hosp", "icu"]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue

        for file in folder_path.glob("*.csv.gz"):
            table_name = f"{folder}.{file.stem.replace('.csv', '')}"
            tables[table_name] = pd.read_csv(file, compression="gzip")

    return tables
