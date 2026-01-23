
from pathlib import Path
import pandas as pd

UCI_RED_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

def download_red_wine_quality(
    url=UCI_RED_WINE_URL,
    data_dir=Path("data"),
    filename="winequality-red.csv",
    force_download=False,
):
    data_dir.mkdir(parents=True, exist_ok=True)
    local_path = data_dir / filename

    if local_path.exists() and not force_download:
        return local_path

    df = pd.read_csv(url, sep=";")
    df.to_csv(local_path, index=False)

    return local_path


def load_red_wine_quality(local_path):
    return pd.read_csv(local_path)
