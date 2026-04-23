from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/MusicGenreClassification")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"

RANDOM_STATE = 42
SAMPLE_RATE = 22050
TEST_SIZE = 0.20
VAL_SIZE = 0.10

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]
