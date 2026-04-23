import os
import pandas as pd
from pathlib import Path


def collect_audio_files(dataset_dir: Path, valid_extensions=(".wav",)):
    """
    Veri setindeki ses dosyalarının yollarını ve etiketlerini toplar.
    Her genre klasörü ayrı bir sınıf kabul edilir.
    """
    records = []

    for genre_folder in sorted(dataset_dir.iterdir()):
        if genre_folder.is_dir():
            genre_name = genre_folder.name
            for audio_file in sorted(genre_folder.iterdir()):
                if audio_file.suffix.lower() in valid_extensions:
                    records.append({
                        "file_path": str(audio_file),
                        "genre": genre_name,
                        "file_name": audio_file.name
                    })

    df = pd.DataFrame(records)
    return df
