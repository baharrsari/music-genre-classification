import numpy as np
import pandas as pd
import librosa


def extract_basic_features(file_path, sample_rate=22050, n_mfcc=13):
    """
    Tek bir ses dosyasından temel akustik özellikleri çıkarır.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)

    features = {}

    # Zero Crossing Rate - Sinyalin ne kadar sık sıfırı geçtiği.
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # Spectral Centroid - Sesin tiz, bas ağırlığı.
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))

    # Spectral Rolloff - Frekansların yayılım genişliği.
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

    # Chroma - 12 müzik notasından hangilerinin baskın olduğu.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"] = float(np.std(chroma))

    # RMS Energy - Sesin genel şiddeti.
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Tempo - Müziğin hızı.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    features["tempo"] = float(tempo[0])

    # MFCC - İnsan kulağının frekans algısını modelliyor.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return features
 
  #tek bir sabit boyutlu vektöre indirgerken hem ortalama davranışı
  #hem de değişkenliği koruduğu için mean + std  

def extract_features_from_metadata(df_metadata, sample_rate=22050, n_mfcc=13, verbose=True):
    """
    Metadata tablosundaki tüm ses dosyaları için feature extraction yapar.
    """
    records = []
    error_files = []

    total_files = len(df_metadata)

    for idx, row in df_metadata.iterrows():
        file_path = row["file_path"]
        file_name = row["file_name"]
        genre = row["genre"]

        try:
            feature_dict = extract_basic_features(
                file_path=file_path,
                sample_rate=sample_rate,
                n_mfcc=n_mfcc
            )

            feature_dict["file_path"] = file_path
            feature_dict["file_name"] = file_name
            feature_dict["genre"] = genre

            records.append(feature_dict)

            if verbose and (idx + 1) % 50 == 0:
                print(f"[INFO] {idx + 1}/{total_files} dosya işlendi.")

        except Exception as e:
            error_files.append({
                "file_path": file_path,
                "file_name": file_name,
                "genre": genre,
                "error": str(e)
            })

            if verbose:
                print(f"[HATA] Dosya işlenemedi: {file_name} | Hata: {e}")

    features_df = pd.DataFrame(records)
    return features_df, error_files
