import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_labels(y):
    """
    Genre etiketlerini sayısal etiketlere dönüştürür.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def scale_features(X_train, X_val=None, X_test=None):
    """
    Özellikleri standardize eder.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
