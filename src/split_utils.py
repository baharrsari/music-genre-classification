from sklearn.model_selection import train_test_split


def create_train_val_test_split(X, y, test_size=0.10, val_size=0.10, random_state=42):
    """
    Önce train+temp / test, sonra train / val ayırır.
    Stratified split kullanır.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
