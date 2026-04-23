from sklearn.base import clone
import time


def train_model(model, X_train, y_train):
    """
    Verilen modeli eğitir ve süreyi ölçer.
    """
    model_instance = clone(model)

    start_time = time.time()
    model_instance.fit(X_train, y_train)
    training_time = time.time() - start_time

    return model_instance, training_time
