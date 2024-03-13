import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=(), patience=10):
        super(tf.keras.callbacks.Callback, self).__init__()

        self.patience = patience
        self.X_val, self.y_val = validation_data
        self.best_f1 = -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        y_pred = np.round(y_pred)
        current_profit = calculate_profit(self.y_val, y_pred)

        if current_profit > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Early stopping due to no improvement in F1 score.")
