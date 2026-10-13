"""Task 5 - training with class weights and early stopping."""

import time

from tensorflow.keras.callbacks import EarlyStopping

from . import config


def make_early_stopping(patience=config.PATIENCE):
    """Stop when val_loss stops improving and roll back to the best epoch."""
    return EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)


def train(model, train_ds, val_ds, class_weight, epochs=config.MAX_EPOCHS, verbose=2):
    """Fit the model; returns (history, seconds_taken).

    class_weight makes mistakes on the rarer NORMAL class count more in the loss.
    The epoch budget is only a ceiling - early stopping decides the real cutoff.
    """
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[make_early_stopping()],
        verbose=verbose,
    )
    return history, time.time() - start
