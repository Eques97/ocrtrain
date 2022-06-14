import json

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from train.model import createModel
from train.cost import ctcloss
from train.dataset import createds
from train import (
    ALPHA,
    EPOCHS,
)

if __name__ == "__main__":

    # create model
    model = createModel()
    model.compile(
        optimizer=RMSprop(learning_rate=ALPHA),
        loss=ctcloss,
        metrics=None,
    )
    model.summary()

    # create dataset
    tds = createds()
    vds = createds(isval=True)

    # create callbacks
    ckptcb = ModelCheckpoint(
        filepath="ckpt/{epoch:03d}",
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=True,
    )
    callbacks = [
        ckptcb,
    ]

    # train model
    history = model.fit(
        x=tds,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=vds,
    )

    with open("history.json", mode="w") as f:
        json.dump(history.history, f, indent = 4)
