from tensorflow.keras.optimizers import RMSprop

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
    )
    model.summary()

    # create dataset
    ds = createds()

    # train model
    model.fit(
        x=ds,
        epochs=EPOCHS,
    )
